"""Export the best student: merge LoRA into fp16 weights, convert to GGUF, verify outputs.

    python -m sensei_sql.train.export merge  --adapter models/adapters/<run>
    python -m sensei_sql.train.export gguf                       # needs a built llama.cpp
    python -m sensei_sql.train.export verify --adapter models/adapters/<run> [--gguf]

* merge  -> models/student-merged/   (base + adapter, fp16 safetensors + tokenizer)
* gguf   -> models/student-gguf/student-{f16,Q8_0,Q4_K_M}.gguf via third_party/llama.cpp
            (convert_hf_to_gguf.py + llama-quantize)
* verify -> greedy SQL on the first 50 dev questions must match between adapter+base and
            the merged model (vLLM), and, with --gguf, each GGUF (llama-server, CPU).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from sensei_sql.paths import MODELS, OUTPUTS, THIRD_PARTY

MERGED = MODELS / "student-merged"
GGUF_DIR = MODELS / "student-gguf"
LLAMA_CPP = THIRD_PARTY / "llama.cpp"
GGUF_QUANTS = ("Q8_0", "Q4_K_M")


def merge(adapter: Path, out: Path = MERGED) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from sensei_sql.eval.runner import resolve_model

    base = resolve_model("student").path
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float16)
    model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(adapter).save_pretrained(out)
    (out / "MERGED_FROM.json").write_text(json.dumps({"base": str(base), "adapter": str(adapter)}, indent=2) + "\n")
    print(f"[export] merged {adapter} into {out}")


def gguf(merged: Path = MERGED, out_dir: Path = GGUF_DIR) -> list[Path]:
    convert = LLAMA_CPP / "convert_hf_to_gguf.py"
    quantize = LLAMA_CPP / "build" / "bin" / "llama-quantize"
    if not quantize.exists():
        sys.exit(f"[export] {quantize} not found: build llama.cpp first (see README, 'CPU / GGUF')")
    out_dir.mkdir(parents=True, exist_ok=True)
    f16 = out_dir / "student-f16.gguf"
    subprocess.run([sys.executable, str(convert), str(merged), "--outfile", str(f16), "--outtype", "f16"], check=True)
    outs = []
    for q in GGUF_QUANTS:
        target = out_dir / f"student-{q}.gguf"
        subprocess.run([str(quantize), str(f16), str(target), q], check=True)
        outs.append(target)
    print(f"[export] wrote {', '.join(p.name for p in [f16, *outs])} to {out_dir}")
    return outs


def _vllm_predictions(model: str, adapter: str | None, limit: int) -> list[str]:
    """Run the eval harness in a subprocess (fresh GPU state) and return its pred_sql list."""
    run = f"verify-{Path(adapter).name if adapter else Path(model).name}-n{limit}"
    cmd = [sys.executable, "-m", "sensei_sql.eval", "--model", model, "--limit", str(limit), "--no-throughput", "--run", run]
    if adapter:
        cmd += ["--adapter", adapter]
    subprocess.run(cmd, check=True)
    rows = (OUTPUTS / "eval" / run / "predictions.jsonl").read_text().splitlines()
    return [json.loads(r)["pred_sql"] for r in rows]


def _gguf_predictions(path: Path, limit: int) -> list[str]:
    from sensei_sql.eval.data import load_split
    from sensei_sql.eval.runner import build_messages
    from sensei_sql.app.backends import LlamaServerBackend

    backend = LlamaServerBackend.launch(path)
    try:
        return [backend.generate_sql(build_messages(ex, [])).sql for ex in load_split("dev", limit)]
    finally:
        backend.close()


def verify(adapter: Path, limit: int, check_gguf: bool) -> dict:
    reference = _vllm_predictions("student", str(adapter), limit)
    report = {"limit": limit, "reference": f"student + {adapter.name} (vLLM)"}
    candidates = {"merged (vLLM)": lambda: _vllm_predictions(str(MERGED), None, limit)}
    if check_gguf:
        for q in GGUF_QUANTS:
            candidates[f"gguf {q} (llama.cpp CPU)"] = lambda q=q: _gguf_predictions(GGUF_DIR / f"student-{q}.gguf", limit)
    for name, fn in candidates.items():
        preds = fn()
        same = sum(a == b for a, b in zip(reference, preds))
        report[name] = {"same_sql": same, "rate": round(same / limit, 3)}
        print(f"[export] {name}: {same}/{limit} identical SQL", flush=True)
    out = OUTPUTS / "export" / "verify.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    report["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    out.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["merge", "gguf", "verify"])
    ap.add_argument("--adapter", type=Path)
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--gguf", action="store_true", help="verify: also check the GGUF exports")
    args = ap.parse_args()
    if args.step in ("merge", "verify") and not args.adapter:
        ap.error("--adapter is required")
    if args.step == "merge":
        merge(args.adapter)
    elif args.step == "gguf":
        gguf()
    else:
        verify(args.adapter, args.limit, args.gguf)


if __name__ == "__main__":
    main()
