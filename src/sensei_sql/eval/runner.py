"""Generate SQL for an evaluation split with vLLM and measure efficiency.

Two passes over the same prompts, both greedy:
  * latency pass, one question at a time (batch size 1, as in interactive use). These outputs
    are the saved predictions, so per-example ``latency_ms`` matches the scored SQL.
  * throughput pass, ``throughput_batch_size`` questions per ``generate`` call.
Peak GPU memory is sampled with NVML across model load and both passes.
"""

from __future__ import annotations

import json
import statistics
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from sensei_sql.core.prompts import Messages, answer_messages, extract_sql
from sensei_sql.eval.data import Example, schema_text
from sensei_sql.paths import CONFIGS, MODELS, ROOT


@dataclass
class ModelSpec:
    name: str
    path: Path
    vllm_args: dict = field(default_factory=dict)
    chat_template_kwargs: dict | None = None
    adapter: Path | None = None


def resolve_model(model: str, adapter: str | None = None) -> ModelSpec:
    """``model`` is a key of configs/models.yaml (``teacher``/``student``) or a local model dir,
    e.g. a merged fine-tuned student. Local dirs reuse the student's vLLM settings."""
    cfg = yaml.safe_load((CONFIGS / "models.yaml").read_text())
    if model in cfg:
        spec = cfg[model]
        path = MODELS / spec["local_dir"]
    else:
        path = Path(model).resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"{model!r} is neither a key in models.yaml nor a directory")
        spec = cfg["student"]
    return ModelSpec(
        name=model if model in cfg else path.name,
        path=path,
        vllm_args=dict(spec.get("vllm", {})),
        chat_template_kwargs=spec.get("chat_template_kwargs"),
        adapter=Path(adapter).resolve() if adapter else None,
    )


def dir_size_gib(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file() and ".cache" not in f.parts) / 2**30


def build_messages(ex: Example, few_shot: list[Example]) -> Messages:
    shots = [(schema_text(s.db_path), s.question, extract_sql(s.gold_sql)) for s in few_shot]
    return answer_messages(schema_text(ex.db_path), ex.question, examples=shots)


class GpuMemoryMonitor:
    """Samples device memory in a background thread; reports peak above the starting level."""

    def __init__(self, device: int = 0, interval_s: float = 0.02):
        import pynvml

        self._nvml = pynvml
        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(device)
        self._interval = interval_s
        self._stop = threading.Event()
        self.baseline = self._used()
        self.peak = self.baseline
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _used(self) -> int:
        return self._nvml.nvmlDeviceGetMemoryInfo(self._handle).used

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak = max(self.peak, self._used())
            time.sleep(self._interval)

    def __enter__(self) -> "GpuMemoryMonitor":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, self._used())
        self._nvml.nvmlShutdown()

    @property
    def peak_gib(self) -> float:
        return (self.peak - self.baseline) / 2**30


def percentile(values: list[float], q: float) -> float:
    return statistics.quantiles(values, n=100, method="inclusive")[q - 1] if len(values) > 1 else values[0]


def generate(
    spec: ModelSpec,
    examples: list[Example],
    few_shot: list[Example],
    eval_cfg: dict,
    throughput: bool = True,
) -> tuple[list[dict], dict]:
    """Returns (prediction rows, efficiency metrics)."""
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    gen = eval_cfg["generation"]
    params = SamplingParams(temperature=gen["temperature"], max_tokens=gen["max_tokens"], seed=gen["seed"])
    conversations = [build_messages(ex, few_shot) for ex in examples]

    vllm_args = dict(spec.vllm_args)
    vllm_args.pop("gpu_memory_utilization", None)  # replaced by a fixed KV-cache size
    vllm_args["kv_cache_memory_bytes"] = int(eval_cfg["kv_cache_gib"] * 2**30)
    lora = None
    if spec.adapter:
        adapter_cfg = json.loads((spec.adapter / "adapter_config.json").read_text())
        vllm_args.update(enable_lora=True, max_lora_rank=max(16, int(adapter_cfg.get("r", 16))))
        lora = LoRARequest("adapter", 1, str(spec.adapter))

    with GpuMemoryMonitor() as mem:
        llm = LLM(model=str(spec.path), seed=gen["seed"], **vllm_args)

        def chat(convs):
            return llm.chat(
                convs,
                params,
                use_tqdm=False,
                lora_request=lora,
                chat_template_kwargs=spec.chat_template_kwargs,
            )

        chat(conversations[:1])  # warm-up (compilation / CUDA graphs), not timed

        rows, latencies = [], []
        for i, (ex, conv) in enumerate(zip(examples, conversations), start=1):
            t0 = time.perf_counter()
            out = chat([conv])[0].outputs[0]
            latency_ms = (time.perf_counter() - t0) * 1000
            latencies.append(latency_ms)
            rows.append(
                {
                    "db_id": ex.db_id,
                    "question": ex.question,
                    "gold_sql": ex.gold_sql,
                    "pred_sql": extract_sql(out.text),
                    "raw_output": out.text,
                    "latency_ms": round(latency_ms, 2),
                    "tokens_out": len(out.token_ids),
                }
            )
            if i % 100 == 0 or i == len(examples):
                print(f"[eval] latency pass {i}/{len(examples)}, mean {statistics.fmean(latencies):.0f} ms", flush=True)

        efficiency = {
            "latency_ms_mean": round(statistics.fmean(latencies), 2),
            "latency_ms_p50": round(percentile(latencies, 50), 2),
            "latency_ms_p95": round(percentile(latencies, 95), 2),
        }

        if throughput:
            bs = eval_cfg["throughput_batch_size"]
            t0, n_tokens, same = time.perf_counter(), 0, 0
            print(f"[eval] throughput pass, batch size {bs}", flush=True)
            for i in range(0, len(conversations), bs):
                for j, o in enumerate(chat(conversations[i : i + bs]), start=i):
                    n_tokens += len(o.outputs[0].token_ids)
                    same += extract_sql(o.outputs[0].text) == rows[j]["pred_sql"]
            elapsed = time.perf_counter() - t0
            efficiency.update(
                {
                    "throughput_batch_size": bs,
                    "throughput_qps": round(len(conversations) / elapsed, 3),
                    "throughput_tokens_per_s": round(n_tokens / elapsed, 1),
                    # greedy batched decoding is not bit-identical to batch size 1
                    "batched_same_sql_rate": round(same / len(conversations), 4),
                }
            )

    efficiency["peak_gpu_mem_gib"] = round(mem.peak_gib, 2)
    efficiency["kv_cache_gib"] = eval_cfg["kv_cache_gib"]
    size = dir_size_gib(spec.path) + (dir_size_gib(spec.adapter) if spec.adapter else 0.0)
    efficiency["model_size_gib"] = round(size, 2)
    return rows, efficiency


def relpath(p: Path | None) -> str | None:
    if p is None:
        return None
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)
