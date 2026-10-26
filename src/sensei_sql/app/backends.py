"""Pluggable inference backends for the app. All take chat messages and return extracted SQL.

* ``vllm``          GPU, base student + optional LoRA adapter, or a merged model dir
* ``transformers``  GPU or CPU, any HF model dir (no extra dependencies)
* ``llama-server``  CPU, a GGUF file served by llama.cpp's ``llama-server`` (started for you)
* ``ollama``        CPU/GPU, a model already loaded in a local Ollama (``ollama create`` from GGUF)

The llama.cpp and Ollama backends talk to localhost only.
"""

from __future__ import annotations

import json
import socket
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from sensei_sql.core.prompts import Messages, extract_sql
from sensei_sql.paths import THIRD_PARTY

MAX_TOKENS = 256


@dataclass
class Generation:
    sql: str
    raw: str
    gen_ms: float


class Backend:
    name = "base"

    def _complete(self, messages: Messages) -> str:
        raise NotImplementedError

    def generate_sql(self, messages: Messages) -> Generation:
        t0 = time.perf_counter()
        raw = self._complete(messages)
        return Generation(sql=extract_sql(raw), raw=raw, gen_ms=(time.perf_counter() - t0) * 1000)

    def close(self) -> None:
        pass


class VLLMBackend(Backend):
    name = "vllm"

    def __init__(self, model: str = "student", adapter: str | None = None, kv_cache_gib: float = 2.0):
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest

        from sensei_sql.eval.runner import resolve_model

        spec = resolve_model(model, adapter)
        args = dict(spec.vllm_args)
        args.pop("gpu_memory_utilization", None)
        args["kv_cache_memory_bytes"] = int(kv_cache_gib * 2**30)
        self._lora = None
        if spec.adapter:
            args.update(enable_lora=True, max_lora_rank=64)
            self._lora = LoRARequest("adapter", 1, str(spec.adapter))
        self._llm = LLM(model=str(spec.path), seed=0, **args)
        self._params = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)
        self._chat_kwargs = spec.chat_template_kwargs

    def _complete(self, messages: Messages) -> str:
        out = self._llm.chat(
            [messages], self._params, use_tqdm=False, lora_request=self._lora, chat_template_kwargs=self._chat_kwargs
        )
        return out[0].outputs[0].text


class TransformersBackend(Backend):
    name = "transformers"

    def __init__(self, model_dir: str, device: str = "auto", adapter: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self._tok = AutoTokenizer.from_pretrained(adapter or model_dir)
        model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype).to(device)
        if adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter)
        self._model = model.eval()
        self._device = device

    def _complete(self, messages: Messages) -> str:
        enc = self._tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(self._device)
        with self._torch.inference_mode():
            out = self._model.generate(**enc, max_new_tokens=MAX_TOKENS, do_sample=False)
        return self._tok.decode(out[0, enc["input_ids"].shape[1] :], skip_special_tokens=True)


def _post_json(url: str, payload: dict, timeout: float = 600) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


class LlamaServerBackend(Backend):
    """llama.cpp ``llama-server`` on localhost (OpenAI-compatible chat endpoint)."""

    name = "llama-server"
    BINARY = THIRD_PARTY / "llama.cpp" / "build" / "bin" / "llama-server"

    def __init__(self, url: str, process: subprocess.Popen | None = None):
        self._url = url.rstrip("/")
        self._process = process

    @classmethod
    def launch(cls, gguf: Path, threads: int | None = None, ctx: int = 8192) -> "LlamaServerBackend":
        if not cls.BINARY.exists():
            raise FileNotFoundError(f"{cls.BINARY} not found: build llama.cpp first (see README)")
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        cmd = [str(cls.BINARY), "-m", str(gguf), "--host", "127.0.0.1", "--port", str(port), "-c", str(ctx)]
        if threads:
            cmd += ["-t", str(threads)]
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        url = f"http://127.0.0.1:{port}"
        for _ in range(600):
            try:
                with urllib.request.urlopen(f"{url}/health", timeout=1) as r:
                    if r.status == 200:
                        return cls(url, proc)
            except OSError:
                if proc.poll() is not None:
                    raise RuntimeError(f"llama-server exited with code {proc.returncode}")
                time.sleep(0.5)
        proc.kill()
        raise TimeoutError("llama-server did not become healthy")

    def _complete(self, messages: Messages) -> str:
        out = _post_json(
            f"{self._url}/v1/chat/completions",
            {"messages": messages, "temperature": 0.0, "max_tokens": MAX_TOKENS, "seed": 0},
        )
        return out["choices"][0]["message"]["content"]

    def close(self) -> None:
        if self._process and self._process.poll() is None:
            self._process.terminate()
            self._process.wait(timeout=30)


class OllamaBackend(Backend):
    name = "ollama"

    def __init__(self, model: str, url: str = "http://127.0.0.1:11434"):
        self._model = model
        self._url = url.rstrip("/")

    def _complete(self, messages: Messages) -> str:
        out = _post_json(
            f"{self._url}/api/chat",
            {
                "model": self._model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": 0, "num_predict": MAX_TOKENS, "seed": 0},
            },
        )
        return out["message"]["content"]


def make_backend(cfg: dict) -> Backend:
    kind = cfg["backend"]
    opts = cfg.get(kind, {}) or {}
    if kind == "vllm":
        return VLLMBackend(opts.get("model", "student"), opts.get("adapter"), opts.get("kv_cache_gib", 2.0))
    if kind == "transformers":
        return TransformersBackend(opts["model_dir"], opts.get("device", "auto"), opts.get("adapter"))
    if kind == "llama-server":
        return LlamaServerBackend.launch(Path(opts["gguf"]), opts.get("threads"))
    if kind == "ollama":
        return OllamaBackend(opts["model"], opts.get("url", "http://127.0.0.1:11434"))
    raise ValueError(f"unknown backend {kind!r}")
