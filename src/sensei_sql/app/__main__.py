"""Launch the offline natural-language-to-SQL app.

    python -m sensei_sql.app                        # backend from configs/app.yaml
    python -m sensei_sql.app --backend llama-server # CPU, GGUF
    make app
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml

from sensei_sql.paths import CONFIGS


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=CONFIGS / "app.yaml")
    ap.add_argument("--backend", choices=["vllm", "transformers", "llama-server", "ollama"])
    ap.add_argument("--port", type=int)
    args = ap.parse_args()

    # Never phone home (Gradio analytics / update checks, HF hub).
    for var, val in {"GRADIO_ANALYTICS_ENABLED": "False", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}.items():
        os.environ.setdefault(var, val)

    cfg = yaml.safe_load(args.config.read_text())
    if args.backend:
        cfg["backend"] = args.backend

    from sensei_sql.app.backends import make_backend
    from sensei_sql.app.service import SQLAssistant
    from sensei_sql.app.ui import build_ui

    backend = make_backend(cfg)
    assistant = SQLAssistant(backend, timeout_s=cfg["exec_timeout_s"], max_rows=cfg["max_rows"], self_correct=cfg["self_correct"])
    demo = build_ui(assistant, title=cfg.get("title", "sensei-sql"))
    try:
        demo.launch(server_name=cfg.get("host", "127.0.0.1"), server_port=args.port or cfg.get("port", 7860), share=False)
    finally:
        backend.close()


if __name__ == "__main__":
    main()
