"""Vendor Pi's generated model metadata into the Python harness.

The generated snapshot is runtime data.  lsm-harness never reads the Pi
checkout while it is running; this script is only the explicit update path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROVIDERS = (
    "amazon-bedrock",
    "ant-ling",
    "anthropic",
    "azure-openai-responses",
    "baseten",
    "cerebras",
    "cloudflare-ai-gateway",
    "cloudflare-workers-ai",
    "deepseek",
    "fireworks",
    "github-copilot",
    "google",
    "google-vertex",
    "groq",
    "huggingface",
    "kimi-coding",
    "minimax",
    "minimax-cn",
    "mistral",
    "moonshotai",
    "moonshotai-cn",
    "nvidia",
    "openai",
    "openai-codex",
    "opencode",
    "opencode-go",
    "openrouter",
    "qwen-token-plan",
    "qwen-token-plan-cn",
    "together",
    "vercel-ai-gateway",
    "xai",
    "xiaomi",
    "xiaomi-token-plan-ams",
    "xiaomi-token-plan-cn",
    "xiaomi-token-plan-sgp",
    "zai",
    "zai-coding-cn",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pi_root", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("src/lsm_harness/ai/data/pi_models.json"),
    )
    args = parser.parse_args()

    data_dir = args.pi_root / "packages/ai/src/providers/data"
    catalog: dict[str, dict[str, object]] = {}
    for provider_id in PROVIDERS:
        path = data_dir / f"{provider_id}.json"
        if not path.is_file():
            raise SystemExit(f"Missing Pi catalog: {path}")
        raw = json.loads(path.read_text(encoding="utf-8"))
        models: dict[str, object] = {}
        for api_models in raw.values():
            models.update(api_models)
        catalog[provider_id] = models

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(catalog, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote {sum(len(models) for models in catalog.values())} models "
        f"from {len(catalog)} providers to {args.output}"
    )


if __name__ == "__main__":
    main()
