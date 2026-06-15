"""Run a full retrain trainer smoke on CPU or MPS with exact token checks.

This probe exercises more than LocalTrainHelper:
- TOML config loading
- dotted data-source plugin loading
- custom reward loading
- flow tracing
- local PyTorch backend creation
- trainer sampling, reward scoring, advantage construction, train_step, metrics
- final adapter save

It creates a tiny local Llama model and tokenizer under a temporary work
directory, so the default path does not require network access or benchmark
data. The generated tiny model has no EOS token, which makes the exact
generated-token assertion stable.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import tempfile
import textwrap
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoConfig, LlamaConfig, LlamaForCausalLM
from transformers import PreTrainedTokenizerFast

from retrain.config import load_config
from retrain.flow import build_flow
from retrain.local_train_helper import _mps_is_available
from retrain.trainer import train


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify full retrain local trainer stack on MPS.",
    )
    parser.add_argument(
        "--device",
        default="mps",
        choices=("mps", "cpu"),
        help="Local backend device to probe.",
    )
    parser.add_argument(
        "--expected-new-tokens",
        type=int,
        default=4,
        help="Required exact number of generated tokens for logged samples.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=2,
        help="Number of trainer steps to run.",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=2,
        help="Samples per prompt. Must be at least 2 to produce GRPO signal.",
    )
    parser.add_argument(
        "--work-dir",
        default="",
        help="Optional directory for generated probe artifacts.",
    )
    return parser


def _tiny_vocab() -> dict[str, int]:
    vocab = {
        "[PAD]": 0,
        "[BOS]": 1,
        "[EOS]": 2,
        "[UNK]": 3,
    }
    for idx in range(4, 64):
        vocab[f"tok{idx}"] = idx
    return vocab


def _write_tiny_model(model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
        bos_token_id=1,
        eos_token_id=None,
        pad_token_id=0,
    )
    LlamaForCausalLM(config).save_pretrained(model_dir)

    tokenizer = Tokenizer(WordLevel(vocab=_tiny_vocab(), unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
    )
    fast.chat_template = (
        "{% for message in messages %}"
        "{{ message['content'] }} "
        "{% endfor %}"
        "{% if add_generation_prompt %}{% endif %}"
    )
    fast.save_pretrained(model_dir)


def _write_probe_plugin(plugin_path: Path) -> None:
    plugin_path.write_text(
        textwrap.dedent(
            """
            from retrain.data import Example


            class ProbeDataSource:
                def load(self):
                    return [
                        Example(prompt="tok4 tok5 tok6", reference="probe", task="probe")
                    ]


            _reward_calls = 0


            def make_data(config):
                return ProbeDataSource()


            def score(response, reference):
                global _reward_calls
                _reward_calls += 1
                return 1.0 if _reward_calls % 2 else 0.0
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )


def _write_config(
    path: Path,
    *,
    model_dir: Path,
    adapter_dir: Path,
    log_dir: Path,
    device: str,
    steps: int,
    group_size: int,
    expected_new_tokens: int,
) -> None:
    path.write_text(
        textwrap.dedent(
            f"""
            [model]
            model = "{model_dir}"
            lora_rank = 2

            [algorithm]
            advantage_mode = "grpo"
            transform_mode = "none"

            [training]
            max_steps = {steps}
            batch_size = 1
            group_size = {group_size}
            max_tokens = {expected_new_tokens}
            lr = 0.0001
            save_every = 0
            temperature = 0.8
            top_p = 0.95

            [backend]
            backend = "local"
            devices = "{device}"
            adapter_path = "{adapter_dir}"

            [backend.options]
            train_microbatch_size = 1
            sample_use_cache = false

            [inference]
            engine = "pytorch"

            [data]
            source = "probe_plugins.make_data"

            [reward]
            type = "custom"
            custom_module = "probe_plugins"
            custom_function = "score"

            [logging]
            log_dir = "{log_dir}"
            log_generations = true
            generation_log_samples_per_prompt = {group_size}
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    if not path.is_file():
        return entries
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            payload = json.loads(stripped)
            if isinstance(payload, dict):
                entries.append(payload)
    return entries


def _adapter_file_exists(policy_ref: str) -> bool:
    policy_dir = Path(policy_ref)
    return (
        (policy_dir / "adapter_model.safetensors").is_file()
        or (policy_dir / "adapter_model.bin").is_file()
    )


def run_probe(args: argparse.Namespace) -> dict[str, object]:
    if args.expected_new_tokens <= 0:
        raise ValueError("--expected-new-tokens must be > 0")
    if args.steps <= 0:
        raise ValueError("--steps must be > 0")
    if args.group_size < 2:
        raise ValueError("--group-size must be >= 2")
    if args.device == "mps" and not _mps_is_available():
        raise RuntimeError("MPS requested, but PyTorch reports MPS unavailable")

    if args.work_dir:
        work_dir = Path(args.work_dir).expanduser().resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
    else:
        work_dir = Path(tempfile.mkdtemp(prefix="retrain-mps-stack-probe-"))

    model_dir = work_dir / "tiny-llama-tokenizer"
    adapter_dir = work_dir / "adapter"
    log_dir = work_dir / "logs"
    config_path = work_dir / "probe.toml"
    plugin_path = work_dir / "probe_plugins.py"

    _write_tiny_model(model_dir)
    _write_probe_plugin(plugin_path)
    _write_config(
        config_path,
        model_dir=model_dir,
        adapter_dir=adapter_dir,
        log_dir=log_dir,
        device=args.device,
        steps=args.steps,
        group_size=args.group_size,
        expected_new_tokens=args.expected_new_tokens,
    )

    sys.path.insert(0, str(work_dir))
    started = time.perf_counter()
    config = load_config(str(config_path))
    flow = build_flow(config, gpu=True)
    trace_result = flow.trace()
    if not trace_result.ok:
        raise RuntimeError(
            "Flow trace failed: "
            + json.dumps(
                [issue.message for issue in trace_result.issues],
                sort_keys=True,
            )
        )

    policy_ref = train(config, flow=flow) or ""
    elapsed_s = time.perf_counter() - started

    metrics_entries = _read_jsonl(log_dir / "metrics.jsonl")
    generation_entries = _read_jsonl(log_dir / "emergence" / "generations.jsonl")
    token_counts = [
        int(entry["num_tokens"])
        for entry in generation_entries
        if "num_tokens" in entry
    ]
    expected = int(args.expected_new_tokens)
    exact_logged_tokens = token_counts and all(count == expected for count in token_counts)
    all_steps_unskipped = bool(metrics_entries) and all(
        not bool(entry.get("skipped", False)) for entry in metrics_entries
    )
    finite_losses = bool(metrics_entries) and all(
        math.isfinite(float(entry.get("loss", 0.0))) for entry in metrics_entries
    )
    adapter_exists = bool(policy_ref) and _adapter_file_exists(policy_ref)
    model_type = AutoConfig.from_pretrained(model_dir).model_type

    result = {
        "ok": bool(
            exact_logged_tokens
            and all_steps_unskipped
            and finite_losses
            and adapter_exists
        ),
        "device_requested": args.device,
        "train_device": getattr(flow.backend, "train_device", ""),
        "autocast_dtype": str(getattr(flow.backend, "autocast_dtype", "")),
        "use_amp": bool(getattr(flow.backend, "use_amp", False)),
        "grad_scaler_enabled": bool(flow.backend.scaler.is_enabled()),  # type: ignore[union-attr]
        "model_type": model_type,
        "config_path": str(config_path),
        "work_dir": str(work_dir),
        "policy_ref": policy_ref,
        "adapter_exists": adapter_exists,
        "expected_new_tokens": expected,
        "logged_generation_token_counts": token_counts,
        "metrics_entries": len(metrics_entries),
        "generation_entries": len(generation_entries),
        "all_steps_unskipped": bool(all_steps_unskipped),
        "finite_losses": bool(finite_losses),
        "trace_probe_cases_passed": trace_result.probe_cases_passed,
        "trace_probe_cases_run": trace_result.probe_cases_run,
        "elapsed_s": elapsed_s,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "mps_available": bool(_mps_is_available()),
    }
    if not result["ok"]:
        raise RuntimeError(json.dumps(result, sort_keys=True))
    return result


def main() -> None:
    result = run_probe(_build_parser().parse_args())
    print("MPS_RETRAIN_STACK_PROBE_RESULT " + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
