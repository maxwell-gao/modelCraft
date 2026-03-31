"""Run the upstream parameter-golf baseline and mirror its logs into Weights & Biases.

This wrapper deliberately does not modify ``parameter-golf/``. It launches the
original ``train_gpt.py`` via ``torchrun``, streams stdout to the terminal, and
parses the emitted metrics into wandb.

Example:

    PYTHONPATH=src \
    RUN_ID=baseline_sp1024_wandb \
    DATA_PATH=./data/datasets/fineweb10B_sp1024 \
    TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
    VOCAB_SIZE=1024 \
    WANDB_PROJECT=parameter-golf \
    WANDB_MODE=offline \
    python -m modelCraft.parameter_golf_wandb
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import wandb

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PARAMETER_GOLF_DIR = REPO_ROOT / "parameter-golf"
DEFAULT_TRAIN_SCRIPT = "train_gpt.py"
DEFAULT_ENTRY_MODULE = "modelCraft.parameter_golf_entry"

TRAIN_LOG_RE = re.compile(
    r"step:(?P<step>\d+)/(?P<iterations>\d+)\s+"
    r"train_loss:(?P<train_loss>[-+0-9.eE]+)\s+"
    r"train_time:(?P<train_time_ms>\d+)ms\s+"
    r"step_avg:(?P<step_avg_ms>[-+0-9.eE]+)ms"
)
VAL_LOG_RE = re.compile(
    r"step:(?P<step>\d+)/(?P<iterations>\d+)\s+"
    r"val_loss:(?P<val_loss>[-+0-9.eE]+)\s+"
    r"val_bpb:(?P<val_bpb>[-+0-9.eE]+)\s+"
    r"train_time:(?P<train_time_ms>\d+)ms\s+"
    r"step_avg:(?P<step_avg_ms>[-+0-9.eE]+)ms"
)
WARMUP_RE = re.compile(r"warmup_step:(?P<warmup_step>\d+)/(?P<warmup_total>\d+)")
STOPPING_RE = re.compile(
    r"stopping_early:\s+wallclock_cap\s+train_time:(?P<train_time_ms>\d+)ms\s+step:(?P<step>\d+)/(?P<iterations>\d+)"
)
MEMORY_RE = re.compile(
    r"peak memory allocated:\s+(?P<allocated_mib>\d+)\s+MiB\s+reserved:\s+(?P<reserved_mib>\d+)\s+MiB"
)
SERIALIZED_MODEL_RE = re.compile(r"Serialized model:\s+(?P<bytes>\d+)\s+bytes")
CODE_SIZE_RE = re.compile(r"Code size:\s+(?P<bytes>\d+)\s+bytes")
TOTAL_SIZE_RE = re.compile(r"Total submission size:\s+(?P<bytes>\d+)\s+bytes")
SERIALIZED_INT8_RE = re.compile(
    r"Serialized model int8\+zlib:\s+(?P<bytes>\d+)\s+bytes\s+"
    r"\(payload:(?P<payload_bytes>\d+)\s+raw_torch:(?P<raw_torch_bytes>\d+)\s+payload_ratio:(?P<payload_ratio>[-+0-9.eE]+)x\)"
)
TOTAL_INT8_SIZE_RE = re.compile(r"Total submission size int8\+zlib:\s+(?P<bytes>\d+)\s+bytes")
FINAL_ROUNDTRIP_RE = re.compile(
    r"final_int8_zlib_roundtrip\s+val_loss:(?P<val_loss>[-+0-9.eE]+)\s+"
    r"val_bpb:(?P<val_bpb>[-+0-9.eE]+)\s+eval_time:(?P<eval_time_ms>\d+)ms"
)
FINAL_ROUNDTRIP_EXACT_RE = re.compile(
    r"final_int8_zlib_roundtrip_exact\s+val_loss:(?P<val_loss>[-+0-9.eE]+)\s+val_bpb:(?P<val_bpb>[-+0-9.eE]+)"
)
MODEL_PARAMS_RE = re.compile(r"model_params:(?P<model_params>\d+)")
WORLD_SIZE_RE = re.compile(r"world_size:(?P<world_size>\d+)\s+grad_accum_steps:(?P<grad_accum_steps>\d+)")
ATTN_MODE_RE = re.compile(r"attention_mode:(?P<attention_mode>\S+)\s+num_heads:(?P<num_heads>\d+)\s+num_kv_heads:(?P<num_kv_heads>\d+)")
TRAIN_SETUP_RE = re.compile(
    r"train_batch_tokens:(?P<train_batch_tokens>\d+)\s+train_seq_len:(?P<train_seq_len>\d+)\s+"
    r"iterations:(?P<iterations>\d+)\s+warmup_steps:(?P<warmup_steps>\d+)\s+"
    r"max_wallclock_seconds:(?P<max_wallclock_seconds>[-+0-9.eE]+)"
)
SEED_RE = re.compile(r"seed:(?P<seed>\d+)")

CONFIG_ENV_KEYS = [
    "DATA_PATH",
    "TOKENIZER_PATH",
    "VOCAB_SIZE",
    "ITERATIONS",
    "WARMDOWN_ITERS",
    "WARMUP_STEPS",
    "TRAIN_BATCH_TOKENS",
    "TRAIN_SEQ_LEN",
    "VAL_BATCH_SIZE",
    "VAL_LOSS_EVERY",
    "TRAIN_LOG_EVERY",
    "MAX_WALLCLOCK_SECONDS",
    "QK_GAIN_INIT",
    "NUM_LAYERS",
    "NUM_HEADS",
    "NUM_KV_HEADS",
    "MODEL_DIM",
    "MLP_MULT",
    "TIE_EMBEDDINGS",
    "ROPE_BASE",
    "LOGIT_SOFTCAP",
    "EMBED_LR",
    "HEAD_LR",
    "TIED_EMBED_LR",
    "TIED_EMBED_INIT_STD",
    "MATRIX_LR",
    "SCALAR_LR",
    "MUON_MOMENTUM",
    "MUON_BACKEND_STEPS",
    "MUON_MOMENTUM_WARMUP_START",
    "MUON_MOMENTUM_WARMUP_STEPS",
    "BETA1",
    "BETA2",
    "ADAM_EPS",
    "GRAD_CLIP_NORM",
    "SEED",
    "PG_INIT_METHOD",
    "PG_INIT_BASE_ALPHA",
    "PG_INIT_ALPHA_DEPTH_SCALE",
    "PG_INIT_QK_SHARED_FRAC",
    "PG_INIT_ENABLE_QK_CORR",
    "PG_INIT_EMBED_ALPHA",
    "PG_INIT_HEAD_ALPHA",
]


def default_wandb_mode() -> str:
    if os.environ.get("WANDB_MODE"):
        return os.environ["WANDB_MODE"]
    return "online" if os.environ.get("WANDB_API_KEY") else "offline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parameter-golf-dir",
        type=Path,
        default=Path(os.environ.get("PG_DIR", DEFAULT_PARAMETER_GOLF_DIR)),
        help="Path to the upstream parameter-golf checkout.",
    )
    parser.add_argument(
        "--nproc-per-node",
        type=int,
        default=int(os.environ.get("PG_NPROC_PER_NODE", "1")),
        help="Value passed to torchrun --nproc_per_node.",
    )
    parser.add_argument(
        "--master-port",
        type=int,
        default=int(os.environ.get("PG_MASTER_PORT", "0")),
        help="Optional torchrun --master_port override; 0 leaves torchrun defaults.",
    )
    parser.add_argument(
        "--run-id",
        default=os.environ.get("RUN_ID", "parameter-golf-wandb"),
        help="Run ID forwarded to parameter-golf and used as the default wandb name.",
    )
    parser.add_argument(
        "--wandb-project",
        default=os.environ.get("WANDB_PROJECT", "parameter-golf"),
        help="wandb project name.",
    )
    parser.add_argument(
        "--wandb-entity",
        default=os.environ.get("WANDB_ENTITY"),
        help="Optional wandb entity.",
    )
    parser.add_argument(
        "--wandb-mode",
        default=default_wandb_mode(),
        choices=("online", "offline", "disabled"),
        help="wandb mode. Defaults to online when WANDB_API_KEY is present, otherwise offline.",
    )
    parser.add_argument(
        "--wandb-name",
        default=os.environ.get("WANDB_NAME"),
        help="Optional wandb run name. Defaults to RUN_ID.",
    )
    parser.add_argument(
        "--wandb-group",
        default=os.environ.get("WANDB_GROUP"),
        help="Optional wandb group.",
    )
    parser.add_argument(
        "--wandb-tags",
        default=os.environ.get("WANDB_TAGS", ""),
        help="Comma-separated wandb tags.",
    )
    return parser.parse_args()


def build_wandb_config(args: argparse.Namespace, child_env: dict[str, str]) -> dict[str, object]:
    config: dict[str, object] = {
        "run_id": args.run_id,
        "parameter_golf_dir": str(args.parameter_golf_dir.resolve()),
        "nproc_per_node": args.nproc_per_node,
        "train_script": DEFAULT_TRAIN_SCRIPT,
        "entry_module": DEFAULT_ENTRY_MODULE,
    }
    for key in CONFIG_ENV_KEYS:
        value = child_env.get(key)
        if value is not None:
            config[key.lower()] = value
    return config


def maybe_log_artifact(run: wandb.sdk.wandb_run.Run | None, artifact_path: Path, artifact_name: str, artifact_type: str) -> None:
    if run is None or not artifact_path.is_file():
        return
    artifact = wandb.Artifact(name=artifact_name, type=artifact_type)
    artifact.add_file(str(artifact_path))
    run.log_artifact(artifact)


def log_line_to_wandb(run: wandb.sdk.wandb_run.Run | None, line: str) -> None:
    if run is None:
        return

    if line.startswith("init_config "):
        payload = {}
        for token in line.split()[1:]:
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            payload[key] = value
        run.config.update({f"init_{key}": value for key, value in payload.items()}, allow_val_change=True)
        return

    if match := TRAIN_LOG_RE.search(line):
        step = int(match["step"])
        run.log(
            {
                "step": step,
                "train/loss": float(match["train_loss"]),
                "train/time_ms": int(match["train_time_ms"]),
                "train/step_avg_ms": float(match["step_avg_ms"]),
                "train/iterations": int(match["iterations"]),
            }
        )
        return

    if match := VAL_LOG_RE.search(line):
        step = int(match["step"])
        run.log(
            {
                "step": step,
                "eval/loss": float(match["val_loss"]),
                "eval/bpb": float(match["val_bpb"]),
                "eval/time_ms": int(match["train_time_ms"]),
                "eval/step_avg_ms": float(match["step_avg_ms"]),
                "eval/iterations": int(match["iterations"]),
            }
        )
        return

    if match := WARMUP_RE.search(line):
        run.log(
            {
                "warmup/step": int(match["warmup_step"]),
                "warmup/total": int(match["warmup_total"]),
            }
        )
        return

    if match := STOPPING_RE.search(line):
        run.summary["stopped_early"] = True
        run.summary["stopping_step"] = int(match["step"])
        run.summary["stopping_iterations"] = int(match["iterations"])
        run.summary["stopping_train_time_ms"] = int(match["train_time_ms"])
        return

    if match := MEMORY_RE.search(line):
        run.summary["peak_memory_allocated_mib"] = int(match["allocated_mib"])
        run.summary["peak_memory_reserved_mib"] = int(match["reserved_mib"])
        return

    if match := SERIALIZED_MODEL_RE.search(line):
        run.summary["serialized_model_bytes"] = int(match["bytes"])
        return

    if match := CODE_SIZE_RE.search(line):
        run.summary["code_size_bytes"] = int(match["bytes"])
        return

    if match := TOTAL_SIZE_RE.search(line):
        run.summary["submission_size_bytes"] = int(match["bytes"])
        return

    if match := SERIALIZED_INT8_RE.search(line):
        run.summary["serialized_model_int8_zlib_bytes"] = int(match["bytes"])
        run.summary["serialized_model_int8_payload_bytes"] = int(match["payload_bytes"])
        run.summary["serialized_model_int8_raw_torch_bytes"] = int(match["raw_torch_bytes"])
        run.summary["serialized_model_int8_payload_ratio"] = float(match["payload_ratio"])
        return

    if match := TOTAL_INT8_SIZE_RE.search(line):
        run.summary["submission_size_int8_zlib_bytes"] = int(match["bytes"])
        return

    if match := FINAL_ROUNDTRIP_RE.search(line):
        run.summary["final_roundtrip_val_loss"] = float(match["val_loss"])
        run.summary["final_roundtrip_val_bpb"] = float(match["val_bpb"])
        run.summary["final_roundtrip_eval_time_ms"] = int(match["eval_time_ms"])
        return

    if match := FINAL_ROUNDTRIP_EXACT_RE.search(line):
        run.summary["final_roundtrip_val_loss_exact"] = float(match["val_loss"])
        run.summary["final_roundtrip_val_bpb_exact"] = float(match["val_bpb"])
        return

    if match := MODEL_PARAMS_RE.search(line):
        run.config.update({"model_params": int(match["model_params"])}, allow_val_change=True)
        return

    if match := WORLD_SIZE_RE.search(line):
        run.config.update(
            {
                "world_size": int(match["world_size"]),
                "grad_accum_steps": int(match["grad_accum_steps"]),
            },
            allow_val_change=True,
        )
        return

    if match := ATTN_MODE_RE.search(line):
        run.config.update(
            {
                "attention_mode": match["attention_mode"],
                "num_heads": int(match["num_heads"]),
                "num_kv_heads": int(match["num_kv_heads"]),
            },
            allow_val_change=True,
        )
        return

    if match := TRAIN_SETUP_RE.search(line):
        run.config.update(
            {
                "train_batch_tokens": int(match["train_batch_tokens"]),
                "train_seq_len": int(match["train_seq_len"]),
                "iterations": int(match["iterations"]),
                "warmup_steps": int(match["warmup_steps"]),
                "max_wallclock_seconds": float(match["max_wallclock_seconds"]),
            },
            allow_val_change=True,
        )
        return

    if match := SEED_RE.search(line):
        run.config.update({"seed": int(match["seed"])}, allow_val_change=True)


def main() -> int:
    args = parse_args()
    parameter_golf_dir = args.parameter_golf_dir.resolve()
    train_script_path = parameter_golf_dir / DEFAULT_TRAIN_SCRIPT
    if not train_script_path.is_file():
        raise FileNotFoundError(f"Could not find {train_script_path}")

    child_env = os.environ.copy()
    child_env["RUN_ID"] = args.run_id
    child_pythonpath = [str(REPO_ROOT / "src"), str(REPO_ROOT)]
    existing_pythonpath = child_env.get("PYTHONPATH")
    if existing_pythonpath:
        child_pythonpath.append(existing_pythonpath)
    child_env["PYTHONPATH"] = os.pathsep.join(child_pythonpath)

    tags = [tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()]
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        mode=args.wandb_mode,
        name=args.wandb_name or args.run_id,
        group=args.wandb_group,
        tags=tags,
        config=build_wandb_config(args, child_env),
    )
    if run is not None:
        run.define_metric("step")
        run.define_metric("train/*", step_metric="step")
        run.define_metric("eval/*", step_metric="step")

    command = ["torchrun", "--standalone", f"--nproc_per_node={args.nproc_per_node}"]
    if args.master_port > 0:
        command.append(f"--master_port={args.master_port}")
    command.extend(["--module", DEFAULT_ENTRY_MODULE])

    if run is not None:
        run.config.update({"command": " ".join(command)}, allow_val_change=True)

    process = subprocess.Popen(
        command,
        cwd=parameter_golf_dir,
        env=child_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    try:
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_line_to_wandb(run, line.strip())
    except KeyboardInterrupt:
        process.terminate()
        raise
    finally:
        return_code = process.wait()

    log_path = parameter_golf_dir / "logs" / f"{args.run_id}.txt"
    model_path = parameter_golf_dir / "final_model.pt"
    model_int8_path = parameter_golf_dir / "final_model.int8.ptz"
    maybe_log_artifact(run, log_path, f"{args.run_id}-logs", "parameter-golf-log")
    maybe_log_artifact(run, model_int8_path, f"{args.run_id}-int8-ptz", "parameter-golf-model")
    maybe_log_artifact(run, model_path, f"{args.run_id}-raw-pt", "parameter-golf-model")

    if run is not None:
        run.summary["exit_code"] = return_code
        wandb.finish(exit_code=return_code)

    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
