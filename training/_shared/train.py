"""Entry point for one fixed model with the shared independent test protocol."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from .protocol import PROTOCOL, SEEDS, TEST_YEARS, TUNING_VALIDATION_YEARS


def main(variant):
    shared = Path(__file__).resolve().parent
    package = shared.parents[1]
    parser = argparse.ArgumentParser(description=f"训练 {variant} 并逐年评估")
    parser.add_argument("--data-dir", type=Path, default=package / "data")
    parser.add_argument("--out-dir", type=Path, help="新训练结果目录")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--plan", action="store_true", help="查看实验配置与年份")
    args = parser.parse_args()

    if args.plan:
        print(json.dumps({
            "variant": variant,
            "protocol": PROTOCOL,
            "seeds": [42] if variant == "ema_only" else list(SEEDS),
            "initial_selection_years": [] if variant == "ema_only" else TUNING_VALIDATION_YEARS,
            "folds": [
                {"historical_validation": [year - 2, year - 1],
                 "refit_end": year - 1, "test_year": year}
                for year in TEST_YEARS
            ],
        }, ensure_ascii=False, indent=2))
        return

    if args.out_dir is None:
        parser.error("请用 --out-dir 指定训练输出目录，或用 --plan 查看设置")

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA 不可用")
    data = args.data_dir.resolve()
    if not data.is_dir():
        parser.error(f"数据目录不存在 {data}")

    output = args.out_dir.resolve()
    if output.exists() and not output.is_dir():
        parser.error(f"输出路径必须是目录 {output}")
    linked_data = output / "data"
    manifest_file = output / "manifest.json"
    if output.exists() and any(output.iterdir()):
        if not linked_data.is_dir() or linked_data.resolve() != data:
            parser.error("继续训练时需使用原训练数据目录")
        if manifest_file.is_file():
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            if manifest.get("protocol") != PROTOCOL or manifest.get("variants") != [variant]:
                parser.error("请为本模型和协议使用新的输出目录")
        elif any((output / name).exists() for name in ("experiments", "tuning", "summary", "status.json")):
            parser.error("输出目录已有任务数据且缺少实验清单")
    else:
        output.mkdir(parents=True, exist_ok=True)
        linked_data.symlink_to(data, target_is_directory=True)

    panel = output / "prepared" / "panel.npz"
    if not panel.is_file():
        subprocess.run([sys.executable, str(shared / "run.py"), "prepare",
                        "--data-dir", str(data), "--output", str(panel.parent)], check=True)
    if not (output / "verification.json").is_file():
        subprocess.run([sys.executable, str(shared / "verify.py"), str(output), "--device", device], check=True)
    subprocess.run([sys.executable, str(shared / "supervise.py"), str(output),
                    "--device", device, "--variants", variant], check=True)
