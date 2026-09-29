"""Train several MLP sizes on the same demonstrations and compare validation loss."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def parameter_count(architecture: list[int]) -> int:
    return sum((architecture[index] + 1) * architecture[index + 1] for index in range(len(architecture) - 1))


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep neural architectures on one fixed dataset")
    parser.add_argument("--data", type=Path, action="append", default=None)
    parser.add_argument("--architectures", default="16;32;64;32,16;64,32;64,32,16")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--output", type=Path, default=Path("artifacts/architecture-sweep.json"))
    args = parser.parse_args()
    datasets = args.data or [Path("data/demos")]
    results = []
    for spec in [item.strip() for item in args.architectures.split(";") if item.strip()]:
        widths = [int(value.strip()) for value in spec.split(",")]
        safe_name = "-".join(map(str, widths))
        model_path = Path("models/saved") / f"sweep-{safe_name}.npz"
        metrics_path = args.output.parent / f"architecture-sweep-{safe_name}.json"
        command = [sys.executable, "train.py"]
        for data in datasets:
            command.extend(["--data", str(data)])
        command.extend(["--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
                        "--hidden-layers", str(len(widths)), "--hidden-units", ",".join(map(str, widths)),
                        "--output", str(model_path), "--metrics-output", str(metrics_path)])
        started = time.monotonic()
        subprocess.run(command, check=True)
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics.update({"model": str(model_path), "seconds": round(time.monotonic() - started, 2),
                        "parameter_count": parameter_count(metrics["architecture"])})
        results.append(metrics)
    results.sort(key=lambda item: float(item["validation_loss"]))
    report = {"data": [str(path) for path in datasets], "epochs": args.epochs, "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
