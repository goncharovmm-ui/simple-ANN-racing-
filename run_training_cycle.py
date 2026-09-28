"""Headless 20×20 DAgger training run with per-epoch trajectory history."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# Must be set before pygame is imported: macOS otherwise terminates a no-window run.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from racing.batch import ParallelDaggerBatch
from racing.neural import NeuralPolicy
from racing.simulator import Track


def trajectory_record(game) -> dict:
    points, color = game.trajectory_history[-1] if game.trajectory_history else (game.trajectory, (220, 60, 65))
    return {"success": game.successful_episodes > 0, "lap_seconds": game.last_lap,
            "collisions": game.collisions, "color": color,
            "points": [[round(point.x, 1), round(point.y, 1)] for point in points[::4]]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run headless DAgger epochs and train after every epoch")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--agents", type=int, default=20)
    parser.add_argument("--map", type=Path, default=Path("maps/default.json"))
    parser.add_argument("--train-epochs", type=int, default=100)
    parser.add_argument("--run-name", default="cycle", help="name used to isolate this experiment's DAgger data")
    parser.add_argument("--output", type=Path, default=Path("artifacts/training-history.json"))
    args = parser.parse_args()
    track = Track.load(args.map)
    policy_path = Path("models/cycle/policy_000.npz")
    NeuralPolicy.random().save(policy_path)
    history = {"map_path": str(args.map), "track": {"road": list(track.road), "inner_grass": list(track.inner_grass), "start": list(track.start),
               "finish": list(track.finish), "obstacles": [list(rect) for rect in track.obstacles],
               "checkpoints": [list(point) for point in track.checkpoints]}, "epochs": []}
    for epoch in range(1, args.epochs + 1):
        mode = f"dagger-{args.run_name}-epoch-{epoch:02d}"
        batch = ParallelDaggerBatch(track, NeuralPolicy.load(policy_path), args.map, agents=args.agents, speed=100, display=False, game_mode=mode, label_all=True, expert_controls=epoch <= 2)
        batch.run()
        trajectories = [trajectory_record(game) for game in batch.games]
        finished = sum(item["success"] for item in trajectories)
        history["epochs"].append({"epoch": epoch, "policy": str(policy_path), "finished": finished, "agents": args.agents, "trajectories": trajectories})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(history, ensure_ascii=False), encoding="utf-8")
        sources = []
        for previous in range(1, epoch + 1):
            sources.extend(["--data", str(Path("data/demos") / f"dagger-{args.run_name}-epoch-{previous:02d}")])
        next_policy = Path("models/cycle") / f"policy_{epoch:03d}.npz"
        command = [sys.executable, "train.py", *sources, "--output", str(next_policy), "--epochs", str(args.train_epochs)]
        print(f"Epoch {epoch}/{args.epochs}: {finished}/{args.agents} finished; training {next_policy}")
        subprocess.run(command, check=True)
        policy_path = next_policy
    Path("models/policy.npz").write_bytes(policy_path.read_bytes())
    print(f"Complete. Final policy: {policy_path}; history: {args.output}")


if __name__ == "__main__":
    main()
