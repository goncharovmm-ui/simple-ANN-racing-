"""Headless 20×20 DAgger training run with per-epoch trajectory history."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Must be set before pygame is imported: macOS otherwise terminates a no-window run.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from racing.batch import ParallelDaggerBatch
from racing.neural import NeuralPolicy, parse_hidden_sizes
from racing.simulator import Track


def trajectory_record(game, fastest: float | None, slowest: float | None) -> dict:
    points, _ = game.trajectory_history[-1] if game.trajectory_history else (game.trajectory, (220, 60, 65))
    success = game.successful_episodes > 0
    if success:
        if fastest is None or slowest is None or slowest == fastest:
            green = 255
        else:
            green = round(255 * (slowest - (game.last_lap or slowest)) / (slowest - fastest))
        color = [0, green, 0]
    else:
        color = [220, 45, 45]
    return {"success": success, "lap_seconds": game.last_lap, "fitness": round(game.fitness, 2), "directional_score": round(game.directional_score, 2), "course_progress": round(game.course_progress, 5), "progress": game.last_lap_progress if success else game.progress,
            "checkpoint_points": round(game.checkpoint_points, 2), "last_checkpoint_accuracy": round(game.last_checkpoint_accuracy, 4),
            "collisions": game.collisions, "color": color,
            "points": [[round(point.x, 1), round(point.y, 1)] for point in points]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run headless DAgger epochs and train after every epoch")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--agents", type=int, help="total collection cars; defaults to best-trajectories × cars-per-trajectory")
    parser.add_argument("--map", type=Path, default=Path("maps/default.json"))
    parser.add_argument("--train-epochs", type=int, default=100)
    parser.add_argument("--eval-agents", type=int, default=20, help="autonomous cars used to score each newly trained policy")
    parser.add_argument("--seed-data", type=Path, action="append", default=None, help="best human CSV or directory used as the initial expert dataset")
    parser.add_argument("--base-model", type=Path, help="existing policy to continue training")
    parser.add_argument("--best-trajectories", type=int, default=3, help="number of best human trajectories included in every training epoch")
    parser.add_argument("--cars-per-trajectory", type=int, default=7, help="collection cars allocated to each selected trajectory")
    parser.add_argument("--map-name", default=None, help="friendly map name stored in training history")
    parser.add_argument("--model-name", default=None, help="friendly model name stored in training history")
    parser.add_argument("--hidden-layers", type=int, default=None, help="number of hidden layers")
    parser.add_argument("--hidden-units", default=None, help="neurons per hidden layer: 64 or 128,64,32")
    parser.add_argument("--run-name", default="cycle", help="name used to isolate this experiment's DAgger data")
    parser.add_argument("--output", type=Path, default=Path("artifacts/training-history.json"))
    args = parser.parse_args()
    track = Track.load(args.map)
    best_trajectories = max(1, args.best_trajectories)
    cars_per_trajectory = max(1, args.cars_per_trajectory)
    collection_agents = args.agents or best_trajectories * cars_per_trajectory
    hidden_sizes = None if args.hidden_layers is None and args.hidden_units is None else parse_hidden_sizes(args.hidden_layers, args.hidden_units)
    model_dir = Path("models/cycle") / args.run_name
    policy_path = model_dir / "policy_000.npz"
    seed_sources = list(args.seed_data or [])
    if args.base_model and args.base_model.exists():
        if seed_sources:
            seed_command = [sys.executable, "train.py", *sum((["--data", str(path)] for path in seed_sources), []), "--output", str(policy_path), "--epochs", str(args.train_epochs), "--init-model", str(args.base_model)]
            if hidden_sizes:
                seed_command.extend(["--hidden-layers", str(len(hidden_sizes)), "--hidden-units", ",".join(map(str, hidden_sizes))])
            subprocess.run(seed_command, check=True)
        else:
            shutil.copy2(args.base_model, policy_path)
    elif seed_sources:
        seed_command = [sys.executable, "train.py", *sum((["--data", str(path)] for path in seed_sources), []), "--output", str(policy_path), "--epochs", str(args.train_epochs)]
        if hidden_sizes:
            seed_command.extend(["--hidden-layers", str(len(hidden_sizes)), "--hidden-units", ",".join(map(str, hidden_sizes))])
        subprocess.run(seed_command, check=True)
    else:
        NeuralPolicy.random(hidden_sizes=hidden_sizes or [64]).save(policy_path)
    best_policy: Path | None = None
    best_score: tuple[float, int, float] | None = None
    history_hidden_sizes = hidden_sizes
    if history_hidden_sizes is None and args.base_model and args.base_model.exists():
        try:
            history_hidden_sizes = NeuralPolicy.load(args.base_model).hidden_sizes
        except (OSError, ValueError, KeyError):
            history_hidden_sizes = [64]
    history = {"map_path": str(args.map), "map_name": args.map_name or track.name, "model_name": args.model_name or args.run_name,
               "settings": {"epochs": args.epochs, "collection_agents": collection_agents, "best_trajectories": best_trajectories,
                            "cars_per_trajectory": cars_per_trajectory, "hidden_layers": len(history_hidden_sizes or [64]),
                            "hidden_units": history_hidden_sizes or [64]},
               "track": {"name": track.name, "road": list(track.road), "inner_grass": list(track.inner_grass), "start": list(track.start),
               "finish": list(track.finish), "obstacles": [list(rect) for rect in track.obstacles],
               "checkpoints": [list(point) for point in track.checkpoints]}, "epochs": []}
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.monotonic()
        mode = f"dagger-{args.run_name}-epoch-{epoch:02d}"
        expert_probability = max(0.10, 0.85 * (0.82 ** (epoch - 1)))
        batch = ParallelDaggerBatch(track, NeuralPolicy.load(policy_path), args.map, agents=collection_agents, speed=100, display=False, game_mode=mode, label_all=True, expert_probability=expert_probability, route_random_starts=True)
        batch.run()
        sources = []
        for seed_source in seed_sources:
            sources.extend(["--data", str(seed_source)])
        for previous in range(1, epoch + 1):
            sources.extend(["--data", str(Path("data/demos") / f"dagger-{args.run_name}-epoch-{previous:02d}")])
        next_policy = model_dir / f"policy_{epoch:03d}.npz"
        command = [sys.executable, "train.py", *sources, "--output", str(next_policy), "--epochs", str(args.train_epochs), "--init-model", str(policy_path)]
        if hidden_sizes:
            command.extend(["--hidden-layers", str(len(hidden_sizes)), "--hidden-units", ",".join(map(str, hidden_sizes))])
        print(f"Epoch {epoch}/{args.epochs}: collecting DAgger data with expert share {expert_probability:.0%}; training {next_policy}")
        subprocess.run(command, check=True)
        evaluation = ParallelDaggerBatch(track, NeuralPolicy.load(next_policy), args.map, agents=args.eval_agents, speed=100, display=False,
                                         game_mode=f"eval-{args.run_name}-epoch-{epoch:02d}", label_all=False, expert_probability=0.0,
                                         route_random_starts=True, record=False)
        evaluation.run()
        lap_times = [game.last_lap for game in evaluation.games if game.successful_episodes > 0 and game.last_lap is not None]
        fastest, slowest = (min(lap_times), max(lap_times)) if lap_times else (None, None)
        trajectories = [trajectory_record(game, fastest, slowest) for game in evaluation.games]
        finished = sum(item["success"] for item in trajectories)
        mean_lap = sum(lap_times) / len(lap_times) if lap_times else float("inf")
        mean_fitness = sum(game.fitness for game in evaluation.games) / len(evaluation.games)
        score = (mean_fitness, finished, -mean_lap)
        promoted = best_score is None or score > best_score
        if promoted:
            best_policy = model_dir / "best.npz"
            best_policy.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(next_policy, best_policy)
            best_score = score
        policy_path = best_policy if best_policy is not None else next_policy
        epoch_seconds = round(time.monotonic() - epoch_started, 1)
        history["epochs"].append({"epoch": epoch, "policy": str(next_policy), "selected_policy": str(policy_path), "expert_probability": expert_probability,
                                  "epoch_seconds": epoch_seconds, "collection_agents": collection_agents, "best_trajectories": best_trajectories,
                                  "cars_per_trajectory": cars_per_trajectory, "finished": finished, "agents": args.eval_agents, "mean_lap_seconds": None if mean_lap == float("inf") else mean_lap,
                                  "mean_fitness": round(mean_fitness, 2),
                                  "promoted": promoted, "trajectories": trajectories})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(history, ensure_ascii=False), encoding="utf-8")
        print(f"Epoch {epoch}/{args.epochs}: autonomous evaluation {finished}/{args.eval_agents}; {epoch_seconds:.1f}s; {'new best policy' if promoted else 'kept previous best'}")
    Path("models/policy.npz").write_bytes(policy_path.read_bytes())
    print(f"Complete. Best policy: {policy_path}; architecture: {NeuralPolicy.load(policy_path).architecture}; history: {args.output}")


if __name__ == "__main__":
    main()
