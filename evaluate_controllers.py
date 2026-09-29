"""Compare a neural policy and an interpretable lidar rule in the simulator."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from racing.baseline import LidarCorrelationPolicy
from racing.batch import ParallelDaggerBatch
from racing.neural import NeuralPolicy
from racing.simulator import Track


def evaluate(name: str, policy, track: Track, map_path: Path, agents: int, max_steps: int, random_start: bool) -> dict[str, object]:
    batch = ParallelDaggerBatch(track, policy, map_path, agents=agents, speed=100, display=False,
                                game_mode=f"controller-eval-{name}", route_random_starts=random_start,
                                record=False, require_full_lap=True, max_steps=max_steps)
    batch.run()
    laps = [game.last_lap for game in batch.games if game.successful_episodes > 0 and game.last_lap is not None]
    return {
        "controller": name,
        "agents": agents,
        "finished": len(laps),
        "finish_rate": round(len(laps) / max(1, agents), 4),
        "mean_lap_seconds": round(sum(laps) / len(laps), 3) if laps else None,
        "mean_fitness": round(sum(game.fitness for game in batch.games) / max(1, agents), 3),
        "mean_directional_score": round(sum(game.directional_score for game in batch.games) / max(1, agents), 3),
        "mean_collisions": round(sum(game.collisions for game in batch.games) / max(1, agents), 3),
        "max_course_progress": round(max(game.course_progress for game in batch.games), 5),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate neural and non-neural controllers")
    parser.add_argument("--map", type=Path, default=Path("maps/default.json"))
    parser.add_argument("--model", type=Path)
    parser.add_argument("--agents", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=3600)
    parser.add_argument("--random-start", action="store_true", help="start each car from a random route segment")
    parser.add_argument("--output", type=Path, default=Path("artifacts/controller-comparison.json"))
    args = parser.parse_args()
    track = Track.load(args.map)
    results = [evaluate("lidar-rule", LidarCorrelationPolicy(), track, args.map, args.agents, args.max_steps, args.random_start)]
    if args.model and args.model.exists():
        results.append(evaluate(args.model.stem, NeuralPolicy.load(args.model), track, args.map, args.agents, args.max_steps, args.random_start))
    report = {"map": str(args.map), "model": str(args.model) if args.model else None, "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
