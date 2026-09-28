import argparse
from pathlib import Path

from racing.menu import MainMenu
from racing.batch import ParallelDaggerBatch
from racing.expert import ArtificialExpert
from racing.neural import NeuralPolicy
from racing.simulator import MapEditor, RacingGame, Track


def main() -> None:
    parser = argparse.ArgumentParser(description="2D racing simulator and demonstration collector")
    parser.add_argument("--record", action="store_true", help="kept for compatibility; player races record automatically")
    parser.add_argument("--editor", action="store_true", help="open the track sandbox/editor (shortcut)")
    parser.add_argument("--mode", choices=("sandbox", "race"), help="skip menu and open a mode")
    parser.add_argument("--driver", choices=("player", "ai"), default="player", help="race driver when --mode race is used")
    parser.add_argument("--map", type=Path, default=Path("maps/default.json"), help="map JSON path")
    parser.add_argument("--ai", type=Path, help="run autonomous mode using a trained .npz policy")
    parser.add_argument("--dagger", type=Path, help="run AI and collect artificial-expert correction labels")
    parser.add_argument("--episodes", type=int, default=20, help="number of sequential DAgger attempts when --agents 1")
    parser.add_argument("--agents", type=int, default=20, help="parallel DAgger cars (default: 20)")
    parser.add_argument("--speed", type=int, default=8, help="simulation steps per rendered frame for DAgger (default: 8)")
    args = parser.parse_args()
    policy_path = args.dagger or args.ai or Path("models/policy.npz")
    if not args.editor and not args.mode and not args.ai and not args.dagger:
        selection = MainMenu(policy_path.exists()).run()
        if selection is None:
            return
        args.mode, args.driver = selection
    track = Track.load(args.map)
    if args.editor or args.mode == "sandbox":
        if args.ai or args.dagger or args.driver == "ai":
            parser.error("sandbox cannot use an AI driver")
        MapEditor(track, args.map).run()
    else:
        use_ai = bool(args.ai or args.dagger) or args.driver == "ai"
        if use_ai and not policy_path.exists():
            parser.error(f"No neural policy at {policy_path}. Record demonstrations and run: python train.py")
        policy = NeuralPolicy.load(policy_path) if use_ai else None
        if args.dagger and args.agents > 1:
            ParallelDaggerBatch(track, policy, args.map, agents=args.agents, speed=args.speed).run()
        else:
            RacingGame(track, record=not use_ai or bool(args.dagger), policy=policy, map_path=args.map, game_mode="dagger" if args.dagger else "race", expert=ArtificialExpert() if args.dagger else None, max_episodes=args.episodes if args.dagger else None).run()


if __name__ == "__main__":
    main()
