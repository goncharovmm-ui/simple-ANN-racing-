import argparse
from pathlib import Path

from racing.menu import MainMenu
from racing.neural import NeuralPolicy
from racing.simulator import MapEditor, RacingGame, Track


def main() -> None:
    parser = argparse.ArgumentParser(description="2D racing simulator and demonstration collector")
    parser.add_argument("--record", action="store_true", help="start recording manual demonstrations")
    parser.add_argument("--editor", action="store_true", help="open the track sandbox/editor (shortcut)")
    parser.add_argument("--mode", choices=("sandbox", "race"), help="skip menu and open a mode")
    parser.add_argument("--driver", choices=("player", "ai"), default="player", help="race driver when --mode race is used")
    parser.add_argument("--map", type=Path, default=Path("maps/default.json"), help="map JSON path")
    parser.add_argument("--ai", type=Path, help="run autonomous mode using a trained .npz policy")
    args = parser.parse_args()
    policy_path = args.ai or Path("models/policy.npz")
    if not args.editor and not args.mode and not args.ai:
        selection = MainMenu(policy_path.exists()).run()
        if selection is None:
            return
        args.mode, args.driver = selection
    track = Track.load(args.map)
    if args.editor or args.mode == "sandbox":
        if args.ai or args.driver == "ai":
            parser.error("sandbox cannot use an AI driver")
        MapEditor(track, args.map).run()
    else:
        use_ai = bool(args.ai) or args.driver == "ai"
        if use_ai and not policy_path.exists():
            parser.error(f"No neural policy at {policy_path}. Record demonstrations and run: python train.py")
        RacingGame(track, record=args.record, policy=NeuralPolicy.load(policy_path) if use_ai else None, map_path=args.map).run()


if __name__ == "__main__":
    main()
