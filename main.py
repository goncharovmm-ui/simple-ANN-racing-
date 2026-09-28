import argparse
from pathlib import Path

from racing.neural import NeuralPolicy
from racing.simulator import MapEditor, RacingGame, Track


def main() -> None:
    parser = argparse.ArgumentParser(description="2D racing simulator and demonstration collector")
    parser.add_argument("--record", action="store_true", help="start recording manual demonstrations")
    parser.add_argument("--editor", action="store_true", help="open the track sandbox/editor")
    parser.add_argument("--map", type=Path, default=Path("maps/default.json"), help="map JSON path")
    parser.add_argument("--ai", type=Path, help="run autonomous mode using a trained .npz policy")
    args = parser.parse_args()
    track = Track.load(args.map)
    if args.editor:
        if args.ai:
            parser.error("--editor and --ai cannot be used together")
        MapEditor(track, args.map).run()
    else:
        RacingGame(track, record=args.record, policy=NeuralPolicy.load(args.ai) if args.ai else None).run()


if __name__ == "__main__":
    main()
