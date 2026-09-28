import argparse

from racing.simulator import RacingGame


def main() -> None:
    parser = argparse.ArgumentParser(description="2D racing simulator and demonstration collector")
    parser.add_argument("--record", action="store_true", help="start recording manual demonstrations")
    args = parser.parse_args()
    RacingGame(record=args.record).run()


if __name__ == "__main__":
    main()
