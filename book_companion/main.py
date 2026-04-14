"""CLI entry point for quick local """

import json
from argparse import ArgumentParser # CLI help

from book_companion.graph import run_graph_once # TODO replace this


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--day", required=True)
    parser.add_argument("--notes-path", required=True)
    args = parser.parse_args()

    result = run_graph_once(
        {
            "day": args.day,
            "daily_notes_path": args.notes_path,
            "loop_count": 0,
        }
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

