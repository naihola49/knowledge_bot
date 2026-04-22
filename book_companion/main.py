"""CLI entry point for quick local graph runs."""

import json
from argparse import ArgumentParser # CLI help

from book_companion.graph import run_graph_once # TODO replace this


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--day", required=True)
    parser.add_argument("--raw-content-path", required=True)
    parser.add_argument("--user-input-path", required=True)
    args = parser.parse_args()

    result = run_graph_once(
        {
            "day": args.day,
            "raw_content_path": args.raw_content_path,
            "user_input_path": args.user_input_path,
            "loop_count": 0,
        }
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

