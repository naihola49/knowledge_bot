"""CLI entry point for quick local graph runs"""

import json
from argparse import ArgumentParser

from book_companion.graph import run_graph_once
from book_companion.run_yaml import load_run_config


def main() -> None:
    parser = ArgumentParser(description="Run book companion graph once.")
    parser.add_argument("--config", help="Path to run YAML (schema: RunConfigModel)")
    parser.add_argument("--day")
    parser.add_argument("--raw-content-path")
    parser.add_argument("--user-input-path")
    args = parser.parse_args()

    if args.config:
        initial = load_run_config(args.config)
    else:
        if not args.day or not args.raw_content_path or not args.user_input_path:
            parser.error("--day, --raw-content-path, and --user-input-path are required without --config")
        initial = {
            "day": args.day,
            "raw_content_path": args.raw_content_path,
            "user_input_path": args.user_input_path,
            "loop_count": 0,
        }

    result = run_graph_once(initial)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

