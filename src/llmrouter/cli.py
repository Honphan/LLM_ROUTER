"""Command line: ``python -m llmrouter <command> --config <experiment.yaml>``.

Commands: toy-data, prepare-data, info, fit, calibrate, evaluate, route, report.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from typing import Any

from llmrouter import pipeline
from llmrouter.config import load_config

log = logging.getLogger("llmrouter")


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="experiment YAML file")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override one config value, e.g. router.params.gamma=5 (repeatable)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llmrouter", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    descriptions = {
        "toy-data": "write synthetic battles and benchmark (offline)",
        "prepare-data": "download public raw data and convert it to the standard tables",
        "info": "describe the data an experiment would fit on",
        "fit": "fit the router and save it",
        "calibrate": "find thresholds for target strong-call rates",
        "evaluate": "compute APGR / CPT on the benchmark, with the random baseline",
        "route": "route prompts with a calibrated threshold",
        "report": "print the headline numbers of an experiment",
    }
    parsers = {}
    for name, description in descriptions.items():
        parsers[name] = commands.add_parser(name, help=description, description=description)
        _add_common(parsers[name])

    parsers["calibrate"].add_argument(
        "--pct",
        dest="rates",
        action="append",
        type=float,
        default=[],
        help="target share of queries sent to the strong model (repeatable)",
    )
    route = parsers["route"]
    route.add_argument("--prompt", action="append", default=[], help="prompt to route (repeatable)")
    route.add_argument("--pct", type=float, default=None, help="calibrated strong-call rate to use")
    route.add_argument("--alpha", type=float, default=None, help="explicit threshold instead")
    route.add_argument("--explain", action="store_true", help="include P(strong) and details")
    return parser


def _run(args: argparse.Namespace) -> Any:
    config = load_config(args.config, args.overrides)
    if args.command == "toy-data":
        return pipeline.toy_data(config)
    if args.command == "prepare-data":
        return pipeline.prepare_sources(config)
    if args.command == "info":
        return pipeline.info(config)
    if args.command == "fit":
        return pipeline.fit(config)
    if args.command == "calibrate":
        return pipeline.calibrate(config, args.rates or [0.3, 0.5, 0.8])
    if args.command == "evaluate":
        summary = pipeline.evaluate(config)
        return {key: value for key, value in summary.items() if key != "curve"}
    if args.command == "route":
        if not args.prompt:
            raise ValueError("give at least one --prompt")
        return pipeline.route(config, args.prompt, args.pct, args.alpha, args.explain)
    if args.command == "report":
        return pipeline.report(config)
    raise ValueError(f"unknown command {args.command}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
        force=True,
    )
    try:
        result = _run(args)
    except (FileNotFoundError, KeyError, ValueError, TypeError) as error:
        log.error("%s", error)
        return 1
    sys.stdout.write(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return 0
