"""Single ``glucotrace`` command that dispatches to every project tool."""

from __future__ import annotations

import importlib
import sys
from typing import Sequence

# subcommand -> (module, function, one-line help)
COMMANDS: dict[str, tuple[str, str, str]] = {
    "encode": ("glucofm.commands", "encode_main", "encode one CGM day to a fingerprint"),
    "compare": ("glucofm.commands", "compare_main", "cosine similarity of two days"),
    "search": ("glucofm.commands", "search_main", "nearest days in canonical corpora"),
    "report": ("glucofm.report", "main", "offline HTML report of a day and neighbours"),
    "download-model": (
        "glucofm.commands",
        "download_main",
        "download and verify the research checkpoint",
    ),
    "prepare-big-ideas": (
        "glucofm.adapters.big_ideas",
        "main",
        "convert PhysioNet BIG IDEAs to canonical days",
    ),
    "prepare-colas": (
        "glucofm.adapters.colas",
        "main",
        "convert the Colas 2019 PLOS dataset to canonical days",
    ),
    "split": ("glucofm.corpus", "main", "participant-disjoint corpus split"),
    "reserve-holdout": ("glucofm.reserve", "main", "reserve a new prospective holdout"),
    "pretrain": ("glucofm.pretrain", "main", "source-balanced latent pretraining"),
    "evaluate": ("glucofm.evaluate", "main", "run a frozen evaluation protocol"),
    "calibrate": ("glucofm.calibrate", "main", "attach fingerprint calibration"),
    "train-csv": ("glucofm.train", "main", "single-CSV reconstruction smoke test"),
}


def _usage() -> str:
    width = max(len(name) for name in COMMANDS)
    lines = [
        "usage: glucotrace <command> [options]",
        "",
        "GlucoTrace: a small, research-only encoder for CGM days.",
        "Not a medical device. Do not use for diagnosis, dosing, or care.",
        "",
        "commands:",
    ]
    lines += [f"  {name:<{width}}  {help_}" for name, (_, _, help_) in COMMANDS.items()]
    lines += ["", "Run 'glucotrace <command> --help' for command options."]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in {"-h", "--help", "help"}:
        print(_usage())
        return 0
    if arguments[0] in {"-V", "--version"}:
        from . import __version__

        print(f"glucotrace {__version__}")
        return 0
    name, rest = arguments[0], arguments[1:]
    if name not in COMMANDS:
        print(f"glucotrace: unknown command {name!r}\n", file=sys.stderr)
        print(_usage(), file=sys.stderr)
        return 2
    module_name, function_name, _ = COMMANDS[name]
    entry = getattr(importlib.import_module(module_name), function_name)
    previous = sys.argv
    sys.argv = [f"glucotrace {name}", *rest]
    try:
        entry()
    except (FileNotFoundError, ValueError) as error:
        print(f"glucotrace {name}: error: {error}", file=sys.stderr)
        return 1
    finally:
        sys.argv = previous
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
