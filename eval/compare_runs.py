#!/usr/bin/env python3
"""Compare benchmark runs, so a regression is visible rather than remembered.

A single benchmark run tells you where the system is. Two tell you which way it
is going, which is the question that actually governs whether to ship. Without
this, "did that change make retrieval worse?" is answered by scrolling back
through terminal output, which is to say it is not answered.

    python eval/compare_runs.py                    # latest against the one before
    python eval/compare_runs.py --baseline <file>  # against a chosen run
    python eval/compare_runs.py --history          # every run, one row each

Regressions are called out explicitly. The threshold is deliberately small: a
metric that moves by more than 2 percentage points has moved for a reason, and
the reason is worth knowing before it compounds.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path
from typing import Any

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Windows consoles default to cp1252, and these reports print box-drawing
# characters, arrows and ellipses. Without this the run dies on a UnicodeEncodeError
# after doing all the work -- which is how a benchmark ends up with no output.
if hasattr(sys.stdout, "reconfigure"):
    with contextlib.suppress(ValueError, OSError):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


#: Metric paths worth tracking across runs, with the direction that counts as
#: better. Abstention is the interesting one: recall on unanswerable questions
#: should rise and false abstention should fall, and a change that moves both the
#: same way is usually a threshold being dragged rather than an improvement.
TRACKED: list[tuple[str, tuple[str, ...], str]] = [
    ("context recall", ("retrieval", "context_recall"), "up"),
    ("context precision", ("retrieval", "context_precision"), "up"),
    ("entity recall (golden)", ("retrieval", "entity_recall"), "up"),
    ("intent accuracy", ("intent_routing", "accuracy"), "up"),
    ("abstention recall", ("abstention", "recall_on_unanswerable"), "up"),
    ("false abstention", ("abstention", "false_abstention_rate"), "down"),
    ("entity precision", ("entity_extraction", "precision_micro"), "up"),
    ("entity recall", ("entity_extraction", "recall_micro"), "up"),
    ("entity F1", ("entity_extraction", "f1_micro"), "up"),
    ("citation validity", ("citation_validity", "validity_rate"), "up"),
    ("groundedness", ("groundedness", "groundedness"), "up"),
    ("compliance accuracy", ("compliance_detection", "accuracy"), "up"),
    ("mention resolution", ("linkage", "mention_resolution_rate"), "up"),
    ("drawing linkage", ("linkage", "drawing_linkage_rate"), "up"),
    ("latency p50 (s)", ("latency", "p50_s"), "down"),
    ("latency p95 (s)", ("latency", "p95_s"), "down"),
]

#: A movement smaller than this is noise -- retrieval over a small corpus is not
#: perfectly deterministic once a cross-encoder is involved.
NOISE_FLOOR = 0.02


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def runs() -> list[Path]:
    return sorted(RESULTS_DIR.glob("*.json"))


def resolve(reference: str) -> Path | None:
    """Find a run from a path, a filename, a stem, or a tag.

    Typing the full path to `eval/results/20260908T195645Z-day7-final.json` is
    the least likely thing anyone does; they type the tag they gave the run.
    """
    direct = Path(reference)
    if direct.is_file():
        return direct
    for candidate in (
        RESULTS_DIR / reference,
        RESULTS_DIR / f"{reference}.json",
    ):
        if candidate.is_file():
            return candidate
    # Fall back to a tag match, newest first.
    for path in reversed(runs()):
        if path.stem.endswith(f"-{reference}") or path.stem == reference:
            return path
        try:
            if json.loads(path.read_text(encoding="utf-8")).get("tag") == reference:
                return path
        except (OSError, json.JSONDecodeError):
            continue
    return None


def dig(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    node: Any = payload
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def compare(baseline: Path, candidate: Path) -> int:
    old = load(baseline)
    new = load(candidate)

    print(f"\n\033[36mbaseline \033[0m{baseline.name}  ({old.get('tag') or 'untagged'})")
    print(f"\033[36mcandidate\033[0m {candidate.name}  ({new.get('tag') or 'untagged'})\n")
    print(f"  {'metric':24} {'baseline':>10} {'candidate':>10} {'change':>10}")
    print(f"  {'-' * 24} {'-' * 10} {'-' * 10} {'-' * 10}")

    regressions: list[str] = []
    improvements: list[str] = []

    for label, path, direction in TRACKED:
        before = dig(old["summary"], path)
        after = dig(new["summary"], path)
        if before is None and after is None:
            continue
        if not isinstance(before, (int, float)) or not isinstance(after, (int, float)):
            print(f"  {label:24} {fmt(before):>10} {fmt(after):>10} {'—':>10}")
            continue

        delta = after - before
        better = delta > 0 if direction == "up" else delta < 0
        if abs(delta) <= NOISE_FLOOR:
            colour, mark = "", ""
        elif better:
            colour, mark = "\033[32m", "▲"
            improvements.append(f"{label} {before:.3f} → {after:.3f}")
        else:
            colour, mark = "\033[31m", "▼"
            regressions.append(f"{label} {before:.3f} → {after:.3f}")
        reset = "\033[0m" if colour else ""
        print(f"  {label:24} {before:>10.3f} {after:>10.3f} {colour}{delta:>+9.3f}{mark}{reset}")

    print()
    if improvements:
        print("\033[32mImproved\033[0m")
        for line in improvements:
            print(f"  {line}")
    if regressions:
        print("\033[31mRegressed\033[0m")
        for line in regressions:
            print(f"  {line}")
        print(
            "\n  A regression above the noise floor has a cause. Find it before "
            "moving on:\n  the per-case results are in the candidate file under 'results'."
        )
    if not improvements and not regressions:
        print("  No metric moved by more than the noise floor.")
    print()
    # Non-zero exit so this can gate a pipeline.
    return 1 if regressions else 0


def history() -> int:
    files = runs()
    if not files:
        print("No benchmark runs in eval/results/.", file=sys.stderr)
        return 1

    print(f"\n  {'run':34} {'tag':18} {'ctx rec':>8} {'F1':>7} {'abst':>7} {'p95 s':>7}")
    print(f"  {'-' * 34} {'-' * 18} {'-' * 8} {'-' * 7} {'-' * 7} {'-' * 7}")
    for path in files:
        payload = load(path)
        summary = payload["summary"]
        print(
            f"  {path.name:34} {(payload.get('tag') or '')[:18]:18} "
            f"{fmt(dig(summary, ('retrieval', 'context_recall'))):>8} "
            f"{fmt(dig(summary, ('entity_extraction', 'f1_micro'))):>7} "
            f"{fmt(dig(summary, ('abstention', 'recall_on_unanswerable'))):>7} "
            f"{fmt(dig(summary, ('latency', 'p95_s'))):>7}"
        )
    print()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", help="run to compare against (default: the previous one)")
    parser.add_argument("--candidate", help="run to evaluate (default: the latest)")
    parser.add_argument(
        "positional",
        nargs="*",
        metavar="RUN",
        help="baseline and candidate, as a tag, a filename or a path",
    )
    parser.add_argument("--history", action="store_true", help="one row per run")
    args = parser.parse_args(argv)

    if args.history:
        return history()

    files = runs()
    if len(files) < 2 and not (args.baseline and args.candidate):
        print("Need at least two runs to compare. Use --history to list them.", file=sys.stderr)
        return 1

    positional = list(args.positional)
    baseline_ref = args.baseline or (positional.pop(0) if positional else None)
    candidate_ref = args.candidate or (positional.pop(0) if positional else None)

    baseline = resolve(baseline_ref) if baseline_ref else files[-2]
    candidate = resolve(candidate_ref) if candidate_ref else files[-1]
    for reference, path in ((baseline_ref, baseline), (candidate_ref, candidate)):
        if path is None:
            print(f"No such run: {reference}", file=sys.stderr)
            print(f"Known runs: {', '.join(p.stem for p in files[-6:])}", file=sys.stderr)
            return 1
    assert baseline and candidate
    return compare(baseline, candidate)


if __name__ == "__main__":
    raise SystemExit(main())
