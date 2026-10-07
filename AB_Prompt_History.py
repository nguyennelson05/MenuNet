import argparse
import csv
from datetime import datetime
import os
import random
import sys

import numpy as np
import torch

import Auction


def _configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parse_distribution(value: str) -> list[float]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("distribution must be two comma-separated numbers, e.g. 1,2")
    try:
        values = [float(parts[0]), float(parts[1])]
        normalized = []
        for v in values:
            normalized.append(int(v) if float(v).is_integer() else v)
        return normalized
    except ValueError as exc:
        raise argparse.ArgumentTypeError("distribution values must be numbers") from exc


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _append_rows(output_path: str, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    _ensure_parent_dir(output_path)
    write_header = not os.path.exists(output_path)
    with open(output_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def run_ab(
    *,
    runs: int,
    seed_start: int,
    rounds: int,
    distribution: list[float],
    model: str,
    reasoning: str,
    rule: bool,
    history_limit: int | None,
    output_path: str,
) -> None:
    # 2x2: prompt format (tuple-only vs reason+bid) x truncation (off/on)
    conditions = [
        {"prompt_type": "MenuNet old", "truncate_history": False, "label": "tuple_only | untruncated"},
        {"prompt_type": "MenuNet", "truncate_history": False, "label": "reason+bid | untruncated"},
        {"prompt_type": "MenuNet old", "truncate_history": True, "label": "tuple_only | truncated"},
        {"prompt_type": "MenuNet", "truncate_history": True, "label": "reason+bid | truncated"},
    ]

    now = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    fieldnames = [
        "date",
        "seed",
        "rule",
        "hist",
        "same_bidder",
        "diff_items",
        "distribution",
        "model",
        "reasoning",
        "prompt_type",
        "truncate_history",
        "history_limit",
        "rounds",
        "truth_count",
        "truth_rate",
        "regret",
        "condition_label",
    ]

    for run_idx in range(runs):
        seed = seed_start + run_idx
        for cond in conditions:
            _set_seed(seed)
            auction = Auction.Auction(round=rounds, distribution=distribution)
            truth_count, regret = auction.auction(
                rule=rule,
                hist=True,
                same_bidder=True,
                diff_items=False,
                model=model,
                reasoning=reasoning,
                auction_type=cond["prompt_type"],
                truncate_history=cond["truncate_history"],
                history_limit=history_limit,
                verbose=False,
            )
            row: dict[str, object] = {
                "date": now,
                "seed": seed,
                "rule": rule,
                "hist": True,
                "same_bidder": True,
                "diff_items": False,
                "distribution": f"{distribution[0]},{distribution[1]}",
                "model": model,
                "reasoning": reasoning,
                "prompt_type": cond["prompt_type"],
                "truncate_history": cond["truncate_history"],
                "history_limit": history_limit if cond["truncate_history"] else "",
                "rounds": rounds,
                "truth_count": truth_count,
                "truth_rate": (truth_count / rounds) if rounds else 0,
                "regret": regret,
                "condition_label": cond["label"],
            }
            _append_rows(output_path, fieldnames, [row])
            print(
                f"seed={seed} {cond['label']}: truth={truth_count}/{rounds} ({row['truth_rate']:.3f}), regret={regret:.4f}"
            )

    print(f"\nSaved: {output_path}")


def main() -> None:
    _configure_stdout()

    parser = argparse.ArgumentParser(
        description="A/B test: prompt variant (MenuNet vs MenuNet old) x history truncation on truthfulness."
    )
    parser.add_argument("--rounds", type=int, default=30, help="Rounds per run (default: 30)")
    parser.add_argument("--runs", type=int, default=30, help="Number of runs (default: 30)")
    parser.add_argument("--seed-start", type=int, default=0, help="Starting seed (default: 0)")
    parser.add_argument("--distribution", type=_parse_distribution, default="1,2", help="e.g. 1,2")
    parser.add_argument("--model", default="gpt-5-mini", help="Model (default: gpt-5-mini)")
    parser.add_argument("--reasoning", default="minimal", help="Reasoning effort (default: minimal)")
    parser.add_argument("--rule", action="store_true", help="Enable rule text (default: off)")
    parser.add_argument(
        "--history-limit",
        type=int,
        default=None,
        help="Override history truncation limit when enabled (default: uses Auction.py logic)",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(__file__), "Auction_Results", "ab_prompt_history.csv"),
        help="Output CSV path",
    )

    args = parser.parse_args()
    if args.rounds <= 0:
        raise SystemExit("--rounds must be > 0")
    if args.runs <= 0:
        raise SystemExit("--runs must be > 0")

    run_ab(
        runs=args.runs,
        seed_start=args.seed_start,
        rounds=args.rounds,
        distribution=args.distribution,
        model=args.model,
        reasoning=args.reasoning,
        rule=bool(args.rule),
        history_limit=args.history_limit,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
