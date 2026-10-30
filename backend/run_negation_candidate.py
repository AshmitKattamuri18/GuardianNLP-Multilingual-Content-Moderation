"""CLI for side-by-side original ensemble and optional scope-rule output."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from indicguard_runtime import IndicGuard
from negation_aware_candidate import predict_side_by_side


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", help="Text to score with the original ensemble and candidate")
    parser.add_argument("--bundle", type=Path,
                        default=Path(__file__).resolve().parent.parent / "guardian_nlp_dual_export",
                        help="Existing trained bundle directory (read-only)")
    parser.add_argument("--device", default="cpu", help="Inference device; defaults to cpu")
    args = parser.parse_args()
    model = IndicGuard(args.bundle, device=args.device)
    json.dump(predict_side_by_side(model, args.text), sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
