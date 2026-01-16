import argparse
import re
import sys
from pathlib import Path


SAMPLE_RE = re.compile(
    r"MNIST NN infer sample\s+(?P<sample>\d+)\s+pred\s+(?P<pred>\d+)\s+label\s+(?P<label>\d+)"
)


def iter_lines(path: Path | None) -> list[str]:
    if path is None:
        return sys.stdin.read().splitlines()
    return path.read_text(encoding="utf-8", errors="ignore").splitlines()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute MNIST inference accuracy from HOI4 game.log lines."
    )
    parser.add_argument(
        "log",
        nargs="?",
        help="Path to game.log (omit to read from stdin).",
    )
    args = parser.parse_args()

    path = Path(args.log) if args.log else None
    lines = iter_lines(path)

    total = 0
    correct = 0
    for line in lines:
        match = SAMPLE_RE.search(line)
        if not match:
            continue
        total += 1
        pred = int(match.group("pred"))
        label = int(match.group("label"))
        if pred == label:
            correct += 1

    if total == 0:
        print("No inference sample lines found.")
        return 1

    accuracy = correct / total
    print(f"total={total} correct={correct} accuracy={accuracy:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
