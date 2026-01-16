import argparse
import re
import struct
from decimal import Decimal, ROUND_HALF_UP, getcontext
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


LINE_RE = re.compile(
    r"MNIST NN W(?P<layer>[12]) idx=(?P<idx>\d+) val=(?P<val>[+-]?\d+(?:\.\d+)?)"
)


def parse_dump(log_path: Path, on_duplicate: str) -> Tuple[Dict[int, Decimal], Dict[int, Decimal]]:
    w1: Dict[int, Decimal] = {}
    w2: Dict[int, Decimal] = {}
    with log_path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            match = LINE_RE.search(line)
            if not match:
                continue
            layer = match.group("layer")
            idx = int(match.group("idx"))
            value = Decimal(match.group("val"))
            target = w1 if layer == "1" else w2
            if idx in target:
                if on_duplicate == "first":
                    continue
                if on_duplicate == "error":
                    raise ValueError(
                        f"Duplicate weight for W{layer} idx={idx} while parsing {log_path}"
                    )
            target[idx] = value
    return w1, w2


def build_dense(
    values: Dict[int, Decimal],
    size: Optional[int],
    fill_missing: bool,
    label: str,
) -> List[Decimal]:
    if not values:
        raise ValueError(f"No weights found for {label}.")
    max_idx = max(values)
    if size is None:
        size = max_idx + 1
    if max_idx >= size:
        raise ValueError(
            f"{label} has max idx {max_idx} which exceeds provided size {size}."
        )
    dense: List[Optional[Decimal]] = [None] * size
    for idx, value in values.items():
        dense[idx] = value
    if fill_missing:
        for i in range(size):
            if dense[i] is None:
                dense[i] = Decimal(0)
    else:
        missing = [i for i, value in enumerate(dense) if value is None]
        if missing:
            sample = ", ".join(str(i) for i in missing[:10])
            more = "" if len(missing) <= 10 else f" (and {len(missing) - 10} more)"
            raise ValueError(
                f"{label} missing {len(missing)} indices starting at {sample}{more}."
            )
    return [value if value is not None else Decimal(0) for value in dense]


def pdxvar_to_float(value: Decimal) -> float:
    bits = int((value * Decimal(1000)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    return struct.unpack(">f", struct.pack(">l", bits))[0]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert mnist_nn_dump_weights log output to NumPy-friendly arrays."
    )
    parser.add_argument(
        "log",
        help="Path to game.log or a weights dump file containing MNIST NN W1/W2 lines.",
    )
    parser.add_argument(
        "--output",
        default="extras/mnist_weights.npz",
        help="Output .npz path.",
    )
    parser.add_argument(
        "--on-duplicate",
        choices=("first", "last", "error"),
        default="last",
        help="How to handle duplicate indices in the log.",
    )
    parser.add_argument(
        "--w1-size",
        type=int,
        default=None,
        help="Override W1 array size instead of using max index + 1.",
    )
    parser.add_argument(
        "--w2-size",
        type=int,
        default=None,
        help="Override W2 array size instead of using max index + 1.",
    )
    parser.add_argument(
        "--fill-missing",
        action="store_true",
        help="Fill missing indices with 0 instead of erroring.",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=None,
        help="Input size (for reshaping W1).",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=None,
        help="Hidden size (for reshaping W1/W2).",
    )
    parser.add_argument(
        "--output-size",
        type=int,
        default=None,
        help="Output size (for reshaping W2).",
    )
    parser.add_argument(
        "--emit-transpose",
        action="store_true",
        help="Emit transposed matrices (PyTorch-style) as w1_t and w2_t.",
    )
    args = parser.parse_args()

    getcontext().prec = 32

    log_path = Path(args.log)
    out_path = Path(args.output)

    w1_map, w2_map = parse_dump(log_path, args.on_duplicate)
    w1_vals = build_dense(w1_map, args.w1_size, args.fill_missing, "W1")
    w2_vals = build_dense(w2_map, args.w2_size, args.fill_missing, "W2")

    w1_float = np.array([pdxvar_to_float(value) for value in w1_vals], dtype=np.float32)
    w2_float = np.array([pdxvar_to_float(value) for value in w2_vals], dtype=np.float32)

    payload = {
        "w1_flat": w1_float,
        "w2_flat": w2_float,
    }

    if args.input_size is not None and args.hidden_size is not None:
        expected = args.input_size * args.hidden_size
        if w1_float.size != expected:
            raise SystemExit(
                f"W1 size {w1_float.size} does not match input_size*hidden_size {expected}."
            )
        w1 = w1_float.reshape(args.input_size, args.hidden_size)
        payload["w1"] = w1
        if args.emit_transpose:
            payload["w1_t"] = w1.T

    if args.hidden_size is not None and args.output_size is not None:
        expected = args.hidden_size * args.output_size
        if w2_float.size != expected:
            raise SystemExit(
                f"W2 size {w2_float.size} does not match hidden_size*output_size {expected}."
            )
        w2 = w2_float.reshape(args.hidden_size, args.output_size)
        payload["w2"] = w2
        if args.emit_transpose:
            payload["w2_t"] = w2.T

    np.savez(out_path, **payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
