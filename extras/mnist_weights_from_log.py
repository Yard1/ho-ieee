import argparse
import re
from decimal import Decimal, getcontext
from pathlib import Path
from typing import Dict, List, Optional, Tuple


LINE_RE = re.compile(
    r"MNIST NN W(?P<layer>[12]) idx=(?P<idx>\d+) val=(?P<val>[+-]?\d+(?:\.\d+)?)"
)


def parse_dump(log_path: Path, on_duplicate: str) -> Tuple[Dict[int, str], Dict[int, str]]:
    w1: Dict[int, str] = {}
    w2: Dict[int, str] = {}
    with log_path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            match = LINE_RE.search(line)
            if not match:
                continue
            layer = match.group("layer")
            idx = int(match.group("idx"))
            value = match.group("val")
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
    values: Dict[int, str],
    size: Optional[int],
    fill_missing: bool,
    label: str,
) -> List[str]:
    if not values:
        raise ValueError(f"No weights found for {label}.")
    max_idx = max(values)
    if size is None:
        size = max_idx + 1
    if max_idx >= size:
        raise ValueError(
            f"{label} has max idx {max_idx} which exceeds provided size {size}."
        )
    dense: List[Optional[str]] = [None] * size
    for idx, value in values.items():
        dense[idx] = value
    if fill_missing:
        for i in range(size):
            if dense[i] is None:
                dense[i] = "0"
    else:
        missing = [i for i, value in enumerate(dense) if value is None]
        if missing:
            sample = ", ".join(str(i) for i in missing[:10])
            more = "" if len(missing) <= 10 else f" (and {len(missing) - 10} more)"
            raise ValueError(
                f"{label} missing {len(missing)} indices starting at {sample}{more}."
            )
    return [value if value is not None else "0" for value in dense]


def emit_effect(
    out_path: Path,
    effect_name: str,
    w1_values: List[str],
    w2_values: List[str],
    minify: bool,
    include_header: bool,
    include_setup: bool,
    skip_zero: bool,
) -> None:
    lines: List[str] = []
    if include_header and not minify:
        lines.append("## Generated from mnist_nn_dump_weights log output.")
        lines.append("")

    if minify:
        lines.append(f"{effect_name}={{")
    else:
        lines.append(f"{effect_name} = {{")

    if include_setup:
        if minify:
            lines.append("mnist_nn_setup=yes")
        else:
            lines.append("    mnist_nn_setup = yes")

    if minify:
        lines.append("clear_array=global.nn_w1")
        lines.append(
            f"resize_array={{array=global.nn_w1 size={len(w1_values)} value=0}}"
        )
        lines.append("clear_array=global.nn_w2")
        lines.append(
            f"resize_array={{array=global.nn_w2 size={len(w2_values)} value=0}}"
        )
    else:
        lines.append("    # Allocate arrays (array elements must exist before setting).")
        lines.append("    clear_array = global.nn_w1")
        lines.append("    resize_array = {")
        lines.append("        array = global.nn_w1")
        lines.append(f"        size = {len(w1_values)}")
        lines.append("        value = 0")
        lines.append("    }")
        lines.append("    clear_array = global.nn_w2")
        lines.append("    resize_array = {")
        lines.append("        array = global.nn_w2")
        lines.append(f"        size = {len(w2_values)}")
        lines.append("        value = 0")
        lines.append("    }")

    if not minify:
        lines.append("")
        lines.append("    # W1 weights (input -> hidden)")
    for idx, value in enumerate(w1_values):
        if skip_zero and Decimal(value) == 0:
            continue
        if minify:
            lines.append(f"set_variable={{global.nn_w1^{idx}={value}}}")
        else:
            lines.append(f"    set_variable = {{ global.nn_w1^{idx} = {value} }}")

    if not minify:
        lines.append("")
        lines.append("    # W2 weights (hidden -> output)")
    for idx, value in enumerate(w2_values):
        if skip_zero and Decimal(value) == 0:
            continue
        if minify:
            lines.append(f"set_variable={{global.nn_w2^{idx}={value}}}")
        else:
            lines.append(f"    set_variable = {{ global.nn_w2^{idx} = {value} }}")

    lines.append("}")
    out_path.write_text("\n".join(lines), encoding="ascii")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Convert mnist_nn_dump_weights log output into a scripted effect file."
        )
    )
    parser.add_argument(
        "log",
        help="Path to game.log or a weights dump file containing MNIST NN W1/W2 lines.",
    )
    parser.add_argument(
        "--output",
        default="common/scripted_effects/mnist_nn_weights_generated.txt",
        help="Output scripted effect file path.",
    )
    parser.add_argument(
        "--effect-name",
        default="mnist_nn_load_weights",
        help="Scripted effect name to generate.",
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
        "--skip-zero",
        action="store_true",
        help="Skip emitting assignments for zero values.",
    )
    parser.add_argument(
        "--minify",
        action="store_true",
        help="Minify output by removing whitespace and comments.",
    )
    parser.add_argument(
        "--no-header",
        action="store_true",
        help="Skip the header comments in the output file.",
    )
    parser.add_argument(
        "--include-setup",
        action="store_true",
        help="Include a mnist_nn_setup call at the top of the effect.",
    )
    args = parser.parse_args()

    log_path = Path(args.log)
    out_path = Path(args.output)

    getcontext().prec = 32

    w1, w2 = parse_dump(log_path, args.on_duplicate)
    w1_values = build_dense(w1, args.w1_size, args.fill_missing, "W1")
    w2_values = build_dense(w2, args.w2_size, args.fill_missing, "W2")

    emit_effect(
        out_path=out_path,
        effect_name=args.effect_name,
        w1_values=w1_values,
        w2_values=w2_values,
        minify=args.minify,
        include_header=not args.no_header,
        include_setup=args.include_setup,
        skip_zero=args.skip_zero,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
