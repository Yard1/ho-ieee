import argparse
import gzip
import struct
from pathlib import Path

import numpy as np
import requests


MNIST_BASE_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
MNIST_FILES = {
    "training_images": "train-images-idx3-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "training_labels": "train-labels-idx1-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def float_to_pdxvar(value: float) -> float:
    bits = struct.unpack(">l", struct.pack(">f", value))[0]
    return bits / 1000.0


def load_mnist(data_dir: Path) -> dict:
    data_dir.mkdir(parents=True, exist_ok=True)
    for fname in MNIST_FILES.values():
        fpath = data_dir / fname
        if not fpath.exists():
            resp = requests.get(MNIST_BASE_URL + fname, stream=True, timeout=60)
            resp.raise_for_status()
            with open(fpath, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=128):
                    fh.write(chunk)

    mnist_dataset = {}
    for key in ("training_images", "test_images"):
        with gzip.open(data_dir / MNIST_FILES[key], "rb") as mnist_file:
            mnist_dataset[key] = np.frombuffer(
                mnist_file.read(), np.uint8, offset=16
            ).reshape(-1, 28, 28)
    for key in ("training_labels", "test_labels"):
        with gzip.open(data_dir / MNIST_FILES[key], "rb") as mnist_file:
            mnist_dataset[key] = np.frombuffer(mnist_file.read(), np.uint8, offset=8)
    return mnist_dataset


def downsample_14x14(images_28x28: np.ndarray) -> np.ndarray:
    reshaped = images_28x28.reshape(-1, 14, 2, 14, 2)
    return reshaped.mean(axis=(2, 4))


def one_hot(labels: np.ndarray, dimension: int = 10) -> np.ndarray:
    out = labels[..., None] == np.arange(dimension)[None]
    return out.astype(np.float32)


def format_pdxvar(value: float) -> str:
    return f"{value:.3f}"


def emit_effect(
    out_path: Path,
    effect_name: str,
    images: np.ndarray,
    labels: np.ndarray,
    skip_zero: bool,
    minify: bool,
    sample_offset: int,
    include_header: bool,
) -> None:
    num_samples = images.shape[0]
    input_size = images.shape[1]
    output_size = labels.shape[1]

    if minify and include_header:
        ti = "a"
        tl = "b"
        lines = [
            f"{effect_name}={{",
            f"set_variable={{global.nn_train_samples={num_samples}}}",
            f"set_temp_variable={{{ti}=global.nn_train_samples}}",
            f"multiply_temp_variable={{{ti}=global.nn_input_size}}",
            "clear_array=global.nn_train_images",
            "resize_array={array=global.nn_train_images size=a value=global.nn_f0}",
            f"set_temp_variable={{{tl}=global.nn_train_samples}}",
            f"multiply_temp_variable={{{tl}=global.nn_output_size}}",
            "clear_array=global.nn_train_labels",
            "resize_array={array=global.nn_train_labels size=b value=global.nn_f0}",
        ]
    elif include_header:
        lines = []
        lines.append("## Generated MNIST dataset (14x14) for training/inference")
        lines.append("## Usage:")
        lines.append(
            "##  - Inputs: global.nn_input_size, global.nn_output_size, global.nn_f0, global.nn_f1"
        )
        lines.append(
            "##  - Outputs: global.nn_train_samples, global.nn_train_images, global.nn_train_labels"
        )
        lines.append(f"{effect_name} = {{")
        lines.append(f"    set_variable = {{ global.nn_train_samples = {num_samples} }}")
        lines.append("")
        lines.append("    set_temp_variable = { total_images = global.nn_train_samples }")
        lines.append("    multiply_temp_variable = { total_images = global.nn_input_size }")
        lines.append("    clear_array = global.nn_train_images")
        lines.append("    resize_array = {")
        lines.append("        array = global.nn_train_images")
        lines.append("        size = total_images")
        lines.append("        value = global.nn_f0")
        lines.append("    }")
        lines.append("")
        lines.append("    set_temp_variable = { total_labels = global.nn_train_samples }")
        lines.append("    multiply_temp_variable = { total_labels = global.nn_output_size }")
        lines.append("    clear_array = global.nn_train_labels")
        lines.append("    resize_array = {")
        lines.append("        array = global.nn_train_labels")
        lines.append("        size = total_labels")
        lines.append("        value = global.nn_f0")
        lines.append("    }")
        lines.append("")
        lines.append("    # Images")
    else:
        if minify:
            lines = [f"{effect_name}={{"]
        else:
            lines = [f"{effect_name} = {{"]

    flat_images = images.reshape(num_samples * input_size)
    for idx, value in enumerate(flat_images):
        if skip_zero and value == 0.0:
            continue
        pdx = float_to_pdxvar(float(value))
        out_idx = sample_offset * input_size + idx
        if minify:
            lines.append(
                f"set_variable={{global.nn_train_images^{out_idx}={format_pdxvar(pdx)}}}"
            )
        else:
            lines.append(
                f"    set_variable = {{ global.nn_train_images^{out_idx} = {format_pdxvar(pdx)} }}"
            )

    if not minify and include_header:
        lines.append("")
        lines.append("    # Labels (one-hot)")
    flat_labels = labels.reshape(num_samples * output_size)
    for idx, value in enumerate(flat_labels):
        if value == 0.0:
            continue
        out_idx = sample_offset * output_size + idx
        if minify:
            lines.append(f"set_variable={{global.nn_train_labels^{out_idx}=global.nn_f1}}")
        else:
            lines.append(
                f"    set_variable = {{ global.nn_train_labels^{out_idx} = global.nn_f1 }}"
            )

    lines.append("}")
    out_path.write_text("\n".join(lines), encoding="ascii")


def export_split(
    images: np.ndarray,
    labels: np.ndarray,
    out_path: Path,
    effect_name: str,
    sample_count: int,
    parts: int,
    skip_zero: bool,
    minify: bool,
) -> None:
    if parts < 1:
        raise SystemExit("--parts must be >= 1")

    if parts == 1:
        emit_effect(
            out_path=out_path,
            effect_name=effect_name,
            images=images,
            labels=labels,
            skip_zero=skip_zero,
            minify=minify,
            sample_offset=0,
            include_header=True,
        )
        return

    base_name = out_path.stem
    suffix = out_path.suffix or ".txt"
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    samples_per_part = (sample_count + parts - 1) // parts
    wrapper_lines = []
    wrapper_effect_name = f"{effect_name}_all"
    wrapper_lines.append("## Wrapper to load all MNIST data parts")
    wrapper_lines.append(f"{wrapper_effect_name} = {{")
    wrapper_lines.append(f"    set_variable = {{ global.nn_train_samples = {sample_count} }}")
    wrapper_lines.append("    set_variable = { global.nn_sparse_ready = 0 }")
    wrapper_lines.append("    set_temp_variable = { total_images = global.nn_train_samples }")
    wrapper_lines.append("    multiply_temp_variable = { total_images = global.nn_input_size }")
    wrapper_lines.append("    clear_array = global.nn_train_images")
    wrapper_lines.append("    resize_array = {")
    wrapper_lines.append("        array = global.nn_train_images")
    wrapper_lines.append("        size = total_images")
    wrapper_lines.append("        value = global.nn_f0")
    wrapper_lines.append("    }")
    wrapper_lines.append("    set_temp_variable = { total_labels = global.nn_train_samples }")
    wrapper_lines.append("    multiply_temp_variable = { total_labels = global.nn_output_size }")
    wrapper_lines.append("    clear_array = global.nn_train_labels")
    wrapper_lines.append("    resize_array = {")
    wrapper_lines.append("        array = global.nn_train_labels")
    wrapper_lines.append("        size = total_labels")
    wrapper_lines.append("        value = global.nn_f0")
    wrapper_lines.append("    }")

    for part_idx in range(parts):
        start = part_idx * samples_per_part
        end = min(start + samples_per_part, sample_count)
        if start >= end:
            break

        part_images = images[start:end]
        part_labels = labels[start:end]

        part_effect = f"{effect_name}_part_{part_idx}"
        part_path = out_dir / f"{base_name}.part{part_idx}{suffix}"
        emit_effect(
            out_path=part_path,
            effect_name=part_effect,
            images=part_images,
            labels=part_labels,
            skip_zero=skip_zero,
            minify=minify,
            sample_offset=start,
            include_header=False,
        )

        wrapper_lines.append(f"    {part_effect} = yes")

    wrapper_lines.append("}")
    out_path.write_text("\n".join(wrapper_lines), encoding="ascii")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export MNIST as a HOI4 scripted effect for this mod."
    )
    parser.add_argument(
        "--data-dir",
        default="_data",
        help="Directory to download/cache MNIST files.",
    )
    parser.add_argument(
        "--output",
        default="common/scripted_effects/mnist_data_generated.txt",
        help="Output scripted effect file path.",
    )
    parser.add_argument(
        "--effect-name",
        default="mnist_nn_load_data",
        help="Scripted effect name to generate.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Number of training samples to export.",
    )
    parser.add_argument(
        "--skip-zero",
        action="store_true",
        help="Skip zero-valued pixels to reduce file size.",
    )
    parser.add_argument(
        "--minify",
        action="store_true",
        help="Minify output by removing whitespace and shortening temp names.",
    )
    parser.add_argument(
        "--parts",
        type=int,
        default=1,
        help="Split output into N parts with a wrapper effect.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_path = Path(args.output)

    mnist = load_mnist(data_dir)
    actual_samples = mnist["training_images"].shape[0]
    sample_count = min(args.samples, actual_samples)
    images = mnist["training_images"][:sample_count]
    labels = mnist["training_labels"][:sample_count]

    images_14 = downsample_14x14(images).astype(np.float32)
    images_14 = images_14 / 255.0
    flat_images = images_14.reshape(images_14.shape[0], 14 * 14)

    one_hot_labels = one_hot(labels)

    export_split(
        images=flat_images,
        labels=one_hot_labels,
        out_path=out_path,
        effect_name=args.effect_name,
        sample_count=sample_count,
        parts=args.parts,
        skip_zero=args.skip_zero,
        minify=args.minify,
    )

    test_images = mnist["test_images"]
    test_labels = mnist["test_labels"]
    test_sample_count = min(args.samples, test_images.shape[0])
    test_images_14 = downsample_14x14(test_images[:test_sample_count]).astype(
        np.float32
    )
    test_images_14 = test_images_14 / 255.0
    test_flat_images = test_images_14.reshape(test_images_14.shape[0], 14 * 14)
    test_one_hot_labels = one_hot(test_labels[:test_sample_count])

    test_out_path = out_path.with_name(
        f"{out_path.stem}_test{out_path.suffix or '.txt'}"
    )
    test_effect_name = f"{args.effect_name}_test"
    export_split(
        images=test_flat_images,
        labels=test_one_hot_labels,
        out_path=test_out_path,
        effect_name=test_effect_name,
        sample_count=test_sample_count,
        parts=args.parts,
        skip_zero=args.skip_zero,
        minify=args.minify,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
