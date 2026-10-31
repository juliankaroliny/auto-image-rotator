#!/usr/bin/env python3
"""Detect which way is up in scanned photos and set the EXIF orientation tag.

Only the EXIF header is changed, usually just the 2-byte Orientation value.
The pixels are never touched, so there is no quality loss and no edge strip.
Runs as a dry run by default; pass --apply to write the tags.
"""
import argparse
import os
import shutil
import struct
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "orientation.onnx"
DEFAULT_FOLDER = HERE / "photos"

# ImageNet normalization, which the model was trained with
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# EXIF orientation values (how a viewer has to turn the stored image)
UPRIGHT, ROTATE_CW, ROTATE_180, ROTATE_CCW = 1, 6, 3, 8
DESCRIPTIONS = {
    UPRIGHT: "upright",
    ROTATE_CW: "rotate 90 CW",
    ROTATE_180: "rotate 180",
    ROTATE_CCW: "rotate 90 CCW",
}

# The model predicts how many times (0-3) an upright image was turned 90° counter-clockwise.
# Index = model class, value = the EXIF orientation that turns it back.
EXIF_FOR_CLASS = [UPRIGHT, ROTATE_CW, ROTATE_180, ROTATE_CCW]

# Choices for --rotate
MANUAL_ROTATIONS = {"upright": UPRIGHT, "cw": ROTATE_CW, "180": ROTATE_180, "ccw": ROTATE_CCW}

ORIENTATION_TAG = 0x0112
JPEG_START = b"\xff\xd8"


# --- Orientation detection -------------------------------------------------

def preprocess(image):
    """Resize to the 224x224 model input and normalize to a CHW float array."""
    pixels = np.asarray(image.resize((224, 224), Image.BILINEAR), dtype=np.float32) / 255
    return ((pixels - MEAN) / STD).transpose(2, 0, 1)


def softmax(logits):
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def predict(session, path):
    """Return (model class, confidence) for one image.

    The model sees the image in all four rotations and the results are averaged,
    which is more robust than a single prediction.
    """
    image = Image.open(path)
    image.draft("RGB", (448, 448))  # let the JPEG decoder downscale, much faster
    image = image.convert("RGB")

    batch = np.stack([preprocess(image.rotate(90 * turns, expand=True)) for turns in range(4)])
    probs = softmax(session.run(None, {"input": batch})[0])

    # Turning the input one more time counter-clockwise shifts the predicted class by one,
    # so shift each row back before averaging.
    averaged = np.mean([np.roll(probs[turns], -turns) for turns in range(4)], axis=0)
    best = int(averaged.argmax())
    return best, float(averaged[best])


# --- EXIF reading and writing ----------------------------------------------

def find_exif(data):
    """Return (segment_start, tiff_start) of the EXIF block in a JPEG, or None."""
    pos = 2  # skip the start-of-image marker (FF D8)
    while pos + 4 <= len(data) and data[pos] == 0xFF:
        marker = data[pos + 1]
        if marker == 0xE1 and data[pos + 4:pos + 10] == b"Exif\0\0":
            return pos, pos + 10
        if marker == 0xDA:  # start of the image data, no more headers after this
            return None
        pos += 2 + struct.unpack_from(">H", data, pos + 2)[0]
    return None


def first_directory(data, tiff_start):
    """Return (byte_order, file offset, number of entries) of the first EXIF directory."""
    # The EXIF payload is a small TIFF structure: byte order ("II" = little endian),
    # a magic number, then the offset of the first directory, which holds the tag.
    byte_order = "<" if data[tiff_start:tiff_start + 2] == b"II" else ">"
    directory = tiff_start + struct.unpack_from(byte_order + "I", data, tiff_start + 4)[0]
    return byte_order, directory, struct.unpack_from(byte_order + "H", data, directory)[0]


def find_orientation(data):
    """Find the Orientation tag in the bytes of a JPEG file.

    Returns (offset, byte_order, value), where offset is the file position of
    the 2-byte value, or None if there is no such tag.
    """
    exif = find_exif(data)
    if exif is None:
        return None
    byte_order, directory, count = first_directory(data, exif[1])
    for n in range(count):
        entry = directory + 2 + 12 * n  # tag (2), type (2), count (4), value (4)
        if struct.unpack_from(byte_order + "H", data, entry)[0] == ORIENTATION_TAG:
            return entry + 8, byte_order, struct.unpack_from(byte_order + "H", data, entry + 8)[0]
    return None


def orientation_entry(byte_order, value):
    # type 3 = SHORT, count 1, the value sits left-aligned in the 4-byte field
    return struct.pack(byte_order + "HHIHH", ORIENTATION_TAG, 3, 1, value, 0)


def with_orientation_entry(data, segment_start, tiff_start, value):
    """Return the JPEG with an Orientation entry added to its existing EXIF block.

    Inserting the entry into the first directory would shift everything after it
    and break the offsets inside the block. Instead, a copy of the directory with
    the extra entry goes to the end of the block and the header points to it.
    """
    byte_order, directory, count = first_directory(data, tiff_start)
    entries = [data[directory + 2 + 12 * n:directory + 14 + 12 * n] for n in range(count)]
    entries.append(orientation_entry(byte_order, value))
    entries.sort(key=lambda entry: struct.unpack_from(byte_order + "H", entry)[0])  # TIFF wants them sorted
    next_directory = data[directory + 2 + 12 * count:directory + 6 + 12 * count]

    segment_end = segment_start + 2 + struct.unpack_from(">H", data, segment_start + 2)[0]
    padding = b"\0" * ((segment_end - tiff_start) % 2)  # directories start at even offsets
    new_offset = segment_end + len(padding) - tiff_start
    new_directory = padding + struct.pack(byte_order + "H", len(entries)) + b"".join(entries) + next_directory

    new_length = segment_end + len(new_directory) - segment_start - 2
    if new_length > 0xFFFF:
        raise ValueError("EXIF block is too large to add the orientation tag")
    return b"".join([
        data[:segment_start + 2], struct.pack(">H", new_length),
        data[segment_start + 4:tiff_start + 4], struct.pack(byte_order + "I", new_offset),
        data[tiff_start + 8:segment_end], new_directory,
        data[segment_end:],
    ])


def with_new_exif_block(data, value):
    """Return the JPEG with a minimal EXIF block that only holds the orientation."""
    entries = [
        orientation_entry(">", value),
        struct.pack(">HHIHH", 0x0213, 3, 1, 1, 0),  # YCbCrPositioning = centered, required by the EXIF spec
    ]
    tiff = b"MM\0\x2a" + struct.pack(">IH", 8, len(entries)) + b"".join(entries) + struct.pack(">I", 0)
    payload = b"Exif\0\0" + tiff
    segment = b"\xff\xe1" + struct.pack(">H", 2 + len(payload)) + payload

    insert_at = 2
    if data[2:4] == b"\xff\xe0":  # keep a JFIF header first, where readers expect it
        insert_at = 4 + struct.unpack_from(">H", data, 4)[0]
    return data[:insert_at] + segment + data[insert_at:]


def set_orientation(path, data, value):
    tag = find_orientation(data)
    if tag:
        # Usual case: overwrite the existing 2-byte value in place
        offset, byte_order, _ = tag
        with open(path, "r+b") as f:
            f.seek(offset)
            f.write(struct.pack(byte_order + "H", value))
        return

    exif = find_exif(data)
    data = with_orientation_entry(data, *exif, value) if exif else with_new_exif_block(data, value)
    # Write next to the original first, so an interruption can't leave a broken photo behind
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    shutil.copymode(path, tmp)
    os.replace(tmp, path)


# --- Command line ---------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="*", type=Path,
                        help="JPEG files or folders, searched recursively (default: the photos folder)")
    parser.add_argument("--apply", action="store_true", help="write the EXIF tag (default: dry run)")
    parser.add_argument("--min-confidence", type=float, default=0.8,
                        help="leave images below this confidence unchanged (default 0.8)")
    parser.add_argument("--force", action="store_true",
                        help="also change files whose tag is already set to something other than upright")
    parser.add_argument("--rotate", choices=MANUAL_ROTATIONS,
                        help="set the orientation by hand for the given files (no detection, written immediately)")
    args = parser.parse_args()

    if not args.paths:
        if args.rotate:
            parser.error("--rotate needs the file name(s) to change")
        args.paths = [DEFAULT_FOLDER]
        if DEFAULT_FOLDER.is_relative_to(Path.cwd()):  # keeps the printed paths short
            args.paths = [DEFAULT_FOLDER.relative_to(Path.cwd())]
    for path in args.paths:
        if not path.exists():
            parser.error(f"{path} does not exist")
    return args


def collect_jpegs(paths):
    """Expand folders (recursively) into their JPEG files. Files are kept as given."""
    files = []
    for path in paths:
        if path.is_dir():
            files += sorted(p for p in path.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg"))
        else:
            files.append(path)
    return files


def rotate_by_hand(files, value):
    for path in files:
        data = path.read_bytes()
        if not data.startswith(JPEG_START):
            print(f"SKIP   {path}: not a JPEG")
            continue
        print(f"SET    {path}: {DESCRIPTIONS[value]} (manual)")
        set_orientation(path, data, value)


def auto_rotate(files, apply, min_confidence, force):
    session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    unsure = []
    total = len(files)

    for n, path in enumerate(files, 1):
        progress = f"[{n:{len(str(total))}}/{total}]"
        data = path.read_bytes()
        if not data.startswith(JPEG_START):
            print(f"{progress} SKIP   {path}: not a JPEG")
            continue
        tag = find_orientation(data)
        current = tag[2] if tag else UPRIGHT  # no tag means upright
        if current != UPRIGHT and not force:
            print(f"{progress} SKIP   {path}: already has orientation {current} (use --force)")
            continue

        model_class, confidence = predict(session, path)
        new = EXIF_FOR_CLASS[model_class]
        result = f"{DESCRIPTIONS[new]} ({confidence:.0%})"

        if confidence < min_confidence:
            guess = f"{path}: {DESCRIPTIONS[new]}? ({confidence:.0%})"
            print(f"{progress} UNSURE {guess} - left unchanged")
            unsure.append(guess)
        elif new == current:
            print(f"{progress} OK     {path}: {result}")
        elif apply:
            print(f"{progress} SET    {path}: {result}")
            set_orientation(path, data, new)
        else:
            print(f"{progress} WOULD  {path}: {result}")

    if unsure:
        print(f"\n{len(unsure)} image(s) need a manual look:")
        for guess in unsure:
            print(f"  {guess}")
        print("Turn them with e.g.:")
        print("  ./run.sh --rotate cw|ccw|180 <file>")
    if not apply:
        print("\nDry run - nothing written. Re-run with --apply to set the tags.")


def main():
    args = parse_args()
    files = collect_jpegs(args.paths)
    if args.rotate:
        rotate_by_hand(files, MANUAL_ROTATIONS[args.rotate])
    else:
        auto_rotate(files, args.apply, args.min_confidence, args.force)


if __name__ == "__main__":
    main()
