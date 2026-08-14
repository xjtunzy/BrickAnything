import json
import os

from BrickAnything.brick_data.brick_library import brick_dimensions_in_library
from BrickAnything.brick_data.brick_structure import Brick


def brick_sequence_dimension_violations(bricks: list) -> list[tuple[int, int, int]]:
    """Bricks as ``[h, w, x, y, z, ...]``.

    Returns a list ``(brick_index, h, w)`` for each brick that cannot be mapped via
    ``brick_library`` (missing from ``brick_library.json``). For malformed entries
    (fewer than 5 numbers), ``h`` and ``w`` are ``-1``.
    """

    bad: list[tuple[int, int, int]] = []
    for i, b in enumerate(bricks):
        if len(b) < 5:
            bad.append((i, -1, -1))
            continue
        h, w = int(b[0]), int(b[1])
        if not brick_dimensions_in_library(h, w):
            bad.append((i, h, w))
    return bad


def brick_sequence_dimensions_valid(bricks: list) -> bool:
    return len(brick_sequence_dimension_violations(bricks)) == 0


def seq2brick(seq, bricks):
    """Decode flatten ``x,y,z,xend,yend`` tokens into ``[h,w,x,y,z]`` bricks."""
    assert len(seq) % 5 == 0, f"len of seq is error:{len(seq)}"
    for i in range(0, len(seq) // 5):
        idx1 = 5 * i
        idx2 = 5 * i + 1
        idx3 = 5 * i + 2
        idx4 = 5 * i + 3
        idx5 = 5 * i + 4
        x = seq[idx1]
        y = seq[idx2]
        z = seq[idx3]
        h = seq[idx4] - x + 1
        w = seq[idx5] - y + 1
        bricks.append([h, w, x, y, z])


def brick2ldr(bricks, ldr_path):
    """Write bricks ``[h, w, x, y, z]`` (optional color) to an LDR file.

    Used by tree_mode inference: generation is tree-tokenized, but the final
    brick list is still the ``[h, w, x, y, z]`` layout expected here.
    """
    lines_out = []
    for b in bricks:
        if len(b) == 6:
            b_color = b[5]
        else:
            b_color = 14
        brick = Brick(h=b[0], w=b[1], x=b[2], y=b[3], z=b[4])
        line = brick.to_ldr(color=b_color)
        lines_out.append(line)

    with open(ldr_path, "w", encoding="utf-8") as f:
        f.writelines(lines_out)


if __name__ == "__main__":
    data_dir = os.environ.get("SEQ2BRICK_DATA_DIR", ".")
    out_dir = os.environ.get("SEQ2BRICK_OUT_DIR", ".")
    json_path = os.path.join(data_dir, "seq.json")
    ldr_path = os.path.join(out_dir, "brick.ldr")
    with open(json_path, "r") as f:
        seq = json.load(f)["Seq"]
    bricks = []
    seq2brick(seq, bricks)
    brick2ldr(bricks, ldr_path)
    print(f"Wrote {ldr_path}")
