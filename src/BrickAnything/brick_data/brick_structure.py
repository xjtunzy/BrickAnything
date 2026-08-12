import re
import warnings
from dataclasses import dataclass

import numpy as np

from BrickAnything.stability_analysis import stability_score, StabilityConfig, connectivity_score
from .brick_library import (brick_library,
                           dimensions_to_brick_id, brick_id_to_dimensions,
                           brick_id_to_part_id, part_id_to_brick_id)


@dataclass(frozen=True, order=True, kw_only=True)
class Brick:
    """
    Represents a 1-unit-tall rectangular brick.
    """
    h: int
    w: int
    x: int
    y: int
    z: int

    @property
    def brick_id(self) -> int:
        return dimensions_to_brick_id(self.h, self.w)

    @property
    def part_id(self) -> str:
        return brick_id_to_part_id(self.brick_id)

    @property
    def ori(self) -> int:
        return 1 if self.h > self.w else 0

    @property
    def area(self) -> int:
        return self.h * self.w

    @property
    def slice_2d(self) -> (slice, slice):
        return slice(self.x, self.x + self.h), slice(self.y, self.y + self.w)

    @property
    def slice(self) -> (slice, slice, int):
        return *self.slice_2d, self.z

    def __repr__(self):
        return self.to_txt()[:-1]

    def to_json(self) -> dict:
        return {
            'brick_id': self.brick_id,
            'x': self.x,
            'y': self.y,
            'z': self.z,
            'ori': self.ori,
        }

    def to_txt(self) -> str:
        return f'{self.h}x{self.w} ({self.x},{self.y},{self.z})\n'

    def to_ldr(self, base_height: float = 0, color: int = 115) -> str:
        x = (self.x + self.h * 0.5) * 20
        z = (self.y + self.w * 0.5) * 20
        y = (self.z + base_height) * -24
        matrix = '0 0 1 0 1 0 -1 0 0' if self.ori == 0 else '-1 0 0 0 1 0 0 0 -1'
        line = f'1 {color} {x} {y} {z} {matrix} {self.part_id}\n'
        step_line = '0 STEP\n'
        return line + step_line

    @classmethod
    def from_list(cls, brick_list: list[int]):
        return cls(h=brick_list[0], w=brick_list[1], x=brick_list[2], y=brick_list[3], z=brick_list[4])

    @classmethod
    def from_json(cls, brick_json: dict):
        h, w = brick_id_to_dimensions(brick_json['brick_id'])
        if brick_json['ori'] == 1:
            h, w = w, h
        x, y, z = brick_json['x'], brick_json['y'], brick_json['z']
        return cls(h=h, w=w, x=x, y=y, z=z)

    @classmethod
    def from_txt(cls, brick_txt: str):
        brick_txt = brick_txt.strip()
        match = re.fullmatch(r'(\d+)x(\d+) \((\d+),(\d+),(\d+)\)', brick_txt)
        if match is None:
            raise ValueError(f'Text Format brick is ill-formatted: {brick_txt}')

        h, w, x, y, z = map(int, match.group(1, 2, 3, 4, 5))
        return cls(h=h, w=w, x=x, y=y, z=z)

    @classmethod
    def from_ldr(cls, brick_ldr: str):
        ldr_components = brick_ldr.strip().split()
        match ldr_components:
            case ['1', _, x0, y0, z0, *matrix, part_id]:
                x0, y0, z0 = map(float, (x0, y0, z0))
                matrix_str = ' '.join(matrix)
                if matrix_str == '0 0 1 0 1 0 -1 0 0':
                    ori = 0
                elif matrix_str == '-1 0 0 0 1 0 0 0 -1':
                    ori = 1
                else:
                    raise ValueError(f'Invalid transformation matrix: {matrix_str}')

                h, w = brick_id_to_dimensions(part_id_to_brick_id(part_id))
                if ori == 1:
                    h, w = w, h

                x = int(x0 / 20 - h * 0.5)
                y = int(z0 / 20 - w * 0.5)
                z = int(-y0 / 24)

                return cls(h=h, w=w, x=x, y=y, z=z)
            case _:
                raise ValueError(f"LDR format is ill-formatted: {brick_ldr}")


class BrickStructure:
    """
    Represents a brick structure in the form of a list of bricks.
    """

    def __init__(self, bricks: list[Brick], world_dim: int = 20):
        self.world_dim = world_dim

        # Check if structure starts at ground level
        z0 = min((brick.z for brick in bricks), default=0)
        if z0 != 0:
            warnings.warn('Brick structure does not start at ground level z=0.')

        # Build structure from bricks
        self.bricks = []
        self.voxel_occupancy = np.zeros((world_dim, world_dim, world_dim), dtype=int)
        for brick in bricks:
            self.add_brick(brick)

    def __len__(self):
        return len(self.bricks)

    def __repr__(self):
        return self.to_txt()

    def __eq__(self, other) -> bool:
        if not isinstance(other, BrickStructure):
            return NotImplemented
        return self.bricks == other.bricks

    def to_json(self) -> dict:
        return {str(i + 1): brick.to_json() for i, brick in enumerate(self.bricks)}

    def to_txt(self) -> str:
        return ''.join([brick.to_txt() for brick in self.bricks])

    def to_ldr(self) -> str:
        return ''.join([brick.to_ldr() for brick in self.bricks])

    def add_brick(self, brick: Brick) -> None:
        self.bricks.append(brick)
        self.voxel_occupancy[brick.slice] += 1

    def undo_add_brick(self) -> None:
        brick = self.bricks[-1]
        self.voxel_occupancy[brick.slice] -= 1
        self.bricks.pop()

    def has_out_of_bounds_bricks(self) -> bool:
        return any(not self.brick_in_bounds(brick) for brick in self.bricks)

    def brick_in_bounds(self, brick: Brick) -> bool:
        return (all(slice_.start >= 0 and slice_.stop <= self.world_dim for slice_ in brick.slice_2d)
                and 0 <= brick.z < self.world_dim)

    def has_collisions(self) -> bool:
        return np.any(self.voxel_occupancy > 1)

    def brick_collides(self, brick: Brick) -> bool:
        return np.any(self.voxel_occupancy[brick.slice])

    def has_floating_bricks(self) -> bool:
        return any(self.brick_floats(brick) for brick in self.bricks)

    def brick_floats(self, brick: Brick) -> bool:
        if brick.z == 0:
            return False  # Supported by ground
        if np.any(self.voxel_occupancy[brick.slice_2d[0], brick.slice_2d[1], brick.z - 1]):
            return False  # Supported from below
        if brick.z != self.world_dim - 1 and np.any(self.voxel_occupancy[brick.slice_2d[0], brick.slice_2d[1], brick.z + 1]):
            return False  # Supported from above
        return True

    def is_stable(self) -> tuple[bool, np.ndarray]:
        if self.has_floating_bricks() or self.has_collisions():
            return False ,np.array([])
        scores = self.stability_scores()
        return scores.max() < 1 ,scores

    def stability_scores(self) -> np.ndarray:
        if self.has_collisions():
            raise ValueError('Cannot compute stability scores - structure has colliding bricks.')
        if self.has_out_of_bounds_bricks():
            raise ValueError('Cannot compute stability scores - structure has out of bounds bricks.')
        scores, _, _, _, _ = stability_score(self.to_json(), brick_library,
                                             StabilityConfig(world_dimension=(self.world_dim,) * 3))
        return scores

    def is_connected(self) -> bool:
        if self.has_floating_bricks() or self.has_collisions():
            print(f'structure has floating bricks or collisions')
            print(f'floating bricks: {self.has_floating_bricks()}')
            print(f'collisions: {self.has_collisions()}')
            return False
        return self.connectivity_scores().max() < 1

    def connectivity_scores(self) -> np.ndarray:
        if self.has_collisions():
            raise ValueError('Cannot compute connectivity scores - structure has colliding bricks.')
        if self.has_out_of_bounds_bricks():
            raise ValueError('Cannot compute connectivity scores - structure has out of bounds bricks.')
        scores = connectivity_score(self)
        return scores

    @classmethod
    def from_json(cls, bricks_json: dict):
        bricks = [Brick.from_json(v) for k, v in bricks_json.items() if k.isdigit()]
        return cls(bricks)

    @classmethod
    def from_txt(cls, bricks_txt: str):
        bricks_txt = bricks_txt.split('\n')
        bricks_txt = [b for b in bricks_txt if b.strip()]  # Remove blank lines
        bricks = [Brick.from_txt(brick) for brick in bricks_txt]
        return cls(bricks)

    @classmethod
    def from_list(cls, brick_list: list[int]):
        if len(brick_list) == 1:
            brick_list = brick_list[0]
        bricks = [Brick.from_list(brick) for brick in brick_list]
        return cls(bricks)

    @classmethod
    def from_ldr(cls, bricks_ldr: str):
        bricks_ldr = bricks_ldr.split('0 STEP')  # Split on step lines
        bricks_ldr = [b for b in bricks_ldr if b.strip()]  # Remove blank or whitespace-only lines
        bricks = [Brick.from_ldr(brick) for brick in bricks_ldr]
        return cls(bricks)
