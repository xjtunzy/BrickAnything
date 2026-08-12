import copy
import json
import logging
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from torch import nn
from transformers import AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList

from BrickAnything.brick_data.brick_structure import BrickStructure
from BrickAnything.miche.encode import load_model
from BrickAnything.models.shape_opt_dynamic import ShapeOPTConfig

logger = logging.getLogger(__name__)


@dataclass
class Brick:
    """Decoded brick with both anchor (x, y, z) and footprint (h, w).

    Mirrors the dataclass in ``data_process/convert_new_seq.py``.
    """

    idx: int
    x: int
    y: int
    z: int
    h: int
    w: int
    parent_idx: Optional[int] = None
    token_span: Tuple[int, int] = (0, 0)


class AllowedTokensProcessor(LogitsProcessor):
    """Force sampling to stay inside ``allowed_token_ids``.

    Implementation note: we set logits of all other tokens to ``-inf`` so
    softmax assigns them zero probability. The processor must receive a
    non-empty allowed set.
    """

    def __init__(self, allowed_token_ids: List[int]):
        super().__init__()
        if not allowed_token_ids:
            raise ValueError("allowed_token_ids must not be empty.")
        self.allowed_token_ids = list(allowed_token_ids)

    def __call__(self, input_ids, scores):
        mask = torch.full_like(scores, float("-inf"))
        idx = torch.tensor(self.allowed_token_ids, dtype=torch.long, device=scores.device)
        mask.index_fill_(1, idx, 0.0)
        return scores + mask


class BrickAnything(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.point_encoder = load_model(ckpt_path=None)

        self.n_discrete_size = cfg['n_discrete_size']
        self.cond_length = cfg['cond_length']
        self.cond_dim = cfg['cond_dim']
        self.pad_id = cfg['pad_id']
        self.mask = cfg['mask']
        self.sampling = cfg['sampling']
        self.max_bricks = cfg['n_max_bricks']
        assert self.sampling is True
        self.top_k = cfg['top_k']
        self.top_p = cfg['top_p']
        self.temperature = cfg['temperature']
        self.max_generation_num = cfg['max_generation_num']
        self.max_rollback_num = cfg['max_rollback_num']
        self.llm_name = cfg['llm_name']

        self.max_length = int(cfg['max_length'] + 3 + self.cond_length)

        self.config = ShapeOPTConfig.from_pretrained(
            self.llm_name,
            n_positions=self.max_length,
            max_position_embeddings=self.max_length,
            vocab_size=self.n_discrete_size + 45,
            _attn_implementation="flash_attention_2"
        )

        # Reserved special tokens.
        self.bos_token_id = 0
        self.eos_token_id = 1
        self.pad_token_id = 2

        # Token ranges. All ids below are *model-space* ids
        # (= raw value + 3, leaving room for BOS/EOS/PAD).
        self.xyz_offset = 3
        self.xyz_start = 3
        self.xyz_end = 22

        self.f_offset = 23
        self.f_start = 23
        self.f_end = 46

        self.m_offset = 47
        self.m_start = 47
        self.m_end = 58

        self.hw_start = 59
        self.hw_end = 63

        self.eop_token_id = 64
        self.vocab_size = 65

        # Bidirectional mapping between hw token id (model space) and physical size.
        self.hw_token_to_size = {
            59: 1,
            60: 2,
            61: 4,
            62: 6,
            63: 8,
        }
        self.size_to_hw_token = {v: k for k, v in self.hw_token_to_size.items()}

        self.config.bos_token_id = self.bos_token_id
        self.config.eos_token_id = self.eos_token_id
        self.config.pad_token_id = self.pad_token_id
        self.config._attn_implementation = "flash_attention_2"
        self.config.n_discrete_size = self.n_discrete_size
        self.config.cond_length = self.cond_length

        if self.config.word_embed_proj_dim != self.config.hidden_size:
            self.config.word_embed_proj_dim = self.config.hidden_size
        self.transformer = AutoModelForCausalLM.from_config(
            config=self.config,
            attn_implementation="flash_attention_2",
        )
        self.cond_head_proj = nn.Linear(self.cond_dim, self.config.word_embed_proj_dim)
        self.cond_proj = nn.Linear(self.cond_dim * 2, self.config.word_embed_proj_dim)

        # Generation state.
        self.generated_ids: Optional[torch.Tensor] = None
        self.kv_cache = None
        self.feature_cache = None
        self.feature_cache_org = None

        # Saved snapshots (rollback safety net; not used in the main mask-driven path).
        self.generated_ids_saved = None
        self.kv_cache_saved = None
        self.voxel_occupancy_saved = None
        self.bricks_saved: Optional[List[Brick]] = None
        self.bfs_queue_saved: Optional[deque] = None
        self.current_parent_idx_saved: Optional[int] = None
        self.used_f_per_parent_saved: Optional[Dict[int, Set[int]]] = None

        # Tree-mode runtime state.
        self.voxel_occupancy = np.zeros(
            (self.n_discrete_size, self.n_discrete_size, self.n_discrete_size),
            dtype=int,
        )
        self.bricks: List[Brick] = []
        self.bfs_queue: deque = deque()
        self.current_parent_idx: Optional[int] = None
        self.used_f_per_parent: Dict[int, Set[int]] = {}
        self.generation_snapshots: Dict[int, Dict] = {}
        self.rejected_bricks: Set[Tuple[int, int, int, int, int]] = set()
        self.generation_count = 0

        self.attempt = 0

        self.available_brick_shapes = self._load_available_brick_shapes()

        self.eval()

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _load_available_brick_shapes(self):
        library_path = Path(__file__).resolve().parent.parent / "brick_data" / "brick_library.json"
        with open(library_path, "r", encoding="utf-8") as f:
            brick_library = json.load(f)
        shape_set = set()
        for item in brick_library.values():
            h = int(item["height"])
            w = int(item["width"])
            shape_set.add((h, w))
            shape_set.add((w, h))
        # Restrict to shapes whose dims have a corresponding hw token.
        valid_dims = set(self.size_to_hw_token.keys())
        shape_set = {(h, w) for (h, w) in shape_set if h in valid_dims and w in valid_dims}
        return shape_set

    def process_point_feature(self, point_feature):
        encode_feature = torch.zeros(
            point_feature.shape[0], self.cond_length, self.config.word_embed_proj_dim,
            device=self.cond_head_proj.weight.device, dtype=self.cond_head_proj.weight.dtype,
        )
        encode_feature[:, 0] = self.cond_head_proj(point_feature[:, 0])
        shape_latents = self.point_encoder.to_shape_latents(point_feature[:, 1:])
        encode_feature[:, 1:] = self.cond_proj(torch.cat([point_feature[:, 1:], shape_latents], dim=-1))
        return encode_feature

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def init_state(self) -> None:
        """Reset runtime state. Keeps ``feature_cache_org`` (the encoder cache)."""
        self.kv_cache = DynamicCache()
        self.kv_cache_saved = None
        if self.feature_cache_org is not None:
            self.feature_cache = copy.deepcopy(self.feature_cache_org)
        else:
            self.feature_cache = None
        self.voxel_occupancy = np.zeros(
            (self.n_discrete_size, self.n_discrete_size, self.n_discrete_size),
            dtype=int,
        )
        self.bricks = []
        self.bfs_queue = deque()
        self.current_parent_idx = None
        self.used_f_per_parent = {}
        self.generation_snapshots = {}

    def reset_cache(self) -> None:
        self.kv_cache = DynamicCache()

    def save_state(self) -> None:
        self.kv_cache_saved = copy.deepcopy(self.kv_cache)
        self.generated_ids_saved = self.generated_ids.clone() if self.generated_ids is not None else None
        self.voxel_occupancy_saved = copy.deepcopy(self.voxel_occupancy)
        self.bricks_saved = copy.deepcopy(self.bricks)
        self.bfs_queue_saved = copy.deepcopy(self.bfs_queue)
        self.current_parent_idx_saved = self.current_parent_idx
        self.used_f_per_parent_saved = copy.deepcopy(self.used_f_per_parent)

    def rollback_to_saved_state(self) -> None:
        self.kv_cache = self.kv_cache_saved
        self.generated_ids = self.generated_ids_saved.clone() if self.generated_ids_saved is not None else None
        self.voxel_occupancy = copy.deepcopy(self.voxel_occupancy_saved) if self.voxel_occupancy_saved is not None else self.voxel_occupancy
        self.bricks = copy.deepcopy(self.bricks_saved) if self.bricks_saved is not None else []
        self.bfs_queue = copy.deepcopy(self.bfs_queue_saved) if self.bfs_queue_saved is not None else deque()
        self.current_parent_idx = self.current_parent_idx_saved
        self.used_f_per_parent = copy.deepcopy(self.used_f_per_parent_saved) if self.used_f_per_parent_saved is not None else {}
        if not (self.kv_cache is not None and len(self.kv_cache) > 0):
            if self.feature_cache_org is not None:
                self.feature_cache = copy.deepcopy(self.feature_cache_org)

    def _store_generation_snapshot(self, next_brick_idx: int) -> None:
        """Save lightweight state before generating ``next_brick_idx``."""
        self.generation_snapshots[next_brick_idx] = {
            "generated_ids": self.generated_ids.clone() if self.generated_ids is not None else None,
            "voxel_occupancy": copy.deepcopy(self.voxel_occupancy),
            "bricks": copy.deepcopy(self.bricks),
            "bfs_queue": copy.deepcopy(self.bfs_queue),
            "current_parent_idx": self.current_parent_idx,
            "used_f_per_parent": copy.deepcopy(self.used_f_per_parent),
        }

    def _restore_generation_snapshot(self, next_brick_idx: int) -> bool:
        snapshot = self.generation_snapshots.get(next_brick_idx)
        if snapshot is None:
            logger.warning("Missing generation snapshot for brick idx %s.", next_brick_idx)
            return False

        self.generated_ids = (
            snapshot["generated_ids"].clone()
            if snapshot["generated_ids"] is not None
            else None
        )
        self.voxel_occupancy = copy.deepcopy(snapshot["voxel_occupancy"])
        self.bricks = copy.deepcopy(snapshot["bricks"])
        self.bfs_queue = copy.deepcopy(snapshot["bfs_queue"])
        self.current_parent_idx = snapshot["current_parent_idx"]
        self.used_f_per_parent = copy.deepcopy(snapshot["used_f_per_parent"])
        self.generation_snapshots = {
            idx: state
            for idx, state in self.generation_snapshots.items()
            if idx <= next_brick_idx
        }

        # Reuse the conditioning cache and let the next sampling call replay the
        # retained prefix into a fresh autoregressive cache.
        self.kv_cache = DynamicCache()
        self.feature_cache = copy.deepcopy(self.feature_cache_org)
        return True

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def _decode_f(self, parent: Brick, f_raw: int) -> Tuple[int, int, int, int]:
        """Decode ``f_raw`` to ``(s_f, u_p, v_p, child_z)``.

        Inverse of ``compute_f_m`` in ``data_process/convert_new_seq.py``.
        """
        s_f, rem = divmod(f_raw, parent.h * parent.w)
        v_p, u_p = divmod(rem, parent.h)
        cz = parent.z + 1 if s_f == 0 else parent.z - 1
        return s_f, u_p, v_p, cz

    def _decode_child_pos(
        self,
        parent: Brick,
        f_raw: int,
        h: int,
        w: int,
        m_raw: int,
        token_start: int,
    ) -> Brick:
        s_f, u_p, v_p, cz = self._decode_f(parent, f_raw)
        v_c, u_c = divmod(m_raw, h)
        xo = parent.x + u_p
        yo = parent.y + v_p
        cx = xo - u_c
        cy = yo - v_c
        return Brick(
            idx=len(self.bricks),
            x=cx,
            y=cy,
            z=cz,
            h=h,
            w=w,
            parent_idx=parent.idx,
            token_span=(token_start, token_start + 4),
        )

    def _is_in_grid(self, cx: int, cy: int, cz: int, h: int, w: int) -> bool:
        N = self.n_discrete_size
        if cx < 0 or cx + h > N:
            return False
        if cy < 0 or cy + w > N:
            return False
        if cz < 0 or cz >= N:
            return False
        return True

    def _is_valid_placement(self, cx: int, cy: int, cz: int, h: int, w: int) -> bool:
        if not self._is_in_grid(cx, cy, cz, h, w):
            return False
        if self.voxel_occupancy[cx:cx + h, cy:cy + w, cz].any():
            return False
        return True

    def _is_valid_m(self, parent: Brick, f_raw: int, h: int, w: int, m_raw: int) -> bool:
        if m_raw < 0 or m_raw >= h * w:
            return False
        _, u_p, v_p, cz = self._decode_f(parent, f_raw)
        v_c, u_c = divmod(m_raw, h)
        cx = parent.x + u_p - u_c
        cy = parent.y + v_p - v_c
        return self._is_valid_placement(cx, cy, cz, h, w)

    def _exists_valid_m(self, parent: Brick, f_raw: int, h: int, w: int) -> bool:
        for m_raw in range(h * w):
            if self._is_valid_m(parent, f_raw, h, w, m_raw):
                return True
        return False

    def _exists_valid_hw_for_f(self, parent: Brick, f_raw: int) -> bool:
        for (h, w) in self.available_brick_shapes:
            if self._exists_valid_m(parent, f_raw, h, w):
                return True
        return False

    def _brick_key(self, brick: Brick) -> Tuple[int, int, int, int, int]:
        return brick.x, brick.y, brick.z, brick.h, brick.w

    def _root_key(self, x: int, y: int, z: int, h: int, w: int) -> Tuple[int, int, int, int, int]:
        return x, y, z, h, w

    def _child_key(self, parent: Brick, f_raw: int, h: int, w: int, m_raw: int) -> Tuple[int, int, int, int, int]:
        _, u_p, v_p, cz = self._decode_f(parent, f_raw)
        v_c, u_c = divmod(m_raw, h)
        cx = parent.x + u_p - u_c
        cy = parent.y + v_p - v_c
        return cx, cy, cz, h, w

    def _is_rejected_key(self, key: Tuple[int, int, int, int, int]) -> bool:
        return key in self.rejected_bricks

    # ------------------------------------------------------------------
    # Allowed-token sets per generation step
    # ------------------------------------------------------------------

    def _allowed_xyz(self) -> List[int]:
        """Loose mask: any in-grid coordinate (h/w not yet known)."""
        return [self.xyz_offset + i for i in range(self.n_discrete_size)]

    def _allowed_root_h(self, root_x: int, root_y: int) -> List[int]:
        N = self.n_discrete_size
        allowed: List[int] = []
        for h_size, h_token in self.size_to_hw_token.items():
            if root_x + h_size > N:
                continue
            has_w = False
            for w_size in self.size_to_hw_token.keys():
                if (h_size, w_size) not in self.available_brick_shapes:
                    continue
                if root_y + w_size > N:
                    continue
                has_w = True
                break
            if has_w:
                allowed.append(h_token)
        return allowed

    def _allowed_root_w(self, root_x: int, root_y: int, root_z: int, h_size: int) -> List[int]:
        N = self.n_discrete_size
        allowed: List[int] = []
        for w_size, w_token in self.size_to_hw_token.items():
            if (h_size, w_size) not in self.available_brick_shapes:
                continue
            if root_y + w_size > N:
                continue
            if self._is_rejected_key(self._root_key(root_x, root_y, root_z, h_size, w_size)):
                continue
            allowed.append(w_token)
        return allowed

    def _allowed_child_f_or_end(self, parent: Brick) -> List[int]:
        N = self.n_discrete_size
        used_f = self.used_f_per_parent.get(parent.idx, set())
        allowed: List[int] = []
        max_f = 2 * parent.h * parent.w
        for f_raw in range(max_f):
            if f_raw in used_f:
                continue
            _, u_p, v_p, cz = self._decode_f(parent, f_raw)
            xo = parent.x + u_p
            yo = parent.y + v_p
            if not (0 <= xo < N and 0 <= yo < N and 0 <= cz < N):
                continue
            # The connection cell on the child side is (xo, yo, cz). If already
            # occupied, no child can use this f.
            if self.voxel_occupancy[xo, yo, cz] != 0:
                continue
            # Thorough check: at least one (h, w, m) must give a fully-valid
            # child placement, otherwise downstream sampling would dead-end.
            if not self._exists_valid_hw_for_f(parent, f_raw):
                continue
            allowed.append(self.f_offset + f_raw)
        # Termination tokens. We let the model pick between EOP and EOS based
        # on its training distribution: in training, trailing EOPs get stripped
        # and EOS lands right after the last m, so the model has been taught
        # exactly when to emit which token. The outer loop turns "EOP with an
        # empty queue" into a terminating event as a safety net.
        allowed.append(self.eop_token_id)
        allowed.append(self.eos_token_id)
        return allowed

    def _allowed_child_h(self, parent: Brick, f_raw: int) -> List[int]:
        allowed: List[int] = []
        for h_size, h_token in self.size_to_hw_token.items():
            found = False
            for w_size in self.size_to_hw_token.keys():
                if (h_size, w_size) not in self.available_brick_shapes:
                    continue
                if self._exists_valid_m(parent, f_raw, h_size, w_size):
                    found = True
                    break
            if found:
                allowed.append(h_token)
        return allowed

    def _allowed_child_w(self, parent: Brick, f_raw: int, h_size: int) -> List[int]:
        allowed: List[int] = []
        for w_size, w_token in self.size_to_hw_token.items():
            if (h_size, w_size) not in self.available_brick_shapes:
                continue
            if self._exists_valid_m(parent, f_raw, h_size, w_size):
                allowed.append(w_token)
        return allowed

    def _allowed_child_m(self, parent: Brick, f_raw: int, h: int, w: int) -> List[int]:
        allowed: List[int] = []
        for m_raw in range(h * w):
            if self._is_valid_m(parent, f_raw, h, w, m_raw):
                if self._is_rejected_key(self._child_key(parent, f_raw, h, w, m_raw)):
                    continue
                allowed.append(self.m_offset + m_raw)
        return allowed

    # ------------------------------------------------------------------
    # Sampling primitive
    # ------------------------------------------------------------------

    def _sample_with_mask(self, allowed_token_ids: List[int], temperature: float) -> Optional[torch.Tensor]:
        """Sample one token, restricting the support to ``allowed_token_ids``.

        Returns the new 1-element token tensor (already appended to
        ``self.generated_ids``), or ``None`` if no token is allowed.
        """
        if not allowed_token_ids:
            logger.warning("Empty allowed-token set; aborting sampling step.")
            return None
        proc = AllowedTokensProcessor(allowed_token_ids)
        dummy_pc_input = torch.zeros(
            1, self.cond_length,
            device=self.generated_ids.device, dtype=self.generated_ids.dtype,
        )
        ids = torch.cat([dummy_pc_input, self.generated_ids.unsqueeze(0)], dim=1)
        cond_length = ids.shape[1]
        output = self.transformer.generate(
            input_ids=ids,
            use_cache=True,
            do_sample=True,
            top_k=self.top_k,
            top_p=self.top_p,
            temperature=temperature,
            max_new_tokens=1,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            past_key_values=(
                self.kv_cache
                if (self.kv_cache is not None and len(self.kv_cache) > 0)
                else self.feature_cache
            ),
            logits_processor=LogitsProcessorList([proc]),
            return_dict_in_generate=True,
        )
        self.kv_cache = output["past_key_values"]
        new_token = output["sequences"][0][cond_length:]
        self.generated_ids = torch.cat([self.generated_ids, new_token], dim=0)
        return new_token

    # ------------------------------------------------------------------
    # Brick commit
    # ------------------------------------------------------------------

    def _commit_brick(self, brick: Brick) -> None:
        self.bricks.append(brick)
        self.voxel_occupancy[brick.x:brick.x + brick.h, brick.y:brick.y + brick.w, brick.z] = 1
        self.bfs_queue.append(brick.idx)

    # ------------------------------------------------------------------
    # Top-level generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward(self, pc_normal):
        # Encode point cloud and seed the kv cache with the BOS token.
        point_feature = self.point_encoder.encode_latents(pc_normal)
        processed_point_feature = self.process_point_feature(point_feature)

        results = self.transformer.generate(
            inputs_embeds=processed_point_feature,
            max_new_tokens=1,
            num_beams=1,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            use_cache=True,
            return_dict_in_generate=True,
        )
        self.kv_cache = results.past_key_values
        self.feature_cache_org = copy.deepcopy(results.past_key_values)
        self.feature_cache = copy.deepcopy(results.past_key_values)
        self.generated_ids = results.sequences[0]

        self.init_state()
        self.rejected_bricks = set()
        self.generation_count = 0
        self._store_generation_snapshot(0)

        temperature = float(self.temperature)
        for rollback_idx in range(self.max_rollback_num + 1):
            self.generation_count = rollback_idx
            if rollback_idx > 0:
                temperature = float(min(1.1, temperature + 0.05))

            if not self._generate_until_done(temperature):
                logger.warning("Failed to generate a valid tree; aborting.")
                break

            stable_flag, bad_idx = self._find_unstable_brick_idx()
            logger.info("tree stability: %s, bad_idx: %s", stable_flag, bad_idx)
            if stable_flag or bad_idx is None:
                break

            bad_brick = self.bricks[bad_idx]
            self.rejected_bricks.add(self._brick_key(bad_brick))
            if rollback_idx >= self.max_rollback_num:
                logger.warning("Stability rollback budget exhausted; returning latest tree.")
                break

            rollback_target_idx = bad_brick.parent_idx if bad_brick.parent_idx is not None else 0
            logger.info(
                "rollback unstable brick idx=%s key=%s to brick idx=%s",
                bad_idx,
                self._brick_key(bad_brick),
                rollback_target_idx,
            )
            if not self._restore_generation_snapshot(rollback_target_idx):
                break

        return self._build_output_dict()

    def _generate_until_done(self, temperature: float) -> bool:
        if len(self.bricks) == 0:
            if not self._generate_root_brick(temperature):
                logger.warning("Failed to generate a valid root brick.")
                return False

        if self.current_parent_idx is not None and self.current_parent_idx < len(self.bricks):
            status = self._expand_parent(temperature)
            self.current_parent_idx = None
            if status == "eos":
                return True

        while len(self.bfs_queue) > 0 and len(self.bricks) < self.max_bricks:
            self.current_parent_idx = self.bfs_queue.popleft()
            self.used_f_per_parent[self.current_parent_idx] = set()
            status = self._expand_parent(temperature)
            self.current_parent_idx = None
            if status == "eos":
                break
        return True

    def _find_unstable_brick_idx(self) -> Tuple[bool, Optional[int]]:
        if not self.bricks:
            return False, None

        gen_brick = [[b.h, b.w, b.x, b.y, b.z] for b in self.bricks]
        try:
            brick_structure = BrickStructure.from_list([gen_brick])
            stable_flag, stable_scores = brick_structure.is_stable()
        except Exception:
            logger.exception("Failed to compute stability.")
            return False, 0

        if stable_flag:
            return True, None
        if stable_scores is None or stable_scores.size == 0:
            return False, 0

        for idx, brick in enumerate(self.bricks):
            brick_scores = stable_scores[
                brick.x:brick.x + brick.h,
                brick.y:brick.y + brick.w,
                brick.z,
            ]
            if np.any(brick_scores >= 1.0):
                return False, idx
        return False, 0

    def _generate_root_brick(self, temperature: float) -> bool:
        token_start = int(self.generated_ids.numel())
        self._store_generation_snapshot(0)

        # x, y, z. The structural mask only enforces in-grid coordinates;
        # h/w bounds are checked in subsequent steps.
        for _ in range(3):
            tok = self._sample_with_mask(self._allowed_xyz(), temperature)
            if tok is None:
                self._restore_generation_snapshot(0)
                return False

        root_x = int(self.generated_ids[-3].item()) - self.xyz_offset
        root_y = int(self.generated_ids[-2].item()) - self.xyz_offset
        root_z = int(self.generated_ids[-1].item()) - self.xyz_offset

        # h.
        allowed_h = self._allowed_root_h(root_x, root_y)
        h_tok = self._sample_with_mask(allowed_h, temperature)
        if h_tok is None:
            self._restore_generation_snapshot(0)
            return False
        h_size = self.hw_token_to_size[int(h_tok.item())]

        # w.
        allowed_w = self._allowed_root_w(root_x, root_y, root_z, h_size)
        w_tok = self._sample_with_mask(allowed_w, temperature)
        if w_tok is None:
            self._restore_generation_snapshot(0)
            return False
        w_size = self.hw_token_to_size[int(w_tok.item())]

        root = Brick(
            idx=0,
            x=root_x,
            y=root_y,
            z=root_z,
            h=h_size,
            w=w_size,
            parent_idx=None,
            token_span=(token_start, token_start + 5),
        )
        self._commit_brick(root)
        return True

    def _expand_parent(self, temperature: float) -> str:
        """Expand ``self.current_parent_idx``.

        Returns one of ``"eos"`` / ``"eop"`` / ``"max"`` to signal what
        terminated this parent's expansion.
        """
        parent = self.bricks[self.current_parent_idx]
        while len(self.bricks) < self.max_bricks:
            token_start = int(self.generated_ids.numel())
            self._store_generation_snapshot(len(self.bricks))

            # Step f / EOP / EOS.
            allowed_f = self._allowed_child_f_or_end(parent)
            tok = self._sample_with_mask(allowed_f, temperature)
            if tok is None:
                self._restore_generation_snapshot(len(self.bricks))
                return "eos"
            tok_id = int(tok.item())
            if tok_id == self.eos_token_id:
                return "eos"
            if tok_id == self.eop_token_id:
                # If the BFS queue is exhausted, an EOP would be a trailing
                # token that the training pipeline strips. Treat it as EOS.
                if len(self.bfs_queue) == 0:
                    return "eos"
                return "eop"
            f_raw = tok_id - self.f_offset

            # Step h.
            allowed_h = self._allowed_child_h(parent, f_raw)
            h_tok = self._sample_with_mask(allowed_h, temperature)
            if h_tok is None:
                self._restore_generation_snapshot(len(self.bricks))
                return "eos"
            h_size = self.hw_token_to_size[int(h_tok.item())]

            # Step w.
            allowed_w = self._allowed_child_w(parent, f_raw, h_size)
            w_tok = self._sample_with_mask(allowed_w, temperature)
            if w_tok is None:
                self._restore_generation_snapshot(len(self.bricks))
                return "eos"
            w_size = self.hw_token_to_size[int(w_tok.item())]

            # Step m.
            allowed_m = self._allowed_child_m(parent, f_raw, h_size, w_size)
            m_tok = self._sample_with_mask(allowed_m, temperature)
            if m_tok is None:
                self._restore_generation_snapshot(len(self.bricks))
                return "eos"
            m_raw = int(m_tok.item()) - self.m_offset

            # Commit child.
            child = self._decode_child_pos(parent, f_raw, h_size, w_size, m_raw, token_start)
            self._commit_brick(child)
            self.used_f_per_parent[self.current_parent_idx].add(f_raw)

        return "max"

    # ------------------------------------------------------------------
    # Output formatting
    # ------------------------------------------------------------------

    def _build_output_dict(self) -> Dict:
        """Return a dict compatible with the v3 inference pipeline.

        ``gen_brick`` uses the ``[h, w, x, y, z]`` layout expected by
        ``brick2ldr``. ``comp_id`` is always 1 because the BFS construction
        produces a single connected component by definition.
        """
        gen_brick = [[b.h, b.w, b.x, b.y, b.z] for b in self.bricks]
        seq_batch = (
            self.generated_ids.unsqueeze(0)
            if self.generated_ids is not None
            else torch.empty((1, 0), dtype=torch.long)
        )
        return {
            "gen_brick": [gen_brick],
            "seq_batch": seq_batch,
            "comp_id": 1,
            "regeneration_count": self.generation_count,
        }
