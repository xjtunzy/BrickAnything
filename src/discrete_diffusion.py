import math
from dataclasses import dataclass
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BrickSpace:
    # Your discrete spaces
    type_classes: int = 14          # 0..13
    coord_classes: int = 20         # 0..19
    # absorbing (mask) ids appended as the last category
    type_mask_id: int = 14          # => total 15
    coord_mask_id: int = 20         # => total 21

    @property
    def type_vocab(self) -> int:
        return self.type_classes + 1

    @property
    def coord_vocab(self) -> int:
        return self.coord_classes + 1


class AbsorbingSchedule:
    """
    Defines gamma_t (masking probability at step t).
    We'll use a cosine schedule on the *cumulative keep probability* alpha_bar(t),
    then derive per-step gamma_t so that alpha_bar_t = prod_s (1 - gamma_s).

    This gives smooth corruption similar to continuous cosine schedules.
    """
    def __init__(self, T: int, s: float = 0.008, device: str = "cpu"):
        self.T = T
        self.s = s
        self.device = device
        self._precompute()

    def _alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        # t in [0..T], continuous-ish
        # cosine schedule used in DDPM: alpha_bar(t) = cos^2((t/T + s)/(1+s) * pi/2)
        T = self.T
        s = self.s
        f = (t / T + s) / (1 + s)
        return torch.cos(f * math.pi / 2) ** 2

    def _precompute(self):
        # compute alpha_bar at integer steps 0..T
        t = torch.arange(0, self.T + 1, device=self.device, dtype=torch.float32)
        alpha_bar = self._alpha_bar(t)
        # normalize so alpha_bar[0]=1
        alpha_bar = alpha_bar / alpha_bar[0].clamp_min(1e-12)

        # per-step keep: alpha_t = alpha_bar[t] / alpha_bar[t-1]
        alpha_t = alpha_bar[1:] / alpha_bar[:-1].clamp_min(1e-12)
        alpha_t = alpha_t.clamp(0.0, 1.0)

        # per-step mask prob gamma_t = 1 - alpha_t
        gamma_t = (1.0 - alpha_t).clamp(0.0, 1.0)

        self.alpha_bar = alpha_bar          # (T+1,)
        self.alpha_t = alpha_t              # (T,)
        self.gamma_t = gamma_t              # (T,)

    def sample_t(self, batch_size: int) -> torch.Tensor:
        # sample integer t in [1..T]
        return torch.randint(1, self.T + 1, (batch_size,), device=self.device)

    def get_alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) int in [0..T]
        return self.alpha_bar[t].to(t.device)

    def get_gamma(self, t: torch.Tensor) -> torch.Tensor:
        # per-step gamma_t for step t (1..T) stored at index t-1
        return self.gamma_t[(t - 1).clamp_min(0)].to(t.device)


class AbsorbingD3PM(nn.Module):
    """
    Absorbing-state forward corruption for brick tokens.

    Representation:
      x0_type: (B, L) int in [0..13] or pad id handled by pad_mask
      x0_xyz : (B, L, 3) each int in [0..19]
    """
    def __init__(self, T: int, space: BrickSpace, token_absorb: bool = True, device: str = "cpu"):
        super().__init__()
        self.T = T
        self.space = space
        self.token_absorb = token_absorb
        self.schedule = AbsorbingSchedule(T=T, device=device)

    @torch.no_grad()
    def q_sample(self,
                 x0_type: torch.Tensor,
                 x0_xyz: torch.Tensor,
                 t: torch.Tensor,
                 pad_mask: Optional[torch.Tensor] = None,
                 ) -> Dict[str, torch.Tensor]:
        """
        Sample x_t ~ q(x_t | x0) for absorbing diffusion.

        If token_absorb=True:
          with prob (1-alpha_bar_t) the whole token becomes [MASK] (all fields masked).
        Else:
          each field independently becomes [MASK] with prob (1-alpha_bar_t).

        pad_mask: (B, L) True for PAD positions (ignored, left unchanged).
        """
        B, L = x0_type.shape
        device = x0_type.device
        assert x0_xyz.shape == (B, L, 3)

        alpha_bar = self.schedule.get_alpha_bar(t).view(B, 1)  # (B,1)
        keep_prob = alpha_bar  # P(not masked by time t)
        mask_prob = (1.0 - keep_prob).clamp(0.0, 1.0)

        if pad_mask is None:
            pad_mask = torch.zeros((B, L), device=device, dtype=torch.bool)

        # sample mask indicators
        if self.token_absorb:
            m = torch.rand((B, L), device=device) < mask_prob  # True => masked token
            m = m & (~pad_mask)
            xt_type = x0_type.clone()
            xt_xyz = x0_xyz.clone()
            xt_type[m] = self.space.type_mask_id
            xt_xyz[m] = self.space.coord_mask_id
            xt_xyz[m] = torch.full((m.sum(), 3), self.space.coord_mask_id, device=device, dtype=xt_xyz.dtype)
        else:
            m_type = (torch.rand((B, L), device=device) < mask_prob) & (~pad_mask)
            m_xyz = (torch.rand((B, L, 3), device=device) < mask_prob.view(B, 1, 1)) & (~pad_mask.unsqueeze(-1))
            xt_type = x0_type.clone()
            xt_xyz = x0_xyz.clone()
            xt_type[m_type] = self.space.type_mask_id
            xt_xyz[m_xyz] = self.space.coord_mask_id

        return {
            "t": t,
            "xt_type": xt_type,
            "xt_xyz": xt_xyz,
            "pad_mask": pad_mask,
        }

    def x0_ce_loss(self,
                   pred_type_logits: torch.Tensor,
                   pred_xyz_logits: torch.Tensor,
                   x0_type: torch.Tensor,
                   x0_xyz: torch.Tensor,
                   xt_type: torch.Tensor,
                   xt_xyz: torch.Tensor,
                   pad_mask: Optional[torch.Tensor] = None,
                   only_masked: bool = True,
                   ) -> torch.Tensor:
        """
        Auxiliary denoising objective: -log p_theta(x0 | xt).

        pred_type_logits: (B, L, 14)  (predict only original 14 classes, exclude mask)
        pred_xyz_logits : (B, L, 3, 20)
        x0_type: (B, L) in [0..13]
        x0_xyz : (B, L, 3) in [0..19]
        xt_* include mask ids.

        only_masked=True: compute loss only where xt was masked (BERT-style).
        """
        B, L = x0_type.shape
        device = x0_type.device
        if pad_mask is None:
            pad_mask = torch.zeros((B, L), device=device, dtype=torch.bool)

        # masked positions (token-wise or field-wise)
        if self.token_absorb:
            masked = (xt_type == self.space.type_mask_id) & (~pad_mask)  # (B,L)
        else:
            masked = ((xt_type == self.space.type_mask_id) | (xt_xyz[..., 0] == self.space.coord_mask_id)) & (~pad_mask)

        if not only_masked:
            masked = ~pad_mask

        # type CE
        # logits are over 14 classes; targets must be 0..13
        type_loss = F.cross_entropy(
            pred_type_logits.reshape(-1, self.space.type_classes),
            x0_type.reshape(-1),
            reduction="none",
        ).view(B, L)

        # xyz CE (3 fields)
        xyz_loss = F.cross_entropy(
            pred_xyz_logits.reshape(-1, self.space.coord_classes),
            x0_xyz.reshape(-1),
            reduction="none",
        ).view(B, L, 3).mean(dim=-1)  # average x,y,z

        loss = type_loss + xyz_loss
        loss = loss[masked].mean() if masked.any() else loss.mean()
        return loss


# -------------------------- Example wiring --------------------------
# Your denoiser network should map (xt_type, xt_xyz, t) -> logits for x0
# Below is a tiny placeholder; replace with Transformer/UNet-like model.

class TinyDenoiser(nn.Module):
    def __init__(self, space: BrickSpace, d_model: int = 256, max_len: int = 1000, T: int = 1000):
        super().__init__()
        self.space = space
        self.type_emb = nn.Embedding(space.type_vocab, d_model)   # includes mask id
        self.coord_emb = nn.Embedding(space.coord_vocab, d_model) # includes mask id
        self.t_emb = nn.Embedding(T + 1, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)

        self.block = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=8, batch_first=True),
            num_layers=4
        )

        self.type_head = nn.Linear(d_model, space.type_classes)   # predict 14 original classes
        self.xyz_head = nn.Linear(d_model, 3 * space.coord_classes)

    def forward(self, xt_type: torch.Tensor, xt_xyz: torch.Tensor, t: torch.Tensor, pad_mask: Optional[torch.Tensor] = None):
        B, L = xt_type.shape
        device = xt_type.device
        if pad_mask is None:
            pad_mask = torch.zeros((B, L), device=device, dtype=torch.bool)

        pos = torch.arange(L, device=device).unsqueeze(0).expand(B, L)

        h = (
            self.type_emb(xt_type)
            + self.coord_emb(xt_xyz[..., 0])
            + self.coord_emb(xt_xyz[..., 1])
            + self.coord_emb(xt_xyz[..., 2])
            + self.t_emb(t).unsqueeze(1)
            + self.pos_emb(pos)
        )
        h = self.block(h, src_key_padding_mask=pad_mask)

        type_logits = self.type_head(h)  # (B,L,14)
        xyz_logits = self.xyz_head(h).view(B, L, 3, self.space.coord_classes)  # (B,L,3,20)
        return type_logits, xyz_logits


# -------------------------- Training step sketch --------------------------
def training_step(model: nn.Module,
                  diffusion: AbsorbingD3PM,
                  batch: Dict[str, torch.Tensor],
                  lambda_aux: float = 1.0,
                  only_masked: bool = True):
    """
    batch should contain:
      x0_type: (B,L) in [0..13]
      x0_xyz : (B,L,3) in [0..19]
      pad_mask: (B,L) bool  (True for PAD)
    """
    x0_type = batch["x0_type"]
    x0_xyz = batch["x0_xyz"]
    pad_mask = batch.get("pad_mask", None)

    B = x0_type.shape[0]
    t = diffusion.schedule.sample_t(B).to(x0_type.device)

    noised = diffusion.q_sample(x0_type, x0_xyz, t, pad_mask=pad_mask)
    xt_type, xt_xyz = noised["xt_type"], noised["xt_xyz"]

    pred_type_logits, pred_xyz_logits = model(xt_type, xt_xyz, t, pad_mask=pad_mask)

    # auxiliary x0 prediction loss (the term in D3PM paper Eq.(5) right part)
    aux = diffusion.x0_ce_loss(
        pred_type_logits, pred_xyz_logits,
        x0_type, x0_xyz,
        xt_type, xt_xyz,
        pad_mask=pad_mask,
        only_masked=only_masked
    )

    # If you want *pure* absorbing diffusion training, aux alone is a strong baseline.
    # Full L_vb (ELBO) is more involved; many implementations use aux (x0 loss) as main objective.
    loss = lambda_aux * aux
    return loss
