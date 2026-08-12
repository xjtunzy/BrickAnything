import datetime
import math
import os
import time
from collections import defaultdict

import torch
import torch.nn.functional as F

from brickanything_train.misc import SmoothedValue


def save_checkpoint(checkpoint_dir, model, optimizer, epoch, args, filename=None):
    checkpoint_name = os.path.join(checkpoint_dir, filename)
    try:
        weight_ckpt = model.module.state_dict()
    except Exception:
        weight_ckpt = model.state_dict()

    sd = {
        "model": weight_ckpt,
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "args": args,
    }
    torch.save(sd, checkpoint_name)


def compute_learning_rate(args, curr_epoch_normalized):
    assert 0.0 <= curr_epoch_normalized <= 1.0
    if (
        curr_epoch_normalized <= (args.warm_lr_epochs / args.max_epoch)
        and args.warm_lr_epochs > 0
    ):
        curr_lr = args.warm_lr + curr_epoch_normalized * args.max_epoch * (
            (args.base_lr - args.warm_lr) / args.warm_lr_epochs
        )
    else:
        curr_lr = args.final_lr + 0.5 * (args.base_lr - args.final_lr) * (
            1 + math.cos(math.pi * curr_epoch_normalized)
        )
    return curr_lr


def adjust_learning_rate(args, optimizer, curr_epoch):
    curr_lr = compute_learning_rate(args, curr_epoch)
    for param_group in optimizer.param_groups:
        param_group["lr"] = curr_lr
    return curr_lr


def _reduce_mean(accelerator, x: torch.Tensor):
    local_mean = x.detach().float().mean()
    if accelerator.num_processes <= 1:
        return float(local_mean.item())
    if hasattr(accelerator, "reduce"):
        reduced = accelerator.reduce(local_mean.clone(), reduction="mean")
        return float(reduced.item())
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(local_mean, op=dist.ReduceOp.SUM)
        local_mean = local_mean / dist.get_world_size()
    return float(local_mean.item())


def _reduce_sum_triplet(accelerator, sum_loss: float, sum_acc_w: float, count: float) -> tuple:
    """All-reduce (sum) three scalars across processes for weighted val metrics."""
    t = torch.tensor(
        [sum_loss, sum_acc_w, count], device=accelerator.device, dtype=torch.float64
    )
    if accelerator.num_processes <= 1:
        return float(t[0].item()), float(t[1].item()), float(t[2].item())
    if hasattr(accelerator, "reduce"):
        reduced = accelerator.reduce(t.clone(), reduction="sum")
        return float(reduced[0].item()), float(reduced[1].item()), float(reduced[2].item())
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t[0].item()), float(t[1].item()), float(t[2].item())


def _reduce_sum_vector(accelerator, values):
    """All-reduce (sum) for a variable-length vector of scalars."""
    t = torch.tensor(values, device=accelerator.device, dtype=torch.float64)
    if accelerator.num_processes <= 1:
        return [float(x.item()) for x in t]
    if hasattr(accelerator, "reduce"):
        reduced = accelerator.reduce(t.clone(), reduction="sum")
        return [float(x.item()) for x in reduced]
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return [float(x.item()) for x in t]


# ---------------------------------------------------------------------------
# DPO loss — mirrors the reference DPO repo (trainers.py::preference_loss)
# ---------------------------------------------------------------------------
def preference_loss(
    policy_chosen_logps: torch.FloatTensor,
    policy_rejected_logps: torch.FloatTensor,
    reference_chosen_logps: torch.FloatTensor,
    reference_rejected_logps: torch.FloatTensor,
    beta: float,
    label_smoothing: float = 0.0,
    reference_free: bool = False,
):
    """Compute the DPO loss (Eq.7 of https://arxiv.org/pdf/2305.18290.pdf).

    Returns (losses, chosen_rewards, rejected_rewards) — all shape (batch_size,).
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    if reference_free:
        ref_logratios = 0

    logits = pi_logratios - ref_logratios

    losses = (
        -F.logsigmoid(beta * logits) * (1 - label_smoothing)
        - F.logsigmoid(-beta * logits) * label_smoothing
    )

    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

    return losses, chosen_rewards, rejected_rewards


# ---------------------------------------------------------------------------
# Forward helper — concatenated chosen+rejected in a single forward pass
# (same idea as the reference repo's concatenated_forward)
# ---------------------------------------------------------------------------
def _concatenated_forward(model, pc_normal, chosen_seq, rejected_seq):
    """Run model on chosen & rejected sequences in one forward pass.

    pc_normal is duplicated so that the batch dimension matches the
    concatenated sequences (chosen first, then rejected).

    Returns (chosen_logps, rejected_logps) — each shape (batch_size,).
    """
    model_for_call = getattr(model, "module", model)

    cat_pc = torch.cat([pc_normal, pc_normal], dim=0)
    cat_seq = torch.cat([chosen_seq, rejected_seq], dim=0)

    all_logps, _, _ = model_for_call.sequence_logprobs(cat_pc, cat_seq)
    bsz = pc_normal.shape[0]
    chosen_logps = all_logps[:bsz]
    rejected_logps = all_logps[bsz:]
    return chosen_logps, rejected_logps


def _maybe_gt_sft_loss(model, batch, pc_normal, dpo_loss_scalar, gt_sft_lambda: float):
    """SFT on ground-truth tokens when lambda>0 and batch provides `gt_sequence`."""
    if gt_sft_lambda > 0 and "gt_sequence" in batch:
        return _gt_sft_loss(model, pc_normal, batch["gt_sequence"])
    return dpo_loss_scalar.new_zeros(())


def _gt_sft_loss(model, pc_normal, gt_seq):
    """Compute mean token-level NLL on gt sequence (masked by valid tokens)."""
    model_for_call = getattr(model, "module", model)
    _, token_logps, masks = model_for_call.sequence_logprobs(pc_normal, gt_seq)
    valid_tokens = masks.sum(dim=1)
    per_sample_nll = -token_logps.sum(dim=1) / torch.clamp(valid_tokens, min=1.0)
    valid_rows = valid_tokens > 0
    if bool(valid_rows.any().item()):
        return per_sample_nll[valid_rows].mean()
    return per_sample_nll.new_zeros(())


def _get_iou_gap_weights(batch, ref_tensor: torch.Tensor) -> torch.Tensor:
    """Fetch per-sample IoU-gap weights from batch; fallback to ones."""
    gaps = batch.get("iou_gap", None)
    if gaps is None:
        return torch.ones_like(ref_tensor)
    if not torch.is_tensor(gaps):
        gaps = torch.tensor(gaps, device=ref_tensor.device)
    gaps = gaps.to(device=ref_tensor.device, dtype=ref_tensor.dtype)
    # Safety: negative gap should not invert preference optimization direction.
    return torch.clamp(gaps, min=0.0)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
@torch.no_grad()
def _evaluate_dpo_on_loader(
    policy_model, ref_model, val_dataloader, args, accelerator
) -> dict:
    policy_model.eval()
    sum_dpo_loss = 0.0
    sum_gt_sft_loss = 0.0
    sum_total_loss = 0.0
    sum_reward_acc = 0.0
    n_samples = 0

    beta = args.dpo_beta
    gt_sft_lambda = float(getattr(args, "gt_sft_lambda", 0.0) or 0.0)
    label_smoothing = getattr(args, "label_smoothing", 0.0)
    reference_free = getattr(args, "reference_free", False)

    for batch in val_dataloader:
        with accelerator.autocast():
            pi_c, pi_r = _concatenated_forward(
                policy_model, batch["pc_normal"],
                batch["chosen_sequence"], batch["rejected_sequence"],
            )
            ref_c, ref_r = _concatenated_forward(
                ref_model, batch["pc_normal"],
                batch["chosen_sequence"], batch["rejected_sequence"],
            )

            losses, chosen_rewards, rejected_rewards = preference_loss(
                pi_c, pi_r, ref_c, ref_r,
                beta=beta,
                label_smoothing=label_smoothing,
                reference_free=reference_free,
            )
            iou_gap_weights = _get_iou_gap_weights(batch, losses)
            dpo_loss = (losses * iou_gap_weights).mean()
            gt_sft_loss = _maybe_gt_sft_loss(
                policy_model, batch, batch["pc_normal"], dpo_loss, gt_sft_lambda
            )
            total_loss = dpo_loss + gt_sft_lambda * gt_sft_loss
            reward_acc = (chosen_rewards > rejected_rewards).float()

        bsz = int(batch["pc_normal"].shape[0])
        sum_dpo_loss += float(dpo_loss.detach().item()) * bsz
        sum_gt_sft_loss += float(gt_sft_loss.detach().item()) * bsz
        sum_total_loss += float(total_loss.detach().item()) * bsz
        sum_reward_acc += float(reward_acc.mean().detach().item()) * bsz
        n_samples += bsz

    sum_dpo_loss, sum_reward_acc, sum_gt_sft_loss, sum_total_loss, n_samples = _reduce_sum_vector(
        accelerator, [sum_dpo_loss, sum_reward_acc, sum_gt_sft_loss, sum_total_loss, float(n_samples)]
    )
    policy_model.train()
    denom = max(int(round(n_samples)), 1)
    return {
        "val_dpo_loss": sum_dpo_loss / denom,
        "val_gt_sft_loss": sum_gt_sft_loss / denom,
        "val_total_loss": sum_total_loss / denom,
        "val_acc_pref": sum_reward_acc / denom,
    }


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def do_train_dpo(args, policy_model, ref_model, dataloader, logger, accelerator, val_dataloader=None):
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, policy_model.parameters()),
        lr=args.base_lr,
        weight_decay=args.weight_decay,
    )

    if args.pretrained_weights is not None:
        sd = torch.load(args.pretrained_weights, map_location=torch.device("cpu"))
        policy_model.load_state_dict(sd["model"], strict=True)
        logger.info("Loaded policy pretrained weights.")

    if args.ref_weights is not None:
        sd_ref = torch.load(args.ref_weights, map_location=torch.device("cpu"))
        ref_model.load_state_dict(sd_ref["model"], strict=True)
        logger.info("Loaded reference pretrained weights.")

    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    if accelerator.state.num_processes > 1:
        policy_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(policy_model)

    if val_dataloader is not None:
        dataloader, val_dataloader, policy_model, ref_model, optimizer = accelerator.prepare(
            dataloader, val_dataloader, policy_model, ref_model, optimizer
        )
    else:
        dataloader, policy_model, ref_model, optimizer = accelerator.prepare(
            dataloader, policy_model, ref_model, optimizer
        )

    max_iters = args.max_epoch * len(dataloader) // args.gradient_accumulation_steps
    time_delta = SmoothedValue(window_size=10)
    curr_iter = 0
    curr_time = time.time()
    loss_dict = defaultdict(list)

    beta = args.dpo_beta
    gt_sft_lambda = float(getattr(args, "gt_sft_lambda", 0.0) or 0.0)
    label_smoothing = getattr(args, "label_smoothing", 0.0)
    reference_free = getattr(args, "reference_free", False)

    policy_model.train()
    ref_model.eval()
    optimizer.zero_grad(set_to_none=True)
    stop_training = False

    best_metric_name = getattr(args, "best_metric", "val_acc_pref")
    save_best = bool(getattr(args, "save_best", True))
    early_stop_patience = int(getattr(args, "early_stop_patience", 0) or 0)
    early_stop_min_delta = float(getattr(args, "early_stop_min_delta", 0.0) or 0.0)
    min_iters_before_stop = int(getattr(args, "min_train_iters_before_stop", 0) or 0)
    if best_metric_name in {"val_dpo_loss", "val_total_loss"}:
        best_metric_value = float("inf")
    else:
        best_metric_value = float("-inf")
    no_improve_evals = 0

    def _is_better(curr_value, best_value):
        if best_metric_name in {"val_dpo_loss", "val_total_loss"}:
            return curr_value < (best_value - early_stop_min_delta)
        return curr_value > (best_value + early_stop_min_delta)

    for curr_epoch in range(args.max_epoch):
        for _, batch in enumerate(dataloader):
            curr_lr = adjust_learning_rate(args, optimizer, curr_iter / max(max_iters, 1))
            with accelerator.accumulate(policy_model):
                with accelerator.autocast():
                    # 2 forward passes total (1 policy, 1 reference) instead of 4
                    pi_c, pi_r = _concatenated_forward(
                        policy_model, batch["pc_normal"],
                        batch["chosen_sequence"], batch["rejected_sequence"],
                    )
                    with torch.no_grad():
                        ref_c, ref_r = _concatenated_forward(
                            ref_model, batch["pc_normal"],
                            batch["chosen_sequence"], batch["rejected_sequence"],
                        )

                    losses, chosen_rewards, rejected_rewards = preference_loss(
                        pi_c, pi_r, ref_c, ref_r,
                        beta=beta,
                        label_smoothing=label_smoothing,
                        reference_free=reference_free,
                    )
                    iou_gap_weights = _get_iou_gap_weights(batch, losses)
                    dpo_loss = (losses * iou_gap_weights).mean()
                    gt_sft_loss = _maybe_gt_sft_loss(
                        policy_model, batch, batch["pc_normal"], dpo_loss, gt_sft_lambda
                    )
                    loss = dpo_loss + gt_sft_lambda * gt_sft_loss

                    reward_acc = (chosen_rewards > rejected_rewards).float().mean()
                    reward_margin = (chosen_rewards - rejected_rewards).mean()

                if not math.isfinite(loss.item()):
                    logger.info("Loss is not finite. Terminate training.")
                    raise RuntimeError("Non-finite DPO loss")

                accelerator.backward(loss)

                if args.clip_gradient > 0 and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(policy_model.parameters(), args.clip_gradient)

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            loss_dict["loss_total"].append(loss.item())
            loss_dict["dpo_loss"].append(dpo_loss.item())
            loss_dict["gt_sft_loss"].append(gt_sft_loss.item())
            loss_dict["reward_chosen"].append(_reduce_mean(accelerator, chosen_rewards))
            loss_dict["reward_rejected"].append(_reduce_mean(accelerator, rejected_rewards))
            loss_dict["reward_margin"].append(_reduce_mean(accelerator, reward_margin))
            loss_dict["reward_acc"].append(_reduce_mean(accelerator, reward_acc))
            loss_dict["iou_gap_mean"].append(_reduce_mean(accelerator, iou_gap_weights))

            if accelerator.sync_gradients:
                time_delta.update(time.time() - curr_time)
                curr_time = time.time()
                curr_iter += 1

                if curr_iter % args.log_every == 0:
                    mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
                    eta_seconds = (max_iters - curr_iter) * time_delta.avg
                    eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
                    log_out = {}
                    for k, v in loss_dict.items():
                        log_out[k] = float(sum(v) / max(len(v), 1))
                    log_out["learning_rate"] = curr_lr
                    logger.info(
                        f"Epoch [{curr_epoch}/{args.max_epoch}] "
                        f"Iter [{curr_iter}/{max_iters}] "
                        f"metrics={log_out} "
                        f"Iter time {time_delta.avg:0.2f}; ETA {eta_str}; Mem {mem_mb:0.2f}MB"
                    )
                    accelerator.log(log_out, step=curr_iter)
                    loss_dict = defaultdict(list)

                eval_every = getattr(args, "eval_every", 0) or 0
                if (
                    val_dataloader is not None
                    and eval_every > 0
                    and curr_iter % eval_every == 0
                ):
                    val_metrics = _evaluate_dpo_on_loader(
                        policy_model, ref_model, val_dataloader, args, accelerator
                    )
                    if accelerator.is_main_process:
                        v_loss = val_metrics["val_dpo_loss"]
                        v_gt = val_metrics["val_gt_sft_loss"]
                        v_total = val_metrics["val_total_loss"]
                        v_acc = val_metrics["val_acc_pref"]
                        msg = (
                            f"[VAL] Epoch [{curr_epoch}/{args.max_epoch}] "
                            f"Iter [{curr_iter}/{max_iters}] "
                            f"val_dpo_loss={v_loss:.6f} "
                            f"val_gt_sft_loss={v_gt:.6f} "
                            f"val_total_loss={v_total:.6f} "
                            f"val_acc_pref={v_acc:.4f}"
                        )
                        logger.info(msg)
                    accelerator.log(val_metrics, step=curr_iter)

                    current_metric = float(val_metrics[best_metric_name])
                    improved = _is_better(current_metric, best_metric_value)
                    if improved:
                        best_metric_value = current_metric
                        no_improve_evals = 0
                        if accelerator.is_main_process and save_best:
                            save_checkpoint(
                                args.checkpoint_dir,
                                policy_model,
                                optimizer,
                                curr_epoch,
                                args,
                                filename="checkpoint_best.pth",
                            )
                            logger.info(
                                f"Saved best checkpoint at iter {curr_iter}: "
                                f"{best_metric_name}={current_metric:.6f}"
                            )
                    else:
                        no_improve_evals += 1
                        if (
                            early_stop_patience > 0
                            and curr_iter >= min_iters_before_stop
                            and no_improve_evals >= early_stop_patience
                        ):
                            if accelerator.is_main_process:
                                logger.info(
                                    "Early stopping triggered at "
                                    f"iter {curr_iter}: no improvement in {no_improve_evals} evals "
                                    f"(best {best_metric_name}={best_metric_value:.6f})."
                                )
                            stop_training = True

                if accelerator.is_main_process and curr_iter % args.save_every == 0:
                    save_checkpoint(
                        args.checkpoint_dir,
                        policy_model,
                        optimizer,
                        curr_epoch,
                        args,
                        filename=f"checkpoint_{curr_iter}.pth",
                    )

                if stop_training:
                    break
        if stop_training:
            break

    accelerator.end_training()
