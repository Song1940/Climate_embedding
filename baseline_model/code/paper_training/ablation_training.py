"""Native stage batches, one shared raw-variable loss, convergence and real resume.

Frozen prefix decoders are differentiable with respect to their inputs. The
chunked adjoint below computes the exact first-order gradient of that loss;
it is NOT a latent-target loss and does not stop the gradient at a frozen decoder.
"""
from __future__ import annotations

import gc
import math
import random
import time
from pathlib import Path
from contextlib import nullcontext

import numpy as np
import torch

from ablation_models import balanced_mse
from ablation_storage import atomic_save, fingerprint, read_json
from common import autocast_context, cpu_state, write_json
from step3_model import pad_even
from vram_monitor import VramRecorder


def tensor(value, device):
    return torch.as_tensor(np.array(value, dtype=np.float32, copy=True), device=device)


def reset_peak(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory(device):
    if device.type != "cuda":
        return dict(max_allocated_bytes=None, max_reserved_bytes=None)
    torch.cuda.synchronize(device)
    return dict(max_allocated_bytes=torch.cuda.max_memory_allocated(device),
                max_reserved_bytes=torch.cuda.max_memory_reserved(device))


def frozen_raw_adjoint(codes, targets, decoder, weights, chunk, profiler=None):
    """Loss and dLoss/dCodes, exact for a deterministic frozen V decoder.

    Dividing each chunk's mean loss by its relative sample count yields the
    identical gradient to one full batch, without retaining all V activations.
    """
    n = len(codes)
    gradients = torch.empty_like(codes, dtype=torch.float32)
    value = 0.0
    for start in range(0, n, chunk):
        stop = min(n, start + chunk)
        scope = (profiler.section("frozen_V_decode_backward_chunk", dict(samples=stop - start))
                 if profiler is not None and start == 0 else nullcontext())
        with scope:
            local = codes[start:stop].detach().float().requires_grad_(True)
            prediction = decoder.decode(local)
            if profiler is not None and start == 0:
                profiler.tensors(input_codes=local, prediction=prediction, target=targets[start:stop])
            loss = balanced_mse(prediction, targets[start:stop], weights) * ((stop - start) / n)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite raw reconstruction loss")
            loss.backward()
        gradients[start:stop] = local.grad
        value += float(loss.detach())
    return value, gradients


def decode_raw_loss(codes, targets, v_decoder, weights, chunk, backward, profiler=None):
    if v_decoder is None:
        loss = balanced_mse(codes, targets, weights)
        if backward:
            loss.backward()
        return float(loss.detach())
    if backward:
        value, gradient = frozen_raw_adjoint(codes, targets, v_decoder, weights, chunk, profiler)
        (codes.float() * gradient).sum().backward()
        return value
    total = 0.0
    for start in range(0, len(codes), chunk):
        stop = min(start + chunk, len(codes))
        loss = balanced_mse(v_decoder.decode(codes[start:stop]), targets[start:stop], weights)
        total += float(loss) * (stop - start) / len(codes)
    return total


def temporal_restore(model, source, device, chunk):
    days, positions, channels = source.shape
    output = np.empty_like(source, dtype=np.float32)
    for start in range(0, positions, chunk):
        stop = min(positions, start + chunk)
        sequence = tensor(source[:, start:stop], device).permute(1, 2, 0)
        prediction, _ = model(pad_even(model.normalize(sequence)))
        restored = model.denormalize(prediction)[..., :days].permute(2, 0, 1)
        output[:, start:stop] = restored.float().detach().cpu().numpy()
    return output


class StageObjective:
    def __init__(self, stage, model, raw, source, weights, device, args,
                 variable_decoder=None, spatial_decoder=None):
        self.stage, self.model, self.raw, self.source = stage, model, raw, source
        self.weights, self.device, self.args = weights, device, args
        self.v, self.s = variable_decoder, spatial_decoder
        self.profiler, self.profile_detailed, self.profile_context = None, False, {}
        self.rng = np.random.default_rng(args.seed + 173)
        fixed = np.random.default_rng(args.seed + 901)
        self.check_grids = np.sort(fixed.choice(raw.shape[1], min(args.check_grids, raw.shape[1]), False))
        self.check_days = np.unique(np.linspace(0, len(raw) - 1, min(args.check_days, len(raw))).round().astype(int))

    def phase(self, name):
        return (self.profiler.section(name, self.profile_context) if self.profile_detailed
                and self.profiler is not None else nullcontext())

    def trace_tensors(self, **values):
        if self.profile_detailed and self.profiler is not None:
            self.profiler.tensors(**values)

    def active_profiler(self):
        return self.profiler if self.profile_detailed else None

    def step(self):
        a, raw, device = self.args, self.raw, self.device
        if self.stage == "V":
            pairs = self.rng.integers(0, raw.shape[0] * raw.shape[1], size=a.v_batch)
            days, grids = np.divmod(pairs, raw.shape[1])
            with self.phase("input_transfer"):
                target = tensor(raw[days, grids], device)
                self.trace_tensors(target=target)
            with self.phase("forward"):
                prediction, latent = self.model(target)
                self.trace_tensors(prediction=prediction, latent=latent)
            with self.phase("raw_loss_backward"):
                loss = balanced_mse(prediction, target, self.weights)
                loss.backward()
            return float(loss.detach()), len(pairs)
        if self.stage == "S":
            day = int(self.rng.integers(len(raw)))
            grids = self.rng.choice(raw.shape[1], min(a.s_targets, raw.shape[1]), False)
            with self.phase("input_transfer"):
                inputs = tensor(self.source[day:day + 1], device)
                self.trace_tensors(full_spatial_input=inputs)
            with self.phase("forward"):
                prediction, latent = self.model(inputs)
                del inputs  # preserve the original lifetime of the temporary input
                self.trace_tensors(prediction=prediction, latent=latent)
            with self.phase("raw_loss_backward"):
                loss = decode_raw_loss(prediction[0, grids], tensor(raw[day, grids], device),
                                       self.v, self.weights, a.decoder_chunk, True, self.active_profiler())
            return loss, len(grids)
        if self.s is None:
            # T-only / VT: native per-position batches; all dates of those grids are targets.
            grids = self.rng.choice(raw.shape[1], min(a.t_batch, raw.shape[1]), False)
            with self.phase("input_transfer"):
                sequence = tensor(self.source[:, grids], device).permute(1, 2, 0)
                self.trace_tensors(sequence=sequence)
            with self.phase("forward"):
                prediction, latent = self.model(pad_even(self.model.normalize(sequence)))
                prediction = self.model.denormalize(prediction)[..., :len(raw)].permute(2, 0, 1)
                self.trace_tensors(prediction=prediction, latent=latent)
            with self.phase("raw_loss_backward"):
                targets = tensor(raw[:, grids], device).reshape(-1, raw.shape[-1])
                self.trace_tensors(target=targets)
                loss = decode_raw_loss(prediction.reshape(-1, prediction.shape[-1]), targets,
                                       self.v, self.weights, a.decoder_chunk, True, self.active_profiler())
            return loss, len(raw) * len(grids)
        # VST: S decoder is global. Never fill unselected tokens with truthful source
        # codes. First decode every T position, obtain the raw-loss spatial adjoint,
        # then recompute each deterministic T batch to backpropagate that adjoint.
        with self.phase("temporal_preview"), torch.no_grad():
            predicted = temporal_restore(self.model, self.source, device, a.t_batch)
        gradient = np.zeros_like(predicted)
        if self.profiler is not None:
            self.profiler.host_intermediate("temporal_preview_and_adjoint_arrays", predicted.nbytes + gradient.nbytes)
        day_ids = self.rng.choice(len(raw), min(a.vst_days, len(raw)), False)
        value, pairs = 0.0, 0
        for day in day_ids:
            grids = self.rng.choice(raw.shape[1], min(a.s_targets, raw.shape[1]), False)
            with self.phase("spatial_raw_adjoint"):
                local = tensor(predicted[day:day + 1], device).requires_grad_(True)
                grid_codes = self.s.decode(local)[0, grids]
                targets = tensor(raw[day, grids], device)
                self.trace_tensors(full_spatial_tokens=local, grid_codes=grid_codes, targets=targets)
                part, adjoint = frozen_raw_adjoint(grid_codes, targets, self.v, self.weights,
                                                  a.decoder_chunk, self.active_profiler())
                (grid_codes.float() * (adjoint / len(day_ids))).sum().backward()
                gradient[day] = local.grad[0].float().cpu().numpy()
            value += part / len(day_ids)
            pairs += len(grids)
        with self.phase("temporal_recompute_backward"):
            for start in range(0, self.source.shape[1], a.t_batch):
                stop = min(start + a.t_batch, self.source.shape[1])
                sequence = tensor(self.source[:, start:stop], device).permute(1, 2, 0)
                prediction, latent = self.model(pad_even(self.model.normalize(sequence)))
                prediction = self.model.denormalize(prediction)[..., :len(raw)].permute(2, 0, 1)
                if start == 0:
                    self.trace_tensors(sequence_chunk=sequence, prediction_chunk=prediction, latent_chunk=latent)
                (prediction.float() * tensor(gradient[:, start:stop], device)).sum().backward()
        return value, pairs

    @torch.no_grad()
    def check(self):
        """Identical fixed raw (day, grid) sample. NOT a held-out validation set."""
        raw, device, grids, days = self.raw, self.device, self.check_grids, self.check_days
        target = tensor(raw[days[:, None], grids[None, :]], device)
        if self.stage == "V":
            prediction, _ = self.model(target.reshape(-1, raw.shape[-1]))
            return float(balanced_mse(prediction, target.reshape(-1, raw.shape[-1]), self.weights))
        if self.stage == "T" and self.s is None:
            sequence = tensor(self.source[:, grids], device).permute(1, 2, 0)
            prediction, _ = self.model(pad_even(self.model.normalize(sequence)))
            codes = self.model.denormalize(prediction)[..., :len(raw)].permute(2, 0, 1)[days]
            return decode_raw_loss(codes.reshape(-1, codes.shape[-1]), target.reshape(-1, raw.shape[-1]),
                                   self.v, self.weights, self.args.decoder_chunk, False)
        spatial = (temporal_restore(self.model, self.source, device, self.args.t_batch)
                   if self.stage == "T" else self.source)
        value = 0.0
        for day in days:
            if self.stage == "S":
                codes, _ = self.model(tensor(spatial[day:day + 1], device))
            else:
                codes = self.s.decode(tensor(spatial[day:day + 1], device))
            value += decode_raw_loss(codes[0, grids], tensor(raw[day, grids], device),
                                     self.v, self.weights, self.args.decoder_chunk, False) / len(days)
        return value


def train_stage(directory, stage, model, objective, args, protocol_hash):
    directory.mkdir(parents=True, exist_ok=True)
    signature = fingerprint(dict(protocol=protocol_hash, stage=stage, config=model.config,
                                  training={k: vars(args)[k] for k in (
                                      "v_epochs", "s_epochs", "t_epochs", "v_steps", "s_steps", "t_steps",
                                      "extension_factor", "patience", "relative_improvement", "seed",
                                      "v_batch", "s_targets", "t_batch", "vst_days", "decoder_chunk",
                                      "check_grids", "check_days")}))
    report_path, best_path = directory / "training.json", directory / "best.pt"
    last_path = directory / "last.pt"
    if report_path.exists():
        report = read_json(report_path)
        if report["signature"] != signature:
            raise ValueError(f"changed configuration in {directory}; choose a new output folder")
        model.load_state_dict(torch.load(best_path, map_location=objective.device, weights_only=False)["state_dict"])
        print(f"[reuse completed {stage}] {directory}", flush=True)
        return report
    key = stage.lower()
    initial_epochs, steps = getattr(args, f"{key}_epochs"), getattr(args, f"{key}_steps")
    maximum = max(initial_epochs, math.ceil(initial_epochs * args.extension_factor))
    minimum = max(1, math.ceil(initial_epochs * .2))
    lr = {"V": 1e-3, "S": 3e-4, "T": 5e-4}[stage]
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.01)
    history, best, significant_best, stale, start_epoch = [], math.inf, math.inf, 0, 1
    updates, pairs, previous_elapsed = 0, 0, 0.0
    previous_peak = dict(max_allocated_bytes=None, max_reserved_bytes=None)
    if last_path.exists():
        saved = torch.load(last_path, map_location="cpu", weights_only=False)
        if saved["signature"] != signature:
            raise ValueError(f"resume fingerprint mismatch: {directory}")
        model.load_state_dict(saved["state_dict"])
        optimizer.load_state_dict(saved["optimizer"])
        history, best, significant_best, stale, updates, pairs = (
            saved[k] for k in ("history", "best", "significant_best", "stale", "updates", "pairs"))
        start_epoch = saved["epoch"] + 1
        previous_elapsed = saved["elapsed_seconds"]
        previous_peak = saved["peak"]
        objective.rng.bit_generator.state = saved["sampler_rng"]
        torch.set_rng_state(saved["torch_rng"])
        np.random.set_state(saved["numpy_rng"])
        random.setstate(saved["python_rng"])
        if objective.device.type == "cuda":
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        print(f"[resume {stage}] epoch {start_epoch}, optimizer and RNG restored", flush=True)
    profiler = VramRecorder(directory / "vram", objective.device,
                            models={"trainable": model, "frozen_V": objective.v, "frozen_S": objective.s},
                            optimizer=optimizer, first_updates=getattr(args, "vram_first_updates", 3),
                            every_updates=getattr(args, "vram_every_updates", 200),
                            metadata=dict(stage=stage, role="training", signature=signature,
                                          source_shape=list(objective.source.shape), raw_shape=list(objective.raw.shape),
                                          native_batches=dict(V=args.v_batch, S=args.s_targets, T=args.t_batch,
                                                              VST_days=args.vst_days, decoder_chunk=args.decoder_chunk)))
    objective.profiler = profiler
    started, status = time.perf_counter(), "maximum_budget_not_converged"
    memory = previous_peak
    # A crash between saving the last epoch and writing training.json must not
    # add another epoch after the declared plateau stopping condition was met.
    if start_epoch > minimum and stale >= args.patience:
        status = "plateau_under_declared_rule"
        maximum_for_loop = start_epoch - 1
    else:
        maximum_for_loop = maximum
    try:
        for epoch in range(start_epoch, maximum_for_loop + 1):
            model.train()
            losses = []
            with profiler.section("train_epoch", dict(epoch=epoch, update=updates)):
                for _ in range(steps):
                    progress = min(updates / max(initial_epochs * steps, 1), 1.0)
                    current_lr = lr * (.05 + .95 * .5 * (1 + math.cos(math.pi * progress)))
                    for group in optimizer.param_groups:
                        group["lr"] = current_lr
                    objective.profile_context = dict(epoch=epoch, update=updates + 1)
                    objective.profile_detailed = profiler.selected(updates + 1)
                    detail = (profiler.section("train_update", objective.profile_context)
                              if objective.profile_detailed else nullcontext())
                    with detail:
                        optimizer.zero_grad(set_to_none=True)
                        with autocast_context(objective.device):
                            loss, n_pairs = objective.step()
                        if not math.isfinite(loss):
                            raise FloatingPointError(f"nonfinite loss in {directory}")
                        with objective.phase("gradient_clip_optimizer"):
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                            optimizer.step()
                        if objective.profile_detailed:
                            profiler.stack[-1]["context"]["raw_date_grid_pairs"] = n_pairs
                    objective.profile_detailed = False
                    losses.append(loss)
                    updates += 1
                    pairs += n_pairs
            model.eval()
            with profiler.section("fixed_training_check", dict(epoch=epoch, update=updates)), autocast_context(objective.device):
                check = objective.check()
            if not math.isfinite(check):
                raise FloatingPointError("nonfinite fixed training check")
            if check < best:
                best = check
                atomic_save(dict(config=model.config, state_dict=cpu_state(model), epoch=epoch,
                                 check_loss=check, signature=signature), best_path)
            if check < significant_best * (1 - args.relative_improvement):
                significant_best, stale = check, 0
            else:
                stale += 1
            memory = profiler.peaks(("train_epoch", "fixed_training_check"))
            memory = {k: max(memory[k] or 0, previous_peak[k] or 0) or None for k in memory}
            history.append(dict(epoch=epoch, updates=updates, train_loss=float(np.mean(losses)),
                                fixed_training_check_loss=check, lr=current_lr, **memory))
            memory_display = (f"{memory['max_allocated_bytes'] / 2**30:.3f}"
                              if memory['max_allocated_bytes'] is not None else "N/A (CPU)")
            print(f"[{directory.parent.name}/{stage} epoch {epoch:03d}] "
                  f"train={np.mean(losses):.6f} fixed_train_check={check:.6f} "
                  f"updates={updates} stale={stale} peak_alloc_GiB="
                  f"{memory_display}", flush=True)
            atomic_save(dict(config=model.config, state_dict=cpu_state(model), optimizer=optimizer.state_dict(),
                             epoch=epoch, history=history, best=best, significant_best=significant_best,
                             stale=stale, updates=updates, pairs=pairs, signature=signature,
                             sampler_rng=objective.rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                             cuda_rng=torch.cuda.get_rng_state_all() if objective.device.type == "cuda" else [],
                             numpy_rng=np.random.get_state(), python_rng=random.getstate(),
                             elapsed_seconds=previous_elapsed + time.perf_counter() - started, peak=memory), last_path)
            if epoch >= minimum and stale >= args.patience:
                status = "plateau_under_declared_rule"
                break
    except BaseException as error:
        profiler.metadata["failure"] = dict(type=type(error).__name__, message=str(error))
        profiler.close()
        objective.profiler = None
        raise
    profile = profiler.close()
    objective.profiler = None
    report = dict(signature=signature, convergence_status=status, epochs=len(history),
                  initial_epoch_budget=initial_epochs, maximum_epochs=maximum, updates=updates,
                  supervised_raw_date_grid_pairs=pairs, best_fixed_training_check_loss=best,
                  training_seconds=previous_elapsed + time.perf_counter() - started, **memory,
                  parameter_count=sum(p.numel() for p in model.parameters()), history=history,
                  validation=False, objective="equal-weight original normalized variable MSE only",
                  upstream_frozen=True, SE=hasattr(model, "enc_se"), vram_profile=profile,
                  training_update_peak=profiler.peaks(("train_epoch",)),
                  fixed_check_peak=profiler.peaks(("fixed_training_check",)))
    write_json(report_path, report)
    model.load_state_dict(torch.load(best_path, map_location=objective.device, weights_only=False)["state_dict"])
    return report


@torch.inference_mode()
def encode_archive(stage, model, source, path, device, chunk):
    if path.exists():
        raise FileExistsError(path)
    model.eval()
    days, positions = source.shape[:2]
    if stage == "V":
        shape = (days, positions, model.config["embed_dim"])
    elif stage == "S":
        shape = (days, model.token_count, model.config["latent_dim"])
    else:
        shape = ((days + 1) // 2, positions, model.config["latent_channels"])
    temporary = path.with_name(path.stem + ".pending.npy")
    result = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=shape)
    with autocast_context(device):
        if stage == "S":
            for day in range(days):
                result[day] = model.encode(tensor(source[day:day + 1], device))[0].float().cpu().numpy()
        else:
            for day in range(days if stage == "V" else 1):
                for start in range(0, positions, chunk):
                    stop = min(start + chunk, positions)
                    if stage == "V":
                        values = model.encode(tensor(source[day, start:stop], device))
                        result[day, start:stop] = values.float().cpu().numpy()
                    else:
                        sequence = tensor(source[:, start:stop], device).permute(1, 2, 0)
                        values = model.encode(pad_even(model.normalize(sequence)))
                        result[:, start:stop] = values.permute(2, 0, 1).float().cpu().numpy()
    result.flush()
    del result
    temporary.replace(path)
    return np.load(path, mmap_mode="r")


def global_statistics(source, chunk=256):
    """Global across time AND position; never store a location-dependent mean map."""
    c = source.shape[-1]
    mean, m2, n = np.zeros(c, np.float64), np.zeros(c, np.float64), 0
    for start in range(0, source.shape[1], chunk):
        values = np.array(source[:, start:start + chunk], dtype=np.float64).reshape(-1, c)
        block_n = len(values)
        block_mean = values.mean(0)
        delta = block_mean - mean
        m2 += ((values - block_mean) ** 2).sum(0) + delta ** 2 * n * block_n / (n + block_n)
        mean += delta * block_n / (n + block_n)
        n += block_n
    std = np.sqrt(m2 / max(n, 1))
    std[std < 1e-6] = 1.0
    return mean[None].astype(np.float32), std.astype(np.float32)
