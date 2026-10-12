"""Bounded, phase-aware CUDA allocator instrumentation (no change to model/loss).

Known parameter/gradient/optimizer storage is counted directly. Peak increments
are NOT labelled 'activation bytes': CUDA workspaces, outputs and other live
tensors may also contribute. Device-wide usage includes OTHER processes.
"""
from __future__ import annotations

from contextlib import contextmanager
import csv
import json
import os
from pathlib import Path
import time

import torch


def tensor_description(value):
    return dict(shape=list(value.shape), dtype=str(value.dtype), device=str(value.device),
                logical_bytes=value.numel() * value.element_size())


def _tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _tensors(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _tensors(item)


def cuda_storage_bytes(tensors, device, seen):
    """Count unique underlying CUDA storage, not views or duplicated aliases."""
    count = 0
    for value in tensors:
        if value.device.type != "cuda" or value.device.index != device.index:
            continue
        storage = value.untyped_storage()
        key = (value.device.index, storage.data_ptr())
        if key not in seen:
            seen.add(key)
            count += storage.nbytes()
    return count


def model_inventory(models, optimizer, device):
    seen, result, parameter_count, logical_bytes = set(), {}, 0, 0
    for role, model in models.items():
        if model is None:
            continue
        parameters = list(model.parameters())
        parameter_count += sum(p.numel() for p in parameters)
        logical_bytes += sum(p.numel() * p.element_size() for p in parameters)
        result[role + "_parameter_cuda_bytes"] = cuda_storage_bytes(parameters, device, seen)
        result[role + "_buffer_cuda_bytes"] = cuda_storage_bytes(model.buffers(), device, seen)
        result[role + "_gradient_cuda_bytes"] = cuda_storage_bytes(
            (p.grad for p in parameters if p.grad is not None), device, seen)
    result["optimizer_state_cuda_bytes"] = cuda_storage_bytes(
        _tensors(optimizer.state) if optimizer is not None else (), device, seen)
    result["known_cuda_storage_bytes"] = sum(result.values())
    result["resident_parameter_count"] = parameter_count
    result["resident_parameter_logical_bytes"] = logical_bytes
    if device.type != "cuda":
        # No fictional '0-byte GPU' efficiency result for CPU correctness tests.
        result = {k: (v if k in ("resident_parameter_count", "resident_parameter_logical_bytes") else None)
                  for k, v in result.items()}
    return result


class VramRecorder:
    def __init__(self, directory, device, models=None, optimizer=None, metadata=None,
                 first_updates=3, every_updates=200):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.models, self.optimizer = models or {}, optimizer
        self.first_updates, self.every_updates = first_updates, every_updates
        self.metadata = dict(metadata or {})
        self.metadata.update(pid=os.getpid(), torch_version=torch.__version__,
                             cuda_runtime=torch.version.cuda, device=str(self.device),
                             cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                             sampling=dict(first_updates=first_updates, every_updates=every_updates),
                             notes=["allocator peaks are this process's PyTorch memory, not all GPU allocations",
                                    "device-wide used/free includes other users and CUDA/library overhead",
                                    "known-storage categories deduplicate aliases; residual is NOT activation-only",
                                    "sampled phase boundaries synchronize CUDA and add timing overhead",
                                    "reported wall times are instrumented implementation times"])
        if self.device.type == "cuda":
            prop = torch.cuda.get_device_properties(self.device)
            self.metadata.update(gpu_name=prop.name, gpu_total_bytes=prop.total_memory)
        self.stack, self.aggregates, self.inventory_maxima = [], {}, {}
        self.highest_end_snapshot = None
        self.trace_path = self.directory / "memory_trace.jsonl"
        self.csv_path = self.directory / "memory_trace.csv"
        self.events, self.recovered_history = 0, False
        self.trace_fields = ("event", "phase", "pid", "wall_seconds", "elapsed_start_seconds", "elapsed_end_seconds",
                             "synchronized", "sampled_detail",
                             "epoch", "update", "position_start", "day", "allocated_start_bytes",
                             "allocated_end_bytes", "reserved_start_bytes", "reserved_end_bytes",
                             "peak_allocated_bytes", "peak_reserved_bytes", "peak_above_start_bytes",
                             "unused_cache_at_end_bytes", "known_cuda_storage_bytes",
                             "unattributed_live_at_end_bytes", "optimizer_state_cuda_bytes",
                             "device_used_end_bytes", "device_free_end_bytes", "inactive_split_end_bytes",
                             "allocator_retries_end", "allocator_ooms_end", "host_intermediate_logical_bytes",
                             "context_json", "inventory_json", "tensors_json")
        # Resume previous profiling (not model training): preserve existing trace rows.
        if self.trace_path.exists():
            with self.trace_path.open(encoding="utf-8") as stream:
                for line in stream:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # interrupted final JSONL line
                    self.events += 1
                    self._aggregate(row)
            self.recovered_history = True
        self.trace_stream = self.trace_path.open("a", encoding="utf-8", buffering=1)
        if self.trace_path.stat().st_size:
            with self.trace_path.open("rb") as previous:
                previous.seek(-1, 2)
                if previous.read(1) != b"\n":
                    self.trace_stream.write("\n")
        # CSV is regenerated from JSONL at close, never append beneath a partial row.
        self.run_started = time.perf_counter()
        self.profiling_sync_seconds = 0.0
        self.host_intermediate_peak = 0

    def selected(self, update):
        return update <= self.first_updates or update % self.every_updates == 0

    def _sync(self):
        if self.device.type == "cuda":
            started = time.perf_counter()
            torch.cuda.synchronize(self.device)
            self.profiling_sync_seconds += time.perf_counter() - started

    def _snapshot(self, detailed):
        if self.device.type != "cuda":
            return dict(allocated_bytes=None, reserved_bytes=None)
        result = dict(allocated_bytes=torch.cuda.memory_allocated(self.device),
                      reserved_bytes=torch.cuda.memory_reserved(self.device))
        if detailed:
            free, total = torch.cuda.mem_get_info(self.device)
            stats = torch.cuda.memory_stats(self.device)
            result.update(device_free_bytes=free, device_used_bytes=total - free,
                          inactive_split_bytes=stats.get("inactive_split_bytes.all.current"),
                          allocator_retries=stats.get("num_alloc_retries"), allocator_ooms=stats.get("num_ooms"))
        return result

    def _capture_peak(self):
        if self.device.type != "cuda":
            return
        allocated = torch.cuda.max_memory_allocated(self.device)
        reserved = torch.cuda.max_memory_reserved(self.device)
        for scope in self.stack:
            scope["peak_allocated_bytes"] = max(scope["peak_allocated_bytes"], allocated)
            scope["peak_reserved_bytes"] = max(scope["peak_reserved_bytes"], reserved)

    def _reset_peak(self):
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    @contextmanager
    def section(self, phase, context=None, detailed=True):
        if detailed:
            self._sync()
        self._capture_peak()  # propagate earlier allocator segment into parent scopes
        begin = self._snapshot(detailed)
        scope = dict(phase=phase, context=dict(context or {}), begin=begin, tensors={},
                     peak_allocated_bytes=begin["allocated_bytes"],
                     peak_reserved_bytes=begin["reserved_bytes"], started=time.perf_counter())
        self.stack.append(scope)
        self._reset_peak()
        try:
            yield self
        finally:
            if detailed:
                self._sync()
            self._capture_peak()
            end = self._snapshot(detailed)
            seconds = time.perf_counter() - scope["started"]
            self.stack.pop()
            inventory = model_inventory(self.models, self.optimizer, self.device) if detailed else {}
            known = inventory.get("known_cuda_storage_bytes")
            row = dict(event="scope", phase=phase, pid=os.getpid(), wall_seconds=seconds,
                       elapsed_start_seconds=scope["started"] - self.run_started,
                       elapsed_end_seconds=time.perf_counter() - self.run_started,
                       synchronized=detailed, sampled_detail=detailed,
                       allocated_start_bytes=begin["allocated_bytes"], allocated_end_bytes=end["allocated_bytes"],
                       reserved_start_bytes=begin["reserved_bytes"], reserved_end_bytes=end["reserved_bytes"],
                       peak_allocated_bytes=scope["peak_allocated_bytes"], peak_reserved_bytes=scope["peak_reserved_bytes"],
                       peak_above_start_bytes=(max(0, scope["peak_allocated_bytes"] - begin["allocated_bytes"])
                                               if self.device.type == "cuda" else None),
                       unused_cache_at_end_bytes=(end["reserved_bytes"] - end["allocated_bytes"]
                                                  if self.device.type == "cuda" else None),
                       unattributed_live_at_end_bytes=(max(0, end["allocated_bytes"] - known)
                                                       if known is not None else None),
                       inventory=inventory, tensors=scope["tensors"], context=scope["context"], **inventory)
            for key in ("device_used_bytes", "device_free_bytes", "inactive_split_bytes",
                        "allocator_retries", "allocator_ooms"):
                stem = key[:-6] if key.endswith("_bytes") else key
                row[stem + "_end" + ("_bytes" if key.endswith("_bytes") else "")] = end.get(key)
            row["host_intermediate_logical_bytes"] = scope["context"].get("host_intermediate_logical_bytes")
            self._emit(row)
            self._reset_peak()  # parent scope continues with a fresh allocator segment

    def tensors(self, **values):
        if not self.stack:
            return
        # Keep only metadata, never references to live tensors/autograd graphs.
        for name, value in values.items():
            if isinstance(value, torch.Tensor):
                self.stack[-1]["tensors"][name] = tensor_description(value)

    def host_intermediate(self, name, logical_bytes):
        self.host_intermediate_peak = max(self.host_intermediate_peak, int(logical_bytes))
        if self.stack:
            scope = self.stack[-1]
            scope["context"]["host_intermediate_name"] = name
            scope["context"]["host_intermediate_logical_bytes"] = int(logical_bytes)

    def _aggregate(self, row):
        phase = row["phase"]
        record = self.aggregates.setdefault(phase, dict(count=0, max_allocated_bytes=None,
                                                       max_reserved_bytes=None, synchronized_count=0,
                                                       synchronized_wall_seconds=0.0, max_increment_bytes=None))
        record["count"] += 1
        for source, destination in (("peak_allocated_bytes", "max_allocated_bytes"),
                                    ("peak_reserved_bytes", "max_reserved_bytes"),
                                    ("peak_above_start_bytes", "max_increment_bytes")):
            value = row.get(source)
            if value is not None:
                record[destination] = max(record[destination] or 0, value)
        if row.get("synchronized"):
            record["synchronized_count"] += 1
            record["synchronized_wall_seconds"] += row["wall_seconds"]
        for key, value in row.get("inventory", {}).items():
            if value is not None:
                self.inventory_maxima[key] = max(self.inventory_maxima.get(key, 0), value)
        allocated = row.get("allocated_end_bytes")
        if allocated is not None and row.get("inventory") and (
                self.highest_end_snapshot is None or allocated > self.highest_end_snapshot["allocated_end_bytes"]):
            self.highest_end_snapshot = dict(phase=row["phase"], pid=row["pid"],
                                            context=row.get("context", {}), allocated_end_bytes=allocated,
                                            known_storage_at_scope_end=row["inventory"],
                                            unattributed_live_at_end_bytes=row.get("unattributed_live_at_end_bytes"),
                                            note="snapshot at a phase boundary, NOT a decomposition at the allocator peak")

    def _emit(self, row):
        self.events += 1
        self._aggregate(row)
        self.trace_stream.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    def peaks(self, phases=None):
        phases = phases or list(self.aggregates)
        records = [self.aggregates[p] for p in phases if p in self.aggregates]
        return dict(max_allocated_bytes=max((r["max_allocated_bytes"] or 0 for r in records), default=0) or None,
                    max_reserved_bytes=max((r["max_reserved_bytes"] or 0 for r in records), default=0) or None)

    def close(self):
        self.trace_stream.close()
        with self.csv_path.open("w", encoding="utf-8-sig", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=self.trace_fields, extrasaction="ignore")
            writer.writeheader()
            with self.trace_path.open(encoding="utf-8") as source:
                for line in source:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    row.update({k: v for k, v in row.get("context", {}).items() if k in self.trace_fields})
                    for key in ("context", "inventory", "tensors"):
                        row[key + "_json"] = json.dumps(row.get(key, {}), ensure_ascii=False)
                    writer.writerow(row)
        summary = dict(metadata=self.metadata, phases=self.aggregates, peaks=self.peaks(),
                       events=self.events, resumed_trace=self.recovered_history,
                       instrumentation_cuda_sync_wait_seconds=self.profiling_sync_seconds,
                       host_intermediate_peak_logical_bytes=self.host_intermediate_peak,
                       inventory_maxima=self.inventory_maxima,
                       highest_allocated_end_snapshot=self.highest_end_snapshot,
                       elapsed_this_process_seconds=time.perf_counter() - self.run_started,
                       note="phase scopes can nest; do NOT sum nested wall times or peak bytes")
        (self.directory / "memory_profile.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return summary
