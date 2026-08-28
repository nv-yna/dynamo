# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Per-request TRT-LLM engine perf-metrics JSONL dump ("Option A", vendored).

TRT-LLM measures rich per-request timing (engine queue, prefill, measured KV
transfer) in ``RequestOutput.request_perf_metrics``, but under Dynamo that
object dies in worker memory: TRT-LLM's own JSONL writer lives in its OpenAI
serving layer (``tensorrt_llm/serve/perf_metrics.py``), which Dynamo bypasses,
and older wheels (<= 1.3.0rc21) do not ship it at all. This module is a
self-contained equivalent: it reads the perf-metrics object directly and
appends one JSON line per finished request, in a shape compatible with the
trtllm-serve worker record (``request_id`` + ``perf_metrics.timing_metrics``
+ ``kv_cache_metrics`` + disagg IDs) plus one extra field TRT-LLM cannot
know — ``dynamo_request_id`` — which makes the row joinable to Dynamo's
request trace, spans, and logs with no auxiliary mapping.

Enabled by setting ``DYN_TRTLLM_PERF_METRICS_DIR`` to a writable directory;
disabled (fast no-op) otherwise. One file per worker process:
``perf_metrics-dynamo-<hostname>-<pid>.jsonl``. All timestamps inside
``timing_metrics`` are the worker host's steady clock (seconds) — only
differences within one record are meaningful; never compare them across
processes.

Failure policy: never break serving. Every write is wrapped; on error the
module logs one warning, counts drops, and keeps going.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import threading
from typing import Any, Optional

logger = logging.getLogger(__name__)

_ENV_DIR = "DYN_TRTLLM_PERF_METRICS_DIR"

# Timing fields mirrored from tensorrt_llm.bindings.executor.RequestPerfMetrics
# .timing_metrics; every one is optional because availability varies by wheel
# version, request mode (agg vs disagg role), and cache-hit shape.
_TIMING_FIELDS = (
    "arrival_time",
    "first_scheduled_time",
    "first_token_time",
    "last_token_time",
    "server_arrival_time",
    "server_first_token_time",
    "kv_cache_transfer_start",
    "kv_cache_transfer_end",
    "kv_cache_size",
)

_KV_FIELDS = (
    "num_total_allocated_blocks",
    "num_new_allocated_blocks",
    "num_reused_blocks",
    "num_missed_blocks",
)

_lock = threading.Lock()
_state: dict[str, Any] = {"fh": None, "path": None, "failed": False, "drops": 0}


def enabled() -> bool:
    return bool(os.environ.get(_ENV_DIR))


def _to_seconds(value: Any) -> Optional[float]:
    """Coerce a timing value to float seconds; None for missing/non-finite.

    Depending on the TRT-LLM wheel, timing fields are floats, ints, or
    ``datetime.timedelta``-like objects exposing ``total_seconds()``.
    """
    if value is None:
        return None
    total_seconds = getattr(value, "total_seconds", None)
    if callable(total_seconds):
        try:
            value = total_seconds()
        except Exception:
            return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f or f in (float("inf"), float("-inf")):
        return None
    return f


def _open_once() -> Optional[Any]:
    if _state["failed"]:
        return None
    if _state["fh"] is not None:
        return _state["fh"]
    out_dir = os.environ.get(_ENV_DIR)
    if not out_dir:
        return None
    try:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(
            out_dir, f"perf_metrics-dynamo-{socket.gethostname()}-{os.getpid()}.jsonl"
        )
        # Line-buffered append: one JSON object per line, flushed per write, so
        # a killed worker loses at most the in-flight line.
        _state["fh"] = open(path, "a", buffering=1, encoding="utf-8")
        _state["path"] = path
        logger.info("perf_metrics_dump: writing per-request engine metrics to %s", path)
        return _state["fh"]
    except OSError as e:
        _state["failed"] = True
        logger.warning("perf_metrics_dump: disabled, cannot open output: %s", e)
        return None


def maybe_dump_perf_metrics(
    request_id: str,
    role: str,
    generation_result: Any,
    res: Any,
    disaggregated_params: Any,
) -> None:
    """Append one JSONL record for a finished request; fast no-op when disabled.

    Call once per request at the first ``res.finished`` observation. All
    arguments are read defensively — a wheel that lacks any field simply
    produces a sparser record, never an exception on the serving path.
    """
    if not enabled():
        return
    try:
        output = None
        for candidate in getattr(res, "outputs", None) or []:
            if getattr(candidate, "request_perf_metrics", None) is not None:
                output = candidate
                break
        record: dict[str, Any] = {
            "schema": "dynamo.trtllm.perf_metrics.v0",
            # The join bridge: no TRT-LLM-native record can carry this.
            "dynamo_request_id": request_id,
            "role": role,
            # Executor client ID — a per-worker-process counter; join it only
            # within this worker's own artifacts. Same value the "Engine ID
            # map" log line carries as trtllm_client_id.
            "request_id": _maybe_str(getattr(generation_result, "request_id", None)),
            "disagg_request_id": _maybe_str(
                getattr(disaggregated_params, "disagg_request_id", None)
                if disaggregated_params is not None
                else None
            ),
        }
        if output is not None:
            record["finish_reason"] = getattr(output, "finish_reason", None)
            pm = output.request_perf_metrics
            perf: dict[str, Any] = {}
            for attr in ("first_iter", "last_iter"):
                value = getattr(pm, attr, None)
                if value is not None:
                    perf[attr] = value
            timing = getattr(pm, "timing_metrics", None)
            if timing is not None:
                perf["timing_metrics"] = {
                    name: (
                        getattr(timing, name, None)
                        if name == "kv_cache_size"
                        else _to_seconds(getattr(timing, name, None))
                    )
                    for name in _TIMING_FIELDS
                }
            kv = getattr(pm, "kv_cache_metrics", None)
            if kv is not None:
                perf["kv_cache_metrics"] = {
                    name: getattr(kv, name, None) for name in _KV_FIELDS
                }
            record["perf_metrics"] = perf
            # ctx_request_id rides the OUTPUT's disaggregated params (set by the
            # engine), unlike disagg_request_id which rides the request's.
            out_disagg = getattr(output, "disaggregated_params", None)
            if out_disagg is not None:
                record["ctx_request_id"] = _maybe_str(
                    getattr(out_disagg, "ctx_request_id", None)
                )
        line = json.dumps(record, default=str) + "\n"
        with _lock:
            fh = _open_once()
            if fh is None:
                _state["drops"] += 1
                return
            fh.write(line)
    except Exception as e:  # never break the serving path for telemetry
        with _lock:
            _state["drops"] += 1
            if _state["drops"] == 1 or _state["drops"] % 1000 == 0:
                logger.warning(
                    "perf_metrics_dump: dropped %d records (last error: %s)",
                    _state["drops"],
                    e,
                )


def _maybe_str(value: Any) -> Optional[str]:
    """64-bit engine IDs as strings so JSON consumers cannot corrupt them."""
    return None if value is None else str(value)
