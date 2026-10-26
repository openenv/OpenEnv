# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Process-based math verification service for high-throughput answer grading.

This service handles answer verification using math_verify in isolated worker
processes to safely handle signal-based timeouts and enable concurrent grading.

Architecture:
- VerifierService: Async orchestrator that routes requests to worker processes
- ProcessPoolExecutor: Manages worker process lifecycle
- Request/response contract: Minimal serializable payloads
- Status mapping: correct, wrong, no_answer, unparsable, timeout, internal_error
"""

import asyncio
import logging
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import Optional

import math_verify

logger = logging.getLogger(__name__)


@dataclass
class VerifyRequest:
    """Request payload for answer verification."""

    request_id: str
    prediction: str
    gold: str
    strict: bool = True
    timeout_seconds: int = 1
    max_prediction_length: int = 1000
    numeric_precision: int = 5
    float_rounding: int = 10


@dataclass
class VerifyResponse:
    """Response payload from verification."""

    request_id: str
    status: str  # correct, wrong, no_answer, unparsable, timeout, internal_error
    elapsed_ms: float
    retry_count: int = 0
    worker_id: Optional[int] = None
    worker_restarted: bool = False
    error_type: Optional[str] = None
    error_message: Optional[str] = None


def _parse_math_verify_expression(value: str):
    """Parse an expression with a boxed fallback for parser compatibility."""
    parsed = math_verify.parse(value)
    if parsed:
        return parsed

    # Use brace-depth tracking so nested expressions like \boxed{\frac{1}{2}}
    # aren't truncated by a non-greedy regex.
    start = value.find(r"\boxed{")
    if start != -1:
        i = start + len(r"\boxed{")
        depth = 1
        while i < len(value) and depth:
            if value[i] == "{":
                depth += 1
            elif value[i] == "}":
                depth -= 1
            i += 1
        inner = value[start + len(r"\boxed{") : i - 1]
        if inner:
            return math_verify.parse(inner)

    return parsed


class VerificationError(Exception):
    """Base exception for verification failures."""


class UnparsableException(VerificationError):
    """Raised when a math answer cannot be parsed for verification."""


class NoAnswerException(VerificationError):
    """Raised when a model output does not contain a boxed final answer."""


class EmptyBoxedException(VerificationError):
    """Raised when a boxed final answer is present but empty."""


def _extract_boxed_answer(text: str) -> str:
    """Extract the content from the innermost \\boxed{...} block.

    Handles nested braces correctly by tracking brace depth.
    """
    start_idx = text.rfind(r"\boxed{")
    if start_idx == -1:
        raise NoAnswerException()

    start_idx += len(r"\boxed{")
    depth = 1
    end_idx = start_idx

    while end_idx < len(text) and depth > 0:
        if text[end_idx] == "{":
            depth += 1
        elif text[end_idx] == "}":
            depth -= 1
        end_idx += 1

    if depth != 0:
        raise UnparsableException()

    answer = text[start_idx : end_idx - 1]
    if not answer.strip():
        raise EmptyBoxedException()

    return answer


def _verify_answer_worker(request: VerifyRequest) -> VerifyResponse:
    """Worker process function that performs the actual verification.

    This runs in an isolated process so that math_verify's signal-based timeout
    works correctly (signal handlers only work in the main thread).

    Args:
        request: VerifyRequest with prediction, gold, and config

    Returns:
        VerifyResponse with status and metadata
    """
    import time

    start_time = time.time()

    try:
        # Extract boxed answer
        boxed_prediction = _extract_boxed_answer(request.prediction)

        if len(boxed_prediction) > request.max_prediction_length:
            status = "unparsable"
        else:
            # Parse both expressions
            gold_parsed = _parse_math_verify_expression(request.gold)
            boxed_prediction_parsed = _parse_math_verify_expression(boxed_prediction)

            if not gold_parsed or not boxed_prediction_parsed:
                status = "unparsable"
            else:
                # Verify equivalence
                try:
                    equivalent = math_verify.verify(
                        gold_parsed,
                        boxed_prediction_parsed,
                        strict=request.strict,
                        timeout_seconds=request.timeout_seconds,
                        numeric_precision=request.numeric_precision,
                        float_rounding=request.float_rounding,
                    )
                    status = "correct" if equivalent else "wrong"
                except Exception as exc:
                    if "timeout" in str(exc).lower():
                        status = "timeout"
                    else:
                        status = "internal_error"
                        return VerifyResponse(
                            request_id=request.request_id,
                            status=status,
                            elapsed_ms=(time.time() - start_time) * 1000,
                            error_type=type(exc).__name__,
                            error_message=str(exc),
                        )

    except NoAnswerException:
        status = "no_answer"
    except (UnparsableException, EmptyBoxedException):
        status = "unparsable"
    except Exception as exc:
        status = "internal_error"
        return VerifyResponse(
            request_id=request.request_id,
            status=status,
            elapsed_ms=(time.time() - start_time) * 1000,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )

    elapsed_ms = (time.time() - start_time) * 1000
    return VerifyResponse(
        request_id=request.request_id,
        status=status,
        elapsed_ms=elapsed_ms,
    )


class MathVerifierService:
    """Async service that manages a pool of verification worker processes.

    This service:
    - Routes verification requests to available worker processes
    - Enforces timeouts and resource limits
    - Maps all outcomes into stable status values
    - Provides graceful startup/shutdown with cleanup

    The service is designed to handle high-throughput concurrent requests
    without blocking the main async loop (async step, async rollout, etc).

    Example:
        service = MathVerifierService(max_workers=4)
        await service.start()
        try:
            response = await service.verify_answer(
                prediction="\\\\boxed{4}",
                gold="4",
                strict=True,
                timeout_seconds=1,
            )
            print(f"status: {response.status}")
        finally:
            await service.stop()
    """

    def __init__(
        self,
        max_workers: int = 2,
        queue_size: int = 100,
        request_timeout_seconds: float = 5.0,
        max_retries: int = 1,
        strict: bool = True,
        numeric_precision: int = 5,
        float_rounding: int = 10,
    ):
        """Initialize the verifier service.

        Args:
            max_workers: Number of worker processes in the pool.
            queue_size: Maximum pending requests before backpressure.
            request_timeout_seconds: Client-side timeout for worker responses.
            max_retries: Max retry attempts for transient worker failures.
            strict: Default strict mode for math_verify.
            numeric_precision: Precision for numeric comparisons.
            float_rounding: Decimal places for float rounding.
        """
        self.max_workers = max(1, max_workers)
        self.queue_size = max(1, queue_size)
        self.request_timeout_seconds = max(0.001, float(request_timeout_seconds))
        self.max_retries = max(0, max_retries)
        self.strict = strict
        self.numeric_precision = numeric_precision
        self.float_rounding = float_rounding
        self._executor: ProcessPoolExecutor | None = None
        self._request_counter = 0
        self._admission_lock = asyncio.Lock()
        self._inflight_requests = 0
        self._restart_lock = asyncio.Lock()
        self._restart_count = 0
        self._metrics_lock = asyncio.Lock()
        self._requests_total = 0
        self._timeouts_total = 0
        self._errors_total = 0
        self._latency_total_ms = 0.0
        self._last_activity_monotonic = time.perf_counter()

    async def start(self) -> None:
        """Start the worker process pool.

        Safe to call multiple times (idempotent).
        """
        if self._executor is not None:
            logger.debug("MathVerifierService already started")
            return

        self._executor = ProcessPoolExecutor(
            max_workers=self.max_workers,
            mp_context=mp.get_context("spawn"),
        )
        logger.info("MathVerifierService started with %d workers", self.max_workers)

    async def stop(self) -> None:
        """Stop the worker process pool and clean up resources.

        Safe to call multiple times (idempotent).
        """
        if self._executor is None:
            logger.debug("MathVerifierService not running")
            return

        loop = asyncio.get_running_loop()
        shutdown_call = partial(self._executor.shutdown, wait=True, cancel_futures=True)
        await loop.run_in_executor(None, shutdown_call)
        self._executor = None
        logger.info("MathVerifierService stopped")

    def close_sync(self) -> None:
        """Synchronously cancel and release the worker pool.

        Called from synchronous close() hooks where no event loop is available.
        Uses cancel_futures=True so in-flight workers are signalled immediately;
        wait=False avoids blocking the caller.
        """
        if self._executor is None:
            return
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._executor = None
        logger.info("MathVerifierService pool released (sync close)")

    async def _restart_pool(self) -> None:
        """Restart the worker pool after a fatal worker/executor failure."""
        async with self._restart_lock:
            await self.stop()
            await self.start()
            self._restart_count += 1

    @staticmethod
    def _requires_restart(response: VerifyResponse) -> bool:
        """Return True when response indicates executor/worker death."""
        if response.status != "internal_error":
            return False

        fatal_error_types = {
            "BrokenProcessPool",
            "EOFError",
            "ConnectionResetError",
            "ConnectionAbortedError",
        }
        if response.error_type in fatal_error_types:
            return True

        msg = (response.error_message or "").lower()
        fatal_fragments = [
            "broken process pool",
            "process terminated",
            "worker process",
            "connection reset",
            "connection aborted",
            "child process",
        ]
        return any(fragment in msg for fragment in fatal_fragments)

    async def health_probe(self) -> dict[str, str | bool | int | float]:
        """Return lightweight health status for long-lived training runs."""
        executor_running = self._executor is not None
        status = "healthy" if executor_running else "stopped"
        heartbeat_lag_ms = max(
            0.0,
            (time.perf_counter() - self._last_activity_monotonic) * 1000,
        )
        return {
            "status": status,
            "executor_running": executor_running,
            "inflight_requests": self._inflight_requests,
            "queue_size": self.queue_size,
            "max_workers": self.max_workers,
            "restart_count": self._restart_count,
            "heartbeat_lag_ms": heartbeat_lag_ms,
        }

    async def metrics_snapshot(self) -> dict[str, float | int]:
        """Return rollout-oriented verifier metrics snapshot."""
        async with self._metrics_lock:
            requests_count = self._requests_total
            latency_avg_ms = (
                self._latency_total_ms / requests_count if requests_count > 0 else 0.0
            )
            heartbeat_lag_ms = max(
                0.0,
                (time.perf_counter() - self._last_activity_monotonic) * 1000,
            )
            return {
                "verifier/requests/count": requests_count,
                "verifier/requests/latency_ms": latency_avg_ms,
                "verifier/requests/timeout_count": self._timeouts_total,
                "verifier/requests/error_count": self._errors_total,
                "verifier/workers/restart_count": self._restart_count,
                "verifier/queue/depth": self._inflight_requests,
                "verifier/workers/heartbeat_lag_ms": heartbeat_lag_ms,
            }

    async def _record_result_metrics(self, response: VerifyResponse) -> None:
        """Update aggregate counters for rollout/staging observability."""
        async with self._metrics_lock:
            self._requests_total += 1
            self._latency_total_ms += max(0.0, float(response.elapsed_ms))
            if response.status == "timeout":
                self._timeouts_total += 1
            if response.status == "internal_error":
                self._errors_total += 1
            self._last_activity_monotonic = time.perf_counter()

    async def _try_admit_request(self) -> bool:
        """Try to admit a request without blocking when saturated."""
        async with self._admission_lock:
            if self._inflight_requests >= self.queue_size:
                return False
            self._inflight_requests += 1
            return True

    async def _release_request_slot(self) -> None:
        """Release a previously admitted request slot."""
        async with self._admission_lock:
            self._inflight_requests = max(0, self._inflight_requests - 1)

    @staticmethod
    def _is_retryable_response(response: VerifyResponse) -> bool:
        """Return True when a response represents a transient infra failure."""
        if response.status == "timeout":
            return True

        if response.status != "internal_error":
            return False

        retryable_error_types = {
            "ClientTimeout",
            "BrokenProcessPool",
            "CancelledError",
            "TimeoutError",
            "EOFError",
            "ConnectionResetError",
            "ConnectionAbortedError",
        }
        if response.error_type in retryable_error_types:
            return True

        error_message = (response.error_message or "").lower()
        retryable_fragments = [
            "broken process pool",
            "executor",
            "cancelled",
            "worker",
            "connection reset",
            "connection aborted",
        ]
        return any(fragment in error_message for fragment in retryable_fragments)

    async def _run_request_once(
        self,
        request: VerifyRequest,
    ) -> VerifyResponse:
        """Execute one verifier attempt without retries."""
        loop = asyncio.get_running_loop()
        request_id = request.request_id

        try:
            future = loop.run_in_executor(
                self._executor,
                _verify_answer_worker,
                request,
            )
            return await asyncio.wait_for(
                future,
                timeout=self.request_timeout_seconds,
            )

        except asyncio.TimeoutError:
            logger.warning(
                "Verification request %s timed out after %.3fs",
                request_id,
                self.request_timeout_seconds,
            )
            return VerifyResponse(
                request_id=request_id,
                status="timeout",
                elapsed_ms=self.request_timeout_seconds * 1000,
                error_type="ClientTimeout",
                error_message=f"Request timed out after {self.request_timeout_seconds}s",
            )

        except Exception as exc:
            logger.exception(
                "Verification request %s failed with exception", request_id
            )
            return VerifyResponse(
                request_id=request_id,
                status="internal_error",
                elapsed_ms=0.0,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )

    async def verify_answer(
        self,
        prediction: str,
        gold: str,
        strict: Optional[bool] = None,
        timeout_seconds: Optional[int] = None,
        max_prediction_length: int = 1000,
        numeric_precision: Optional[int] = None,
        float_rounding: Optional[int] = None,
    ) -> VerifyResponse:
        """Verify an answer prediction against a gold reference.

        This is the main async entry point for answer verification. It submits
        the request to a worker process and awaits the result with a client-side
        timeout.

        Args:
            prediction: The model's predicted answer (may contain \\boxed{...}).
            gold: The reference answer (may contain \\boxed{...}).
            strict: Whether to use strict equivalence checking. Defaults to config.
            timeout_seconds: Worker timeout for math_verify.verify(). Defaults to 1.
            max_prediction_length: Max length of boxed answer.

        Returns:
            VerifyResponse with status (correct, wrong, no_answer, unparsable, timeout, internal_error)
        """
        if self._executor is None:
            await self.start()

        admitted = await self._try_admit_request()
        if not admitted:
            rejection = VerifyResponse(
                request_id=f"rejected-{self._request_counter + 1}",
                status="internal_error",
                elapsed_ms=0.0,
                error_type="QueueFull",
                error_message=(
                    f"Verifier queue saturated: in_flight={self._inflight_requests}, "
                    f"queue_size={self.queue_size}"
                ),
            )
            await self._record_result_metrics(rejection)
            return rejection

        self._request_counter += 1
        request_id = f"req-{self._request_counter}"
        started_at = time.perf_counter()

        request = VerifyRequest(
            request_id=request_id,
            prediction=prediction,
            gold=gold,
            strict=strict if strict is not None else self.strict,
            timeout_seconds=max(1, int(timeout_seconds or 1)),
            max_prediction_length=max_prediction_length,
            numeric_precision=(
                numeric_precision
                if numeric_precision is not None
                else self.numeric_precision
            ),
            float_rounding=(
                float_rounding if float_rounding is not None else self.float_rounding
            ),
        )

        retry_count = 0
        worker_restarted = False
        try:
            while True:
                response = await self._run_request_once(request)

                if retry_count < self.max_retries and self._is_retryable_response(
                    response
                ):
                    if self._requires_restart(response):
                        await self._restart_pool()
                        worker_restarted = True
                    retry_count += 1
                    continue

                response.retry_count = retry_count
                response.worker_restarted = worker_restarted
                response.elapsed_ms = (time.perf_counter() - started_at) * 1000
                await self._record_result_metrics(response)
                return response
        finally:
            await self._release_request_slot()
