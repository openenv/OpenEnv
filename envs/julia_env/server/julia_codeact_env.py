# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Julia Code Action Environment.

This module provides a server-side environment implementation for executing
Julia code actions using JuliaExecutor.
"""

import itertools
import logging
import re
import time
import uuid

# Support both in-repo and standalone imports
try:
    # In-repo imports (when running from OpenEnv repository)
    from openenv.core.env_server.interfaces import Action, Environment, Observation

    from ..models import JuliaAction, JuliaObservation, JuliaState
    from .julia_executor import JuliaExecutor
    from .julia_transforms import create_safe_julia_transform
except ImportError:
    from models import JuliaAction, JuliaObservation, JuliaState

    # Standalone imports (when environment is standalone)
    from openenv.core.env_server.interfaces import Action, Environment, Observation
    from server.julia_executor import JuliaExecutor
    from server.julia_transforms import create_safe_julia_transform

# Get logger for this module (inherits from julia_env logger)
logger = logging.getLogger("julia_env.codeact")

# Thread-safe request counter for tracking
_request_counter = itertools.count(1)


def _detect_infinite_loop(code: str) -> tuple[bool, str]:
    """
    Detect potential infinite loops in Julia code.

    This function scans for `while true` loops without break/return/error statements.

    Args:
        code: Julia code string to analyze

    Returns:
        Tuple of (has_infinite_loop: bool, reason: str)
    """
    # Remove comments and strings to avoid false positives
    # Remove single-line comments
    code_without_comments = re.sub(r"#.*", "", code)
    # Remove multi-line strings (triple quotes)
    code_without_comments = re.sub(
        r'""".*?"""', "", code_without_comments, flags=re.DOTALL
    )
    # Remove single-line strings
    code_without_comments = re.sub(r'"[^"]*"', "", code_without_comments)

    # Find all while true blocks
    while_true_pattern = r"\bwhile\s+true\b"
    while_true_matches = list(
        re.finditer(while_true_pattern, code_without_comments, re.IGNORECASE)
    )

    if not while_true_matches:
        return False, ""

    # For each while true, check if there's a break/return/error in the same block
    for match in while_true_matches:
        start_pos = match.end()

        # Find the end of this while block by counting 'while'/'end' pairs
        # Simplified heuristic: look for break/return/error before the corresponding 'end'
        remaining_code = code_without_comments[start_pos:]

        # Extract potential loop body (up to next 'end' keyword)
        # This is a simplified check - doesn't perfectly handle nested blocks
        end_match = re.search(r"\bend\b", remaining_code)
        if end_match:
            loop_body = remaining_code[: end_match.start()]
        else:
            loop_body = remaining_code

        # Check for loop exit mechanisms in this block
        has_break = re.search(r"\bbreak\b", loop_body) is not None
        has_return = re.search(r"\breturn\b", loop_body) is not None
        has_error = re.search(r"\berror\(", loop_body) is not None
        has_throw = re.search(r"\bthrow\(", loop_body) is not None
        has_exit = re.search(r"\bexit\(", loop_body) is not None

        if not (has_break or has_return or has_error or has_throw or has_exit):
            loop_preview = loop_body[:100].strip()
            return (
                True,
                f"Infinite loop detected: 'while true' without break/return/error/throw. Preview: {loop_preview}",
            )

    return False, ""


class JuliaCodeActEnv(Environment):
    """
    Julia Code Action Environment for executing code and tracking state.

    This environment executes Julia code submitted as JuliaAction during step,
    maintains the last exit code in its state, and returns results wrapped
    in JuliaObservation.

    Example:
        >>> env = JuliaCodeActEnv()
        >>> obs = env.reset()
        >>> action = JuliaAction(core_code='println("Hello, Julia!")', test_code='')
        >>> obs = env.step(action)
        >>> print(obs.stdout)  # "Hello, Julia!\\n"
        >>> print(obs.exit_code)  # 0
        >>> print(env.state.last_exit_code)  # 0
    """

    # Allow concurrent sessions - each session has its own isolated state
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, use_process_pool: bool = True):
        """
        Initialize the Julia Code Act Environment.

        Args:
            use_process_pool: Use persistent Julia process pool for better performance
                            and to avoid Juliaup lock contention (default: True)
        """
        self._executor = JuliaExecutor(use_process_pool=use_process_pool)
        self._state = JuliaState()
        self.transform = create_safe_julia_transform()

    def reset(self, **kwargs) -> Observation:
        """
        Reset environment for a fresh Julia execution session.
        Returns an empty JuliaObservation with exit_code=0.

        Note: Executor is reused to leverage process pool.
        """
        self._state = JuliaState(episode_id=str(uuid.uuid4()), step_count=0)
        self._state.last_exit_code = 0
        self._state.last_code_compiles = True
        # Don't recreate executor - reuse it to leverage process pool

        observation = JuliaObservation(
            stdout="",
            stderr="",
            exit_code=0,
            reward=0.0,
            metadata={"core_code": "", "test_code": ""},
            tests_passed=0,
            tests_failed=0,
            code_compiles=True,
        )

        observation = self._apply_transform(observation)
        return observation

    def step(self, action: Action, **kwargs) -> Observation:
        """
        Execute Julia code and return the result as JuliaObservation.

        Optimized single-pass execution:
        - Runs core_code + test_code together
        - Infers compilation status from combined execution
        - 2x faster than double execution

        Args:
            action: JuliaAction with core_code and optional test_code
            **kwargs: Optional parameters including:
                - timeout: Execution timeout in seconds (default: 120)
        """
        request_id = next(_request_counter)

        if not isinstance(action, JuliaAction):
            logger.error(f"[REQ-{request_id}] Invalid action type: {type(action)}")
            raise ValueError(f"Expected JuliaAction, got {type(action)}")

        # Get timeout from kwargs (default handled by executor)
        timeout = kwargs.get("timeout")

        # Log request details
        code_preview = (
            action.core_code[:200] + "..."
            if len(action.core_code) > 200
            else action.core_code
        )
        logger.info(f"[REQ-{request_id}] === NEW EXECUTION REQUEST ===")
        logger.info(
            f"[REQ-{request_id}] Session: {self._state.episode_id}, Step: {self._state.step_count}"
        )
        logger.info(
            f"[REQ-{request_id}] Code length: {len(action.core_code)} chars, Test length: {len(action.test_code or '')} chars"
        )
        logger.debug(f"[REQ-{request_id}] Code preview: {code_preview}")
        logger.info(
            f"[REQ-{request_id}] Timeout: {timeout}s"
            if timeout
            else f"[REQ-{request_id}] Timeout: default"
        )

        start_time = time.time()

        # Single execution: Run core_code + test_code together (if test_code provided)
        if action.test_code:
            combined_code = action.core_code + "\n\n" + action.test_code
        else:
            combined_code = action.core_code

        # Pre-execution check: detect infinite loops to avoid timeout
        has_infinite_loop, loop_reason = _detect_infinite_loop(action.core_code)
        if has_infinite_loop:
            logger.warning(f"[REQ-{request_id}] INFINITE LOOP DETECTED: {loop_reason}")

            # Update environment state
            self._state.step_count += 1
            self._state.last_exit_code = 1
            self._state.last_code_compiles = True  # Code compiles but has infinite loop
            self._state.total_tests_passed = 0
            self._state.total_tests_failed = 0

            # Build observation with penalty
            observation = JuliaObservation(
                stdout="",
                stderr=f"Infinite loop detected (pre-execution check): {loop_reason}",
                exit_code=1,
                reward=-1.0,  # Penalize infinite loops
                metadata={
                    "core_code": action.core_code,
                    "test_code": action.test_code or "",
                    "infinite_loop_detected": True,
                    "infinite_loop_reason": loop_reason,
                },
                tests_passed=0,
                tests_failed=0,
                code_compiles=True,  # Code would compile, but not run
            )

            logger.info(
                f"[REQ-{request_id}] RESULT: infinite_loop=True, "
                f"tests_passed=0, tests_failed=0, reward=-1.00"
            )

            observation = self._apply_transform(observation)
            return observation

        try:
            full_result = self._executor.run(combined_code, timeout=timeout)
            execution_time = time.time() - start_time

            logger.info(
                f"[REQ-{request_id}] Execution completed in {execution_time:.2f}s, exit_code={full_result.exit_code}"
            )

            # Log stderr if present (often contains errors or test output)
            if full_result.stderr:
                stderr_preview = (
                    full_result.stderr[:500] + "..."
                    if len(full_result.stderr) > 500
                    else full_result.stderr
                )
                logger.debug(f"[REQ-{request_id}] Stderr: {stderr_preview}")

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(
                f"[REQ-{request_id}] EXECUTION FAILED after {execution_time:.2f}s: {e}"
            )
            raise

        # Parse test results from execution output
        tests_passed, tests_failed = self._parse_test_results(
            full_result.stdout, full_result.stderr
        )

        # Infer compilation status from execution
        # If tests ran, code compiled successfully
        # If exit_code != 0 and no tests ran, code didn't compile
        code_compiles = (
            full_result.exit_code == 0  # Clean execution
            or tests_passed > 0  # Some tests passed (code must have compiled)
            or tests_failed > 0  # Some tests failed (code compiled but tests failed)
        )

        # If no tests detected and non-zero exit, check for compilation errors
        if not code_compiles and tests_passed == 0 and tests_failed == 0:
            # Check stderr for compilation errors
            stderr_lower = full_result.stderr.lower()
            if any(
                err in stderr_lower
                for err in ["error", "syntax", "undefined", "loadError"]
            ):
                code_compiles = False
            else:
                # If no clear compilation error, assume it compiled
                code_compiles = True

        # Calculate reward based on compilation and test results
        reward = self._calculate_reward(code_compiles, tests_passed, tests_failed)

        # Log final results
        logger.info(
            f"[REQ-{request_id}] RESULT: compiles={code_compiles}, "
            f"tests_passed={tests_passed}, tests_failed={tests_failed}, reward={reward:.2f}"
        )

        # Update environment state
        self._state.step_count += 1
        self._state.last_exit_code = full_result.exit_code
        self._state.last_code_compiles = code_compiles
        self._state.total_tests_passed = tests_passed
        self._state.total_tests_failed = tests_failed

        # Build observation
        observation = JuliaObservation(
            stdout=full_result.stdout,
            stderr=full_result.stderr,
            exit_code=full_result.exit_code,
            reward=reward,
            metadata={
                "core_code": action.core_code,
                "test_code": action.test_code or "",
            },
            tests_passed=tests_passed,
            tests_failed=tests_failed,
            code_compiles=code_compiles,
        )

        # Apply safety and quality transforms
        observation = self._apply_transform(observation)

        return observation

    def _parse_test_results(self, stdout: str, stderr: str) -> tuple[int, int]:
        """
        Parse Julia test output to count passed/failed tests.

        Julia's Test module outputs results like:
        "Test Summary:      | Pass  Fail  Total  Time"
        "Add function Tests |    1     1      2  1.5s"

        Also checks error messages:
        "Some tests did not pass: 1 passed, 1 failed, 0 errored, 0 broken."

        Args:
            stdout: Standard output from Julia execution
            stderr: Standard error from Julia execution

        Returns:
            Tuple of (tests_passed, tests_failed)
        """
        # Combine stdout and stderr for analysis
        passed = 0
        failed = 0
        output = stdout + "\n" + stderr

        # Method 1: Look for "Some tests did not pass" error message
        # Pattern: "Some tests did not pass: X passed, Y failed, Z errored, W broken."
        error_pattern = r"Some tests did not pass:\s*(\d+)\s+passed,\s*(\d+)\s+failed,\s*(\d+)\s+errored"
        match = re.search(error_pattern, output)

        if match:
            passed = int(match.group(1))
            failed = int(match.group(2))
            errored = int(match.group(3))
            return passed, failed + errored  # Treat errors as failures

        # Method 2: Look for Test Summary table
        # Multiple possible formats:
        # All pass:     "Test Summary: | Pass  Total  Time"
        #               "My Tests     |    3      3  0.5s"
        # Some fail:    "Test Summary: | Pass  Fail  Total  Time"
        #               "My Tests     |    2     1      3  0.5s"
        # All error:    "Test Summary: | Error  Total  Time"
        #               "My Tests     |     3      3  0.9s"
        # Mixed:        "Test Summary: | Pass  Fail  Error  Total  Time"
        #               "My Tests     |    1     1      1      3  0.5s"
        summary_lines = output.split("\n")
        for i, line in enumerate(summary_lines):
            if "Test Summary:" in line and i + 1 < len(summary_lines):
                header_line = line
                next_line = summary_lines[i + 1]

                # Determine which columns are present
                has_pass = "Pass" in header_line
                has_fail = "Fail" in header_line
                has_error = "Error" in header_line

                # Extract all numbers from the line
                all_numbers = re.findall(r"\d+", next_line)
                if not all_numbers:
                    continue

                # Last number is always Total, second to last is Time (skip it)
                # Extract based on which columns exist
                if has_pass and has_fail and has_error:
                    # Pass  Fail  Error  Total  Time
                    if len(all_numbers) >= 5:
                        passed = int(all_numbers[0])
                        failed = int(all_numbers[1]) + int(
                            all_numbers[2]
                        )  # Fail + Error
                        return passed, failed
                elif has_pass and has_fail:
                    # Pass  Fail  Total  Time
                    if len(all_numbers) >= 4:
                        passed = int(all_numbers[0])
                        failed = int(all_numbers[1])
                        return passed, failed
                elif has_pass and has_error:
                    # Pass  Error  Total  Time
                    if len(all_numbers) >= 4:
                        passed = int(all_numbers[0])
                        failed = int(all_numbers[1])  # Treat errors as failures
                        return passed, failed
                elif has_fail and has_error:
                    # Fail  Error  Total  Time (no passes)
                    if len(all_numbers) >= 4:
                        passed = 0
                        failed = int(all_numbers[0]) + int(all_numbers[1])
                        return passed, failed
                elif has_pass:
                    # Pass  Total  Time (no failures/errors)
                    if len(all_numbers) >= 3:
                        passed = int(all_numbers[0])
                        failed = 0
                        return passed, failed
                elif has_error:
                    # Error  Total  Time (all errors, no passes)
                    if len(all_numbers) >= 3:
                        passed = 0
                        failed = int(all_numbers[0])  # Treat all errors as failures
                        return passed, failed
                elif has_fail:
                    # Fail  Total  Time (all failures, no passes)
                    if len(all_numbers) >= 3:
                        passed = 0
                        failed = int(all_numbers[0])
                        return passed, failed

        return passed, failed

    def _calculate_reward(
        self, code_compiles: bool, tests_passed: int, tests_failed: int
    ) -> float:
        """
        Normalized percentage-based reward for Julia GRPO.
        Returns rewards in [-1, 1.5] range for comparability across problems.
        """
        if not code_compiles:
            return -1.0

        total_tests = tests_passed + tests_failed
        if total_tests == 0:
            return 0.0  # No signal when no tests run

        pass_rate = tests_passed / total_tests

        # Scaled 0-1 with bonus for perfection
        if pass_rate == 1.0:
            return 1.5  # Bonus for passing all tests
        return pass_rate

    def _apply_transform(self, observation: JuliaObservation) -> JuliaObservation:
        """Apply safety and quality transforms to observation."""
        if self.transform:
            observation = self.transform(observation)
        return observation

    @property
    def state(self) -> JuliaState:
        """Return current environment state."""
        return self._state
