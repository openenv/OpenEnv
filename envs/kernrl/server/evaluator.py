"""
Local GPU Evaluator for KernelBench

Runs kernels on local GPU with comprehensive profiling:
- Compilation check with error capture
- Correctness check with atol/rtol statistics
- Benchmark with warmup and timing statistics
- NSight Systems profiling (system-level)
- NSight Compute profiling (kernel-level)
- Compute Sanitizer (correctness bugs)
- torch.profiler (PyTorch-level)
- Assembly analysis (PTX/SASS)
- Roofline metrics (arithmetic intensity, theoretical vs achieved)

All feedback is curated to be actionable for LLM agents.
"""

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .profiler import (
    AssemblyAnalysis,
    GPUProfiler,
    NcuProfile,
    NsysProfile,
    RooflineMetrics,
    SanitizerResult,
    TorchProfile,
)


@dataclass
class CompilationResult:
    """Result of compilation check."""

    success: bool
    error: Optional[str] = None
    warnings: list[str] = field(default_factory=list)


@dataclass
class CorrectnessResult:
    """Result of correctness check."""

    correct: bool
    max_diff: float = 0.0
    mean_diff: float = 0.0
    median_diff: float = 0.0
    std_diff: float = 0.0
    atol: float = 0.05
    rtol: float = 0.02
    tolerance: float = 0.0  # atol + rtol * max_ref
    num_elements: int = 0
    num_mismatched: int = 0
    mismatch_percentage: float = 0.0
    error: Optional[str] = None


@dataclass
class BenchmarkResult:
    """Result of benchmark."""

    baseline_time_us: float = 0.0
    solution_time_us: float = 0.0
    speedup: float = 0.0
    baseline_std_us: float = 0.0
    solution_std_us: float = 0.0
    warmup_runs: int = 10
    benchmark_runs: int = 100
    error: Optional[str] = None


@dataclass
class EvalResult:
    """Complete evaluation result with all profiling data."""

    # Step info
    step: int = 0
    problem_id: str = ""

    # Compilation
    compilation: CompilationResult = field(
        default_factory=lambda: CompilationResult(success=False)
    )

    # Correctness (only if compiled)
    correctness: Optional[CorrectnessResult] = None

    # Benchmark (only if correct)
    benchmark: Optional[BenchmarkResult] = None

    # Profiling - all enabled by default
    nsys: Optional[NsysProfile] = None
    ncu: Optional[NcuProfile] = None
    sanitizer: Optional[SanitizerResult] = None
    torch_profile: Optional[TorchProfile] = None
    assembly: Optional[AssemblyAnalysis] = None
    roofline: Optional[RooflineMetrics] = None

    # Overall
    reward: float = 0.0

    def to_agent_feedback(self) -> str:
        """Format as actionable feedback string for the agent."""
        lines = [f"{'=' * 60}", f"EVALUATION RESULT - Step {self.step}", f"{'=' * 60}"]

        # Compilation
        lines.append("\n## COMPILATION")
        if self.compilation.success:
            lines.append("Status: PASS")
            if self.compilation.warnings:
                lines.append(f"Warnings ({len(self.compilation.warnings)}):")
                for w in self.compilation.warnings[:2]:
                    lines.append(f"  - {w[:100]}")
        else:
            lines.append("Status: FAIL")
            lines.append(f"Error:\n{self.compilation.error}")
            lines.append(f"\n{'=' * 60}")
            lines.append(f"REWARD: {self.reward:.3f}")
            lines.append(f"{'=' * 60}")
            return "\n".join(lines)

        # Compute Sanitizer (early - shows correctness bugs)
        if self.sanitizer and self.sanitizer.success:
            lines.append("")
            lines.append(self.sanitizer.to_agent_summary())

        # Correctness
        lines.append("\n## CORRECTNESS")
        if self.correctness:
            c = self.correctness
            lines.append(f"Status: {'PASS' if c.correct else 'FAIL'}")
            lines.append(f"  max_diff:    {c.max_diff:.6e}")
            lines.append(f"  mean_diff:   {c.mean_diff:.6e}")
            lines.append(
                f"  tolerance:   {c.tolerance:.6e} (atol={c.atol}, rtol={c.rtol})"
            )
            lines.append(
                f"  mismatched:  {c.num_mismatched:,}/{c.num_elements:,} ({c.mismatch_percentage:.2f}%)"
            )
            if c.error:
                lines.append(f"  Error: {c.error[:200]}")

        # Benchmark
        lines.append("\n## BENCHMARK")
        if self.benchmark:
            b = self.benchmark
            lines.append(
                f"  Baseline: {b.baseline_time_us:>8.2f} +/- {b.baseline_std_us:.2f} us"
            )
            lines.append(
                f"  Solution: {b.solution_time_us:>8.2f} +/- {b.solution_std_us:.2f} us"
            )
            lines.append(
                f"  Speedup:  {b.speedup:.2f}x {'(FASTER)' if b.speedup > 1 else '(SLOWER)'}"
            )
            if b.error:
                lines.append(f"  Error: {b.error[:200]}")
        else:
            lines.append("  Skipped (correctness check failed)")

        # NSight Systems
        if self.nsys and self.nsys.success:
            lines.append("")
            lines.append(self.nsys.to_agent_summary())

        # NSight Compute
        if self.ncu and self.ncu.success:
            lines.append("")
            lines.append(self.ncu.to_agent_summary())

        # Roofline Analysis
        if self.roofline and self.roofline.success:
            lines.append("")
            lines.append(self.roofline.to_agent_summary())

        # torch.profiler
        if self.torch_profile and self.torch_profile.success:
            lines.append("")
            lines.append(self.torch_profile.to_agent_summary())

        # Assembly Analysis
        if self.assembly and self.assembly.success:
            lines.append("")
            lines.append(self.assembly.to_agent_summary())

        # Final reward
        lines.append(f"\n{'=' * 60}")
        lines.append(f"REWARD: {self.reward:.3f}")
        lines.append(f"{'=' * 60}")

        return "\n".join(lines)


class LocalGPUEvaluator:
    """
    Evaluates kernel submissions on local GPU with comprehensive profiling.

    Features:
    - Compilation check with detailed error messages
    - Correctness check with statistical breakdown
    - Benchmark with proper warmup and timing
    - NSight Systems profiling (system-level)
    - NSight Compute profiling (kernel-level)
    - Compute Sanitizer (memory/sync errors)
    - torch.profiler (PyTorch operators)
    - Assembly analysis (PTX/SASS)
    - Roofline metrics (arithmetic intensity)

    All output is formatted to be actionable for LLM agents.
    """

    def __init__(
        self,
        device: str = "cuda:0",
        atol: float = 0.05,
        rtol: float = 0.02,
        warmup_runs: int = 10,
        benchmark_runs: int = 100,
        # Profiling toggles - all enabled by default
        enable_nsys: bool = True,
        enable_ncu: bool = False,  # NCU is slow, opt-in
        enable_sanitizer: bool = True,
        enable_torch_profiler: bool = True,
        enable_assembly: bool = True,
        enable_roofline: bool = True,
        timeout: int = 60,
    ):
        self.device = device
        self.atol = atol
        self.rtol = rtol
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs
        self.timeout = timeout

        # Create profiler with all tools
        self.profiler = GPUProfiler(
            enable_nsys=enable_nsys,
            enable_ncu=enable_ncu,
            enable_sanitizer=enable_sanitizer,
            enable_torch_profiler=enable_torch_profiler,
            enable_assembly=enable_assembly,
            enable_roofline=enable_roofline,
            nsys_timeout=timeout,
            ncu_timeout=timeout * 2,
            sanitizer_timeout=timeout,
        )

    def evaluate(
        self,
        solution_code: str,
        reference_code: str,
        problem_id: str = "",
        step: int = 0,
    ) -> EvalResult:
        """
        Fully evaluate a solution with all profiling.

        Returns EvalResult with all profiling data.
        """
        result = EvalResult(step=step, problem_id=problem_id)

        # Create temp directory for all files
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Write files
            solution_path = tmpdir / "solution.py"
            reference_path = tmpdir / "reference.py"

            solution_path.write_text(solution_code)
            reference_path.write_text(reference_code)

            # Step 1: Compilation check
            result.compilation = self._check_compilation(solution_path)
            if not result.compilation.success:
                return result

            # Step 2: Compute Sanitizer (early - catches memory bugs)
            if self.profiler.enable_sanitizer:
                runner_path = self._create_runner_script(
                    solution_path, reference_path, tmpdir
                )
                result.sanitizer = self.profiler.run_sanitizer(runner_path, tmpdir)

            # Step 3: Correctness check
            result.correctness = self._check_correctness(
                solution_path, reference_path, tmpdir
            )

            # Step 4: Benchmark (only if correct)
            if result.correctness and result.correctness.correct:
                result.benchmark = self._run_benchmark(
                    solution_path, reference_path, tmpdir
                )

            # Step 5: All profiling (if compiled)
            if result.compilation.success:
                runner_path = self._create_runner_script(
                    solution_path, reference_path, tmpdir
                )

                # NSight Systems
                if self.profiler.enable_nsys:
                    result.nsys = self.profiler.run_nsys(runner_path, tmpdir)

                # NSight Compute
                if self.profiler.enable_ncu:
                    result.ncu = self.profiler.run_ncu(runner_path, tmpdir)

                # torch.profiler
                if self.profiler.enable_torch_profiler:
                    result.torch_profile = self.profiler.run_torch_profiler(
                        solution_path, tmpdir
                    )

                # Assembly analysis
                if self.profiler.enable_assembly:
                    result.assembly = self.profiler.run_assembly_analysis(
                        solution_path, tmpdir
                    )

                # Roofline metrics (needs NCU data)
                if self.profiler.enable_roofline and result.ncu and result.ncu.success:
                    benchmark_time = (
                        result.benchmark.solution_time_us
                        if result.benchmark
                        else 1000.0
                    )
                    result.roofline = self.profiler.compute_roofline(
                        result.ncu, benchmark_time
                    )

        # Calculate reward
        result.reward = self._compute_reward(result)

        return result

    def _create_runner_script(
        self,
        solution_path: Path,
        reference_path: Path,
        tmpdir: Path,
    ) -> Path:
        """Create a runner script for profiling."""
        runner_path = tmpdir / "profile_runner.py"
        runner_path.write_text(f'''
import torch
import importlib.util

def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

ref_mod = load_module("{reference_path}", "reference")
sol_mod = load_module("{solution_path}", "solution")

device = "{self.device}"

if hasattr(ref_mod, "get_init_inputs"):
    init_inputs = ref_mod.get_init_inputs()
else:
    init_inputs = []

model = sol_mod.Model(*init_inputs).to(device).eval()

if hasattr(ref_mod, "get_inputs"):
    inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in ref_mod.get_inputs()]
else:
    inputs = [torch.randn(16, 1024, device=device)]

# Warmup
with torch.no_grad():
    for _ in range(5):
        model(*inputs)

torch.cuda.synchronize()

# Profile this
with torch.no_grad():
    for _ in range(10):
        model(*inputs)

torch.cuda.synchronize()
''')
        return runner_path

    def _check_compilation(self, solution_path: Path) -> CompilationResult:
        """Check if solution compiles and has required interface."""
        check_script = f'''
import sys
import warnings
captured_warnings = []

def warn_handler(message, category, filename, lineno, file=None, line=None):
    captured_warnings.append(str(message))

old_showwarning = warnings.showwarning
warnings.showwarning = warn_handler

try:
    import torch
    import importlib.util
    spec = importlib.util.spec_from_file_location("solution", "{solution_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    assert hasattr(mod, "Model"), "Missing Model class"

    # Try to instantiate
    model = mod.Model()
    assert hasattr(model, "forward"), "Model missing forward method"

    print("OK")
    for w in captured_warnings:
        print(f"WARNING: {{w}}")
except Exception as e:
    print(f"ERROR: {{e}}")
    import traceback
    traceback.print_exc()
'''
        try:
            proc = subprocess.run(
                [sys.executable, "-c", check_script],
                capture_output=True,
                text=True,
                timeout=30,
            )

            output = proc.stdout + proc.stderr

            if "OK" in proc.stdout:
                warnings = [
                    line.replace("WARNING: ", "")
                    for line in proc.stdout.split("\n")
                    if line.startswith("WARNING:")
                ]
                return CompilationResult(success=True, warnings=warnings)
            else:
                return CompilationResult(success=False, error=output[:2000])

        except subprocess.TimeoutExpired:
            return CompilationResult(success=False, error="Compilation timeout (30s)")
        except Exception as e:
            return CompilationResult(success=False, error=str(e))

    def _check_correctness(
        self,
        solution_path: Path,
        reference_path: Path,
        tmpdir: Path,
    ) -> CorrectnessResult:
        """Run correctness check comparing solution to reference."""

        correctness_script = f'''
import sys
import json
import torch
import importlib.util

def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

try:
    ref_mod = load_module("{reference_path}", "reference")
    sol_mod = load_module("{solution_path}", "solution")

    device = "{self.device}"

    # Get inputs from reference module
    if hasattr(ref_mod, "get_init_inputs"):
        init_inputs = ref_mod.get_init_inputs()
    else:
        init_inputs = []

    ref_model = ref_mod.Model(*init_inputs).to(device).eval()
    sol_model = sol_mod.Model(*init_inputs).to(device).eval()

    if hasattr(ref_mod, "get_inputs"):
        inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in ref_mod.get_inputs()]
    else:
        inputs = [torch.randn(16, 1024, device=device)]

    with torch.no_grad():
        ref_out = ref_model(*inputs)
        sol_out = sol_model(*inputs)

    # Convert to float for comparison
    ref_f = ref_out.float() if isinstance(ref_out, torch.Tensor) else torch.tensor(ref_out).float()
    sol_f = sol_out.float() if isinstance(sol_out, torch.Tensor) else torch.tensor(sol_out).float()

    # Compute statistics
    diff = (ref_f - sol_f).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    median_diff = diff.median().item()
    std_diff = diff.std().item()

    # Tolerance calculation
    atol = {self.atol}
    rtol = {self.rtol}
    max_ref = ref_f.abs().max().item()
    tolerance = atol + rtol * max_ref

    # Count mismatches
    threshold = atol + rtol * ref_f.abs()
    mismatched = (diff > threshold).sum().item()
    total = diff.numel()

    correct = max_diff < tolerance

    result = {{
        "correct": correct,
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "median_diff": median_diff,
        "std_diff": std_diff,
        "atol": atol,
        "rtol": rtol,
        "tolerance": tolerance,
        "num_elements": total,
        "num_mismatched": mismatched,
        "mismatch_percentage": 100.0 * mismatched / total if total > 0 else 0.0,
    }}

    print(json.dumps(result))

except Exception as e:
    import traceback
    print(json.dumps({{"error": str(e), "traceback": traceback.format_exc()}}))
'''

        try:
            proc = subprocess.run(
                [sys.executable, "-c", correctness_script],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )

            # Parse JSON output
            try:
                data = json.loads(proc.stdout.strip().split("\n")[-1])
            except:
                return CorrectnessResult(
                    correct=False,
                    error=f"Failed to parse output: {proc.stdout[:500]} {proc.stderr[:500]}",
                )

            if "error" in data:
                return CorrectnessResult(
                    correct=False,
                    error=f"{data['error']}\n{data.get('traceback', '')[:1000]}",
                )

            return CorrectnessResult(
                correct=data["correct"],
                max_diff=data["max_diff"],
                mean_diff=data["mean_diff"],
                median_diff=data["median_diff"],
                std_diff=data["std_diff"],
                atol=data["atol"],
                rtol=data["rtol"],
                tolerance=data["tolerance"],
                num_elements=data["num_elements"],
                num_mismatched=data["num_mismatched"],
                mismatch_percentage=data["mismatch_percentage"],
            )

        except subprocess.TimeoutExpired:
            return CorrectnessResult(correct=False, error=f"Timeout ({self.timeout}s)")
        except Exception as e:
            return CorrectnessResult(correct=False, error=str(e))

    def _run_benchmark(
        self,
        solution_path: Path,
        reference_path: Path,
        tmpdir: Path,
    ) -> BenchmarkResult:
        """Run benchmark comparing solution to reference."""

        benchmark_script = f'''
import sys
import json
import torch
import importlib.util
import time

def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

try:
    ref_mod = load_module("{reference_path}", "reference")
    sol_mod = load_module("{solution_path}", "solution")

    device = "{self.device}"
    warmup = {self.warmup_runs}
    runs = {self.benchmark_runs}

    # Get inputs
    if hasattr(ref_mod, "get_init_inputs"):
        init_inputs = ref_mod.get_init_inputs()
    else:
        init_inputs = []

    ref_model = ref_mod.Model(*init_inputs).to(device).eval()
    sol_model = sol_mod.Model(*init_inputs).to(device).eval()

    if hasattr(ref_mod, "get_inputs"):
        inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in ref_mod.get_inputs()]
    else:
        inputs = [torch.randn(16, 1024, device=device)]

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            ref_model(*inputs)
            sol_model(*inputs)

    torch.cuda.synchronize()

    # Benchmark reference
    ref_times = []
    with torch.no_grad():
        for _ in range(runs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            ref_model(*inputs)
            torch.cuda.synchronize()
            end = time.perf_counter()
            ref_times.append((end - start) * 1e6)  # Convert to microseconds

    # Benchmark solution
    sol_times = []
    with torch.no_grad():
        for _ in range(runs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            sol_model(*inputs)
            torch.cuda.synchronize()
            end = time.perf_counter()
            sol_times.append((end - start) * 1e6)

    import statistics

    ref_mean = statistics.mean(ref_times)
    sol_mean = statistics.mean(sol_times)
    ref_std = statistics.stdev(ref_times) if len(ref_times) > 1 else 0
    sol_std = statistics.stdev(sol_times) if len(sol_times) > 1 else 0

    speedup = ref_mean / sol_mean if sol_mean > 0 else 0

    result = {{
        "baseline_time_us": ref_mean,
        "solution_time_us": sol_mean,
        "speedup": speedup,
        "baseline_std_us": ref_std,
        "solution_std_us": sol_std,
        "warmup_runs": warmup,
        "benchmark_runs": runs,
    }}

    print(json.dumps(result))

except Exception as e:
    import traceback
    print(json.dumps({{"error": str(e), "traceback": traceback.format_exc()}}))
'''

        try:
            proc = subprocess.run(
                [sys.executable, "-c", benchmark_script],
                capture_output=True,
                text=True,
                timeout=self.timeout * 2,  # Longer timeout for benchmark
            )

            try:
                data = json.loads(proc.stdout.strip().split("\n")[-1])
            except:
                return BenchmarkResult(
                    error=f"Failed to parse: {proc.stdout[:500]} {proc.stderr[:500]}"
                )

            if "error" in data:
                return BenchmarkResult(error=data["error"])

            return BenchmarkResult(
                baseline_time_us=data["baseline_time_us"],
                solution_time_us=data["solution_time_us"],
                speedup=data["speedup"],
                baseline_std_us=data["baseline_std_us"],
                solution_std_us=data["solution_std_us"],
                warmup_runs=data["warmup_runs"],
                benchmark_runs=data["benchmark_runs"],
            )

        except subprocess.TimeoutExpired:
            return BenchmarkResult(error=f"Benchmark timeout ({self.timeout * 2}s)")
        except Exception as e:
            return BenchmarkResult(error=str(e))

    def _compute_reward(self, result: EvalResult) -> float:
        """
        Compute reward from evaluation result.

        Reward structure:
        - Compilation failure: -0.5 (penalty)
        - Correctness failure: -0.25 (penalty)
        - Correct but slower than baseline: small negative (speedup - 1.0) * 0.5
        - Correct and faster: positive reward = min(speedup - 1.0, 2.0)

        Key principle: Only positive reward when speedup > 1.0x baseline.
        Compilation + correctness alone does NOT give positive reward.
        """
        # Compilation failure: penalty
        if not result.compilation.success:
            return -0.5

        # Correctness failure: penalty
        if result.correctness and not result.correctness.correct:
            return -0.25

        # Benchmark available: reward based on speedup
        if result.benchmark and result.benchmark.speedup:
            speedup = result.benchmark.speedup
            if speedup > 1.0:
                # Positive reward only when beating baseline, capped at 2.0
                return min(speedup - 1.0, 2.0)
            else:
                # Small penalty for being slower than baseline
                return (speedup - 1.0) * 0.5

        # Correct but no benchmark (edge case): no reward
        return 0.0
