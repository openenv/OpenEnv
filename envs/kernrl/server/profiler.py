"""
GPU Profiling for KernelBench

Comprehensive profiling suite that extracts actionable metrics:
- NSight Systems (system-level timing)
- NSight Compute (kernel-level performance)
- Compute Sanitizer (correctness bugs)
- torch.profiler (PyTorch-level view)
- Assembly analysis (PTX/SASS)
- Roofline metrics (arithmetic intensity, theoretical vs achieved)
- Hardware counters (warp divergence, memory bandwidth)

All metrics are curated to be:
1. Actionable - agent can do something with this info
2. Interpretable - clear what good/bad looks like
3. Structured - returned as dataclasses, not raw text
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from enum import auto, Enum
from pathlib import Path
from typing import Optional


class ProfilerType(Enum):
    """Available profilers."""

    NSYS = auto()  # NSight Systems - system-level
    NCU = auto()  # NSight Compute - kernel-level
    SANITIZER = auto()  # Compute Sanitizer - correctness
    TORCH = auto()  # torch.profiler - PyTorch-level
    ASSEMBLY = auto()  # PTX/SASS analysis


@dataclass
class KernelInfo:
    """Information about a single kernel invocation."""

    name: str
    duration_us: float = 0.0
    grid_size: tuple = (0, 0, 0)
    block_size: tuple = (0, 0, 0)
    registers_per_thread: int = 0
    shared_mem_bytes: int = 0
    # Performance metrics
    compute_throughput_pct: float = 0.0
    memory_throughput_pct: float = 0.0
    achieved_occupancy_pct: float = 0.0
    # Bottleneck indicators
    is_memory_bound: bool = False
    is_compute_bound: bool = False
    is_latency_bound: bool = False


@dataclass
class NsysProfile:
    """NSight Systems profile - system-level view."""

    success: bool = False
    error: Optional[str] = None

    # Timing breakdown
    total_gpu_time_us: float = 0.0
    total_cuda_api_time_us: float = 0.0
    total_memory_time_us: float = 0.0

    # Operation counts
    kernel_launches: int = 0
    memory_operations: int = 0
    sync_operations: int = 0

    # Per-kernel breakdown
    kernels: list[dict] = field(default_factory=list)

    # Actionable insights
    insights: list[str] = field(default_factory=list)

    def to_agent_summary(self) -> str:
        """Format as actionable summary for the agent."""
        if not self.success:
            return f"NSight Systems: Failed - {self.error}"

        lines = ["## NSight Systems Profile (System-Level)"]
        lines.append("")
        lines.append("### Timing Breakdown")
        lines.append(f"  GPU Kernel Time: {self.total_gpu_time_us:.2f} us")
        lines.append(f"  CUDA API Overhead: {self.total_cuda_api_time_us:.2f} us")
        lines.append(f"  Memory Operations: {self.total_memory_time_us:.2f} us")

        lines.append("")
        lines.append("### Operation Counts")
        lines.append(f"  Kernel Launches: {self.kernel_launches}")
        lines.append(f"  Memory Ops: {self.memory_operations}")
        lines.append(f"  Sync Points: {self.sync_operations}")

        if self.kernels:
            lines.append("")
            lines.append("### Kernel Breakdown")
            for k in self.kernels[:5]:  # Top 5 kernels
                name = k.get("name", "unknown")[:40]
                time = k.get("time_us", 0)
                pct = k.get("time_pct", 0)
                lines.append(f"  {name}: {time:.2f} us ({pct:.1f}%)")

        if self.insights:
            lines.append("")
            lines.append("### Optimization Hints")
            for insight in self.insights:
                lines.append(f"  - {insight}")

        return "\n".join(lines)


@dataclass
class NcuProfile:
    """NSight Compute profile - kernel-level view."""

    success: bool = False
    error: Optional[str] = None

    # Aggregate metrics
    total_kernel_time_us: float = 0.0
    avg_compute_throughput_pct: float = 0.0
    avg_memory_throughput_pct: float = 0.0
    avg_achieved_occupancy_pct: float = 0.0

    # Resource usage
    max_registers_per_thread: int = 0
    max_shared_mem_bytes: int = 0
    total_dram_bytes_read: int = 0
    total_dram_bytes_written: int = 0

    # Bottleneck analysis
    bottleneck: str = "unknown"  # "memory", "compute", "latency", "balanced"
    limiting_factor: str = ""

    # Per-kernel details
    kernels: list[KernelInfo] = field(default_factory=list)

    # Actionable insights
    insights: list[str] = field(default_factory=list)

    def to_agent_summary(self) -> str:
        """Format as actionable summary for the agent."""
        if not self.success:
            return f"NSight Compute: Failed - {self.error}"

        lines = ["## NSight Compute Profile (Kernel-Level)"]

        lines.append("")
        lines.append("### Performance Summary")
        lines.append(
            f"  Compute Throughput: {self.avg_compute_throughput_pct:.1f}% of peak"
        )
        lines.append(
            f"  Memory Throughput: {self.avg_memory_throughput_pct:.1f}% of peak"
        )
        lines.append(f"  Achieved Occupancy: {self.avg_achieved_occupancy_pct:.1f}%")
        lines.append(f"  Bottleneck: {self.bottleneck.upper()}")
        if self.limiting_factor:
            lines.append(f"  Limiting Factor: {self.limiting_factor}")

        lines.append("")
        lines.append("### Resource Usage")
        lines.append(f"  Registers/Thread: {self.max_registers_per_thread}")
        lines.append(f"  Shared Memory: {self.max_shared_mem_bytes:,} bytes")
        lines.append(f"  DRAM Read: {self.total_dram_bytes_read:,} bytes")
        lines.append(f"  DRAM Written: {self.total_dram_bytes_written:,} bytes")

        if self.kernels:
            lines.append("")
            lines.append("### Kernel Details")
            for k in self.kernels[:3]:  # Top 3 kernels
                lines.append(f"  {k.name[:40]}:")
                lines.append(f"    Duration: {k.duration_us:.2f} us")
                lines.append(f"    Grid: {k.grid_size}, Block: {k.block_size}")
                lines.append(f"    Occupancy: {k.achieved_occupancy_pct:.1f}%")
                if k.is_memory_bound:
                    lines.append(f"    Status: MEMORY BOUND")
                elif k.is_compute_bound:
                    lines.append(f"    Status: COMPUTE BOUND")

        if self.insights:
            lines.append("")
            lines.append("### Optimization Hints")
            for insight in self.insights:
                lines.append(f"  - {insight}")

        return "\n".join(lines)


@dataclass
class SanitizerResult:
    """Compute Sanitizer results - correctness checking."""

    success: bool = False
    error: Optional[str] = None

    # Error counts by type
    memcheck_errors: int = 0
    racecheck_errors: int = 0
    initcheck_errors: int = 0
    synccheck_errors: int = 0

    # Detailed error messages
    errors: list[dict] = field(default_factory=list)  # {type, message, location}

    # Summary
    has_memory_errors: bool = False
    has_race_conditions: bool = False
    has_uninitialized_access: bool = False
    has_sync_errors: bool = False

    def to_agent_summary(self) -> str:
        """Format as actionable summary for the agent."""
        if not self.success:
            return f"Compute Sanitizer: Failed - {self.error}"

        total_errors = (
            self.memcheck_errors
            + self.racecheck_errors
            + self.initcheck_errors
            + self.synccheck_errors
        )

        if total_errors == 0:
            return "## Compute Sanitizer: PASS (no memory/sync errors detected)"

        lines = ["## Compute Sanitizer: ERRORS DETECTED"]
        lines.append("")

        if self.memcheck_errors > 0:
            lines.append(f"### Memory Errors: {self.memcheck_errors}")
            lines.append("  Out-of-bounds or misaligned memory access detected.")
            lines.append("  Fix: Check array bounds and pointer arithmetic.")

        if self.racecheck_errors > 0:
            lines.append(f"### Race Conditions: {self.racecheck_errors}")
            lines.append("  Shared memory data races detected.")
            lines.append("  Fix: Add __syncthreads() or use atomic operations.")

        if self.initcheck_errors > 0:
            lines.append(f"### Uninitialized Access: {self.initcheck_errors}")
            lines.append("  Reading uninitialized global memory.")
            lines.append("  Fix: Initialize memory before reading.")

        if self.synccheck_errors > 0:
            lines.append(f"### Sync Errors: {self.synccheck_errors}")
            lines.append("  Invalid synchronization primitive usage.")
            lines.append("  Fix: Ensure all threads reach sync points.")

        if self.errors:
            lines.append("")
            lines.append("### Error Details")
            for err in self.errors[:5]:  # Top 5 errors
                lines.append(
                    f"  [{err.get('type', 'unknown')}] {err.get('message', '')[:80]}"
                )
                if err.get("location"):
                    lines.append(f"    at {err['location']}")

        return "\n".join(lines)


@dataclass
class TorchProfile:
    """torch.profiler results - PyTorch-level view."""

    success: bool = False
    error: Optional[str] = None

    # CPU time breakdown
    total_cpu_time_us: float = 0.0
    total_cuda_time_us: float = 0.0

    # Top operators
    top_operators: list[dict] = field(
        default_factory=list
    )  # {name, cpu_time_us, cuda_time_us, calls}

    # Memory events
    peak_memory_bytes: int = 0
    memory_allocated_bytes: int = 0

    def to_agent_summary(self) -> str:
        """Format as actionable summary for the agent."""
        if not self.success:
            return f"torch.profiler: Failed - {self.error}"

        lines = ["## torch.profiler (PyTorch-Level)"]
        lines.append("")
        lines.append("### Time Breakdown")
        lines.append(f"  Total CPU Time: {self.total_cpu_time_us:.2f} us")
        lines.append(f"  Total CUDA Time: {self.total_cuda_time_us:.2f} us")

        if self.top_operators:
            lines.append("")
            lines.append("### Top Operators (by CUDA time)")
            for op in self.top_operators[:10]:
                name = op.get("name", "unknown")[:30]
                cuda_time = op.get("cuda_time_us", 0)
                cpu_time = op.get("cpu_time_us", 0)
                calls = op.get("calls", 0)
                lines.append(
                    f"  {name}: {cuda_time:.1f} us (CPU: {cpu_time:.1f} us, calls: {calls})"
                )

        if self.peak_memory_bytes > 0:
            lines.append("")
            lines.append("### Memory")
            lines.append(f"  Peak Memory: {self.peak_memory_bytes / 1e6:.2f} MB")
            lines.append(f"  Allocated: {self.memory_allocated_bytes / 1e6:.2f} MB")

        return "\n".join(lines)


@dataclass
class AssemblyAnalysis:
    """PTX/SASS assembly analysis."""

    success: bool = False
    error: Optional[str] = None

    # PTX stats
    ptx_instructions: int = 0
    ptx_registers: int = 0
    ptx_shared_mem: int = 0

    # SASS stats (actual GPU assembly)
    sass_instructions: int = 0
    sass_registers: int = 0

    # Instruction mix
    memory_instructions: int = 0
    compute_instructions: int = 0
    control_instructions: int = 0

    # Key patterns detected
    patterns: list[str] = field(default_factory=list)

    # Raw assembly (truncated)
    ptx_snippet: str = ""
    sass_snippet: str = ""

    def to_agent_summary(self) -> str:
        """Format as actionable summary for the agent."""
        if not self.success:
            return f"Assembly Analysis: Failed - {self.error}"

        lines = ["## Assembly Analysis (PTX/SASS)"]
        lines.append("")

        lines.append("### Instruction Counts")
        lines.append(f"  PTX Instructions: {self.ptx_instructions}")
        lines.append(f"  SASS Instructions: {self.sass_instructions}")
        lines.append(f"  Registers Used: {self.sass_registers}")

        if (
            self.memory_instructions
            + self.compute_instructions
            + self.control_instructions
            > 0
        ):
            lines.append("")
            lines.append("### Instruction Mix")
            total = (
                self.memory_instructions
                + self.compute_instructions
                + self.control_instructions
            )
            lines.append(
                f"  Memory: {self.memory_instructions} ({100 * self.memory_instructions / total:.1f}%)"
            )
            lines.append(
                f"  Compute: {self.compute_instructions} ({100 * self.compute_instructions / total:.1f}%)"
            )
            lines.append(
                f"  Control: {self.control_instructions} ({100 * self.control_instructions / total:.1f}%)"
            )

        if self.patterns:
            lines.append("")
            lines.append("### Detected Patterns")
            for pattern in self.patterns:
                lines.append(f"  - {pattern}")

        if self.sass_snippet:
            lines.append("")
            lines.append("### SASS Snippet (first 20 instructions)")
            lines.append("```")
            lines.append(self.sass_snippet[:1000])
            lines.append("```")

        return "\n".join(lines)


@dataclass
class RooflineMetrics:
    """Roofline model metrics for performance analysis."""

    success: bool = False
    error: Optional[str] = None

    # Arithmetic intensity (FLOPs per byte)
    arithmetic_intensity: float = 0.0

    # Theoretical peaks (for the target GPU)
    peak_flops_tflops: float = 0.0  # Theoretical peak TFLOPS
    peak_bandwidth_gbps: float = 0.0  # Theoretical peak memory bandwidth

    # Achieved performance
    achieved_flops_tflops: float = 0.0
    achieved_bandwidth_gbps: float = 0.0

    # Efficiency
    compute_efficiency_pct: float = 0.0  # achieved / peak FLOPs
    memory_efficiency_pct: float = 0.0  # achieved / peak bandwidth

    # Roofline classification
    roofline_bound: str = "unknown"  # "compute", "memory", "balanced"
    ridge_point: float = 0.0  # AI where compute = memory bound

    # Warp-level metrics
    warp_execution_efficiency_pct: float = 0.0
    branch_divergence_pct: float = 0.0
    active_warps_per_sm: float = 0.0

    def to_agent_summary(self) -> str:
        """Format as actionable summary for the agent."""
        if not self.success:
            return f"Roofline Analysis: Failed - {self.error}"

        lines = ["## Roofline Analysis"]
        lines.append("")

        lines.append("### Arithmetic Intensity")
        lines.append(f"  AI: {self.arithmetic_intensity:.2f} FLOPs/byte")
        lines.append(f"  Ridge Point: {self.ridge_point:.2f} FLOPs/byte")
        if self.arithmetic_intensity < self.ridge_point:
            lines.append(f"  Status: MEMORY BOUND (AI < ridge point)")
        else:
            lines.append(f"  Status: COMPUTE BOUND (AI >= ridge point)")

        lines.append("")
        lines.append("### Theoretical vs Achieved")
        lines.append(f"  Peak Compute: {self.peak_flops_tflops:.1f} TFLOPS")
        lines.append(
            f"  Achieved Compute: {self.achieved_flops_tflops:.3f} TFLOPS ({self.compute_efficiency_pct:.1f}%)"
        )
        lines.append(f"  Peak Bandwidth: {self.peak_bandwidth_gbps:.0f} GB/s")
        lines.append(
            f"  Achieved Bandwidth: {self.achieved_bandwidth_gbps:.1f} GB/s ({self.memory_efficiency_pct:.1f}%)"
        )

        lines.append("")
        lines.append("### Warp Efficiency")
        lines.append(
            f"  Warp Execution Efficiency: {self.warp_execution_efficiency_pct:.1f}%"
        )
        lines.append(f"  Branch Divergence: {self.branch_divergence_pct:.1f}%")
        lines.append(f"  Active Warps/SM: {self.active_warps_per_sm:.1f}")

        # Insights
        lines.append("")
        lines.append("### Optimization Guidance")
        if self.roofline_bound == "memory":
            lines.append("  - Kernel is memory-bound. Optimize memory access patterns.")
            lines.append("  - Consider: coalescing, shared memory caching, data reuse.")
        elif self.roofline_bound == "compute":
            lines.append("  - Kernel is compute-bound. Good memory efficiency.")
            lines.append("  - Consider: instruction-level parallelism, tensor cores.")
        if self.branch_divergence_pct > 10:
            lines.append(
                f"  - High branch divergence ({self.branch_divergence_pct:.1f}%). Reduce conditionals."
            )
        if self.warp_execution_efficiency_pct < 80:
            lines.append(
                f"  - Low warp efficiency ({self.warp_execution_efficiency_pct:.1f}%). Improve thread utilization."
            )

        return "\n".join(lines)


# GPU specifications for roofline analysis
GPU_SPECS = {
    "RTX 3090": {"peak_tflops": 35.6, "peak_bandwidth_gbps": 936, "sm_count": 82},
    "RTX 4090": {"peak_tflops": 82.6, "peak_bandwidth_gbps": 1008, "sm_count": 128},
    "A100": {"peak_tflops": 19.5, "peak_bandwidth_gbps": 2039, "sm_count": 108},  # FP32
    "H100": {"peak_tflops": 67.0, "peak_bandwidth_gbps": 3350, "sm_count": 132},  # FP32
    "B200": {
        "peak_tflops": 90.0,
        "peak_bandwidth_gbps": 8000,
        "sm_count": 160,
    },  # FP32 estimate
    "default": {"peak_tflops": 20.0, "peak_bandwidth_gbps": 1000, "sm_count": 80},
}


class GPUProfiler:
    """
    Comprehensive GPU profiler with all metrics.

    Usage:
        profiler = GPUProfiler(enable_all=True)
        results = profiler.profile_all(script_path, workdir)
    """

    def __init__(
        self,
        enable_nsys: bool = True,
        enable_ncu: bool = True,
        enable_sanitizer: bool = True,
        enable_torch_profiler: bool = True,
        enable_assembly: bool = True,
        enable_roofline: bool = True,
        nsys_timeout: int = 60,
        ncu_timeout: int = 120,
        sanitizer_timeout: int = 60,
    ):
        self.enable_nsys = enable_nsys
        self.enable_ncu = enable_ncu
        self.enable_sanitizer = enable_sanitizer
        self.enable_torch_profiler = enable_torch_profiler
        self.enable_assembly = enable_assembly
        self.enable_roofline = enable_roofline
        self.nsys_timeout = nsys_timeout
        self.ncu_timeout = ncu_timeout
        self.sanitizer_timeout = sanitizer_timeout

        # Find profiler binaries
        self.nsys_path = shutil.which("nsys")
        self.ncu_path = shutil.which("ncu")
        self.sanitizer_path = shutil.which("compute-sanitizer")
        self.cuobjdump_path = shutil.which("cuobjdump")
        self.nvdisasm_path = shutil.which("nvdisasm")

        # Disable tools if not found
        if enable_nsys and not self.nsys_path:
            print("Warning: nsys not found, NSight Systems disabled")
            self.enable_nsys = False

        if enable_ncu and not self.ncu_path:
            print("Warning: ncu not found, NSight Compute disabled")
            self.enable_ncu = False

        if enable_sanitizer and not self.sanitizer_path:
            print("Warning: compute-sanitizer not found, Sanitizer disabled")
            self.enable_sanitizer = False

        if enable_assembly and not self.cuobjdump_path:
            print("Warning: cuobjdump not found, Assembly analysis disabled")
            self.enable_assembly = False

        # Detect GPU for roofline
        self.gpu_name = self._detect_gpu()
        self.gpu_specs = GPU_SPECS.get(self.gpu_name, GPU_SPECS["default"])

    def _detect_gpu(self) -> str:
        """Detect GPU name for specs lookup."""
        try:
            import torch

            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                for key in GPU_SPECS:
                    if key.lower() in name.lower():
                        return key
        except:
            pass
        return "default"

    # =========================================================================
    # NSight Systems
    # =========================================================================

    def run_nsys(self, script_path: Path, workdir: Path) -> NsysProfile:
        """Run NSight Systems profiling."""
        if not self.enable_nsys:
            return NsysProfile(success=False, error="nsys disabled")

        output_base = workdir / "nsys_report"

        try:
            proc = subprocess.run(
                [
                    self.nsys_path,
                    "profile",
                    "-o",
                    str(output_base),
                    "-f",
                    "true",
                    "--stats=true",
                    "--export=sqlite",
                    sys.executable,
                    str(script_path),
                ],
                capture_output=True,
                text=True,
                timeout=self.nsys_timeout,
                cwd=workdir,
            )

            raw_output = proc.stdout + proc.stderr
            return self._parse_nsys_output(raw_output, output_base)

        except subprocess.TimeoutExpired:
            return NsysProfile(success=False, error=f"Timeout ({self.nsys_timeout}s)")
        except Exception as e:
            return NsysProfile(success=False, error=str(e))

    def _parse_nsys_output(self, raw_output: str, output_base: Path) -> NsysProfile:
        """Parse nsys output to extract metrics."""
        profile = NsysProfile(success=True)
        lines = raw_output.split("\n")

        current_section = None

        for i, line in enumerate(lines):
            if "Executing '" in line and "stats report" in line:
                section_match = re.search(r"Executing '(\w+)'", line)
                if section_match:
                    section_name = section_match.group(1)
                    if "cuda_api" in section_name:
                        current_section = "api"
                    elif "cuda_gpu_kern" in section_name:
                        current_section = "kern"
                    elif "cuda_gpu_mem_time" in section_name:
                        current_section = "memtime"
                    elif "cuda_gpu_mem" in section_name:
                        current_section = "mem"
                    else:
                        current_section = None
                continue

            if line.strip().startswith("---") or line.strip().startswith("==="):
                continue
            if "Time (%)" in line or line.strip() == "":
                continue

            if current_section == "api":
                parts = line.split()
                if len(parts) >= 9:
                    try:
                        api_name = parts[-1].lower()
                        total_time_ns = float(parts[1].replace(",", ""))
                        total_time_us = total_time_ns / 1000.0
                        instances = int(parts[2].replace(",", ""))

                        profile.total_cuda_api_time_us += total_time_us

                        if "launch" in api_name:
                            profile.kernel_launches += instances
                        if "memcpy" in api_name or "memset" in api_name:
                            profile.memory_operations += instances
                        if "synchronize" in api_name:
                            profile.sync_operations += instances
                    except (ValueError, IndexError):
                        pass

            elif current_section == "kern":
                parts = line.split()
                if len(parts) >= 9:
                    try:
                        time_pct = float(parts[0].replace(",", ""))
                        total_time_ns = float(parts[1].replace(",", ""))
                        total_time_us = total_time_ns / 1000.0
                        instances = int(parts[2].replace(",", ""))
                        kernel_name = (
                            " ".join(parts[8:]) if len(parts) > 8 else "unknown"
                        )

                        profile.total_gpu_time_us += total_time_us
                        profile.kernels.append(
                            {
                                "name": kernel_name,
                                "time_us": total_time_us,
                                "time_pct": time_pct,
                                "instances": instances,
                            }
                        )
                    except (ValueError, IndexError):
                        pass

            elif current_section == "memtime":
                parts = line.split()
                if len(parts) >= 9:
                    try:
                        total_time_ns = float(parts[1].replace(",", ""))
                        total_time_us = total_time_ns / 1000.0
                        instances = int(parts[2].replace(",", ""))
                        profile.total_memory_time_us += total_time_us
                        profile.memory_operations += instances
                    except (ValueError, IndexError):
                        pass

        profile.kernels.sort(key=lambda x: x.get("time_us", 0), reverse=True)
        profile.insights = self._generate_nsys_insights(profile)
        return profile

    def _generate_nsys_insights(self, profile: NsysProfile) -> list[str]:
        """Generate actionable insights from nsys profile."""
        insights = []

        if profile.kernel_launches > 10:
            insights.append(
                f"High kernel launch count ({profile.kernel_launches}). "
                "Consider fusing kernels to reduce launch overhead."
            )

        if profile.total_cuda_api_time_us > 0 and profile.total_gpu_time_us > 0:
            api_ratio = profile.total_cuda_api_time_us / profile.total_gpu_time_us
            if api_ratio > 0.5:
                insights.append(
                    f"CUDA API overhead is {api_ratio:.1f}x GPU time. "
                    "Consider reducing API calls or using CUDA graphs."
                )

        if profile.total_memory_time_us > 0 and profile.total_gpu_time_us > 0:
            mem_ratio = profile.total_memory_time_us / profile.total_gpu_time_us
            if mem_ratio > 0.3:
                insights.append(
                    f"Memory operations take {mem_ratio * 100:.0f}% of GPU time. "
                    "Consider reducing memory transfers or using pinned memory."
                )

        if profile.sync_operations > 5:
            insights.append(
                f"Multiple sync points ({profile.sync_operations}). "
                "Consider batching operations to reduce synchronization."
            )

        if not insights:
            insights.append("No major system-level bottlenecks detected.")

        return insights

    # =========================================================================
    # NSight Compute
    # =========================================================================

    def run_ncu(self, script_path: Path, workdir: Path) -> NcuProfile:
        """Run NSight Compute profiling."""
        if not self.enable_ncu:
            return NcuProfile(success=False, error="ncu disabled")

        try:
            proc = subprocess.run(
                [
                    self.ncu_path,
                    "--metrics",
                    "sm__throughput.avg.pct_of_peak_sustained_elapsed,"
                    "dram__throughput.avg.pct_of_peak_sustained_elapsed,"
                    "sm__warps_active.avg.pct_of_peak_sustained_elapsed,"
                    "dram__bytes_read.sum,"
                    "dram__bytes_write.sum,"
                    "l2__throughput.avg.pct_of_peak_sustained_elapsed,"
                    "launch__registers_per_thread,"
                    "launch__shared_mem_per_block_driver,"
                    "launch__grid_size,"
                    "launch__block_size,"
                    "smsp__thread_inst_executed_per_inst_executed.ratio,"
                    "smsp__sass_average_branch_targets_threads_uniform.pct",
                    "--csv",
                    "--target-processes",
                    "all",
                    sys.executable,
                    str(script_path),
                ],
                capture_output=True,
                text=True,
                timeout=self.ncu_timeout,
                cwd=workdir,
            )

            raw_output = proc.stdout + proc.stderr
            return self._parse_ncu_output(raw_output)

        except subprocess.TimeoutExpired:
            return NcuProfile(success=False, error=f"Timeout ({self.ncu_timeout}s)")
        except Exception as e:
            return NcuProfile(success=False, error=str(e))

    def _parse_ncu_output(self, raw_output: str) -> NcuProfile:
        """Parse ncu CSV output to extract metrics."""
        profile = NcuProfile(success=True)
        lines = raw_output.strip().split("\n")

        header_idx = -1
        for i, line in enumerate(lines):
            if '"Kernel Name"' in line or "Kernel Name" in line:
                header_idx = i
                break

        if header_idx < 0:
            return self._parse_ncu_text_output(raw_output)

        try:
            import csv
            from io import StringIO

            csv_text = "\n".join(lines[header_idx:])
            reader = csv.DictReader(StringIO(csv_text))

            compute_throughputs = []
            memory_throughputs = []
            occupancies = []

            for row in reader:
                kernel = KernelInfo(name=row.get("Kernel Name", "unknown")[:60])

                sm_tp = row.get("sm__throughput.avg.pct_of_peak_sustained_elapsed", "0")
                dram_tp = row.get(
                    "dram__throughput.avg.pct_of_peak_sustained_elapsed", "0"
                )

                try:
                    kernel.compute_throughput_pct = float(
                        sm_tp.replace(",", "").replace("%", "")
                    )
                    compute_throughputs.append(kernel.compute_throughput_pct)
                except:
                    pass

                try:
                    kernel.memory_throughput_pct = float(
                        dram_tp.replace(",", "").replace("%", "")
                    )
                    memory_throughputs.append(kernel.memory_throughput_pct)
                except:
                    pass

                occ = row.get("sm__warps_active.avg.pct_of_peak_sustained_elapsed", "0")
                try:
                    kernel.achieved_occupancy_pct = float(
                        occ.replace(",", "").replace("%", "")
                    )
                    occupancies.append(kernel.achieved_occupancy_pct)
                except:
                    pass

                regs = row.get("launch__registers_per_thread", "0")
                try:
                    kernel.registers_per_thread = int(float(regs.replace(",", "")))
                    profile.max_registers_per_thread = max(
                        profile.max_registers_per_thread, kernel.registers_per_thread
                    )
                except:
                    pass

                smem = row.get("launch__shared_mem_per_block_driver", "0")
                try:
                    kernel.shared_mem_bytes = int(float(smem.replace(",", "")))
                    profile.max_shared_mem_bytes = max(
                        profile.max_shared_mem_bytes, kernel.shared_mem_bytes
                    )
                except:
                    pass

                dram_read = row.get("dram__bytes_read.sum", "0")
                dram_write = row.get("dram__bytes_write.sum", "0")
                try:
                    profile.total_dram_bytes_read += int(
                        float(dram_read.replace(",", ""))
                    )
                    profile.total_dram_bytes_written += int(
                        float(dram_write.replace(",", ""))
                    )
                except:
                    pass

                if kernel.memory_throughput_pct > kernel.compute_throughput_pct + 10:
                    kernel.is_memory_bound = True
                elif kernel.compute_throughput_pct > kernel.memory_throughput_pct + 10:
                    kernel.is_compute_bound = True
                else:
                    kernel.is_latency_bound = True

                profile.kernels.append(kernel)

            if compute_throughputs:
                profile.avg_compute_throughput_pct = sum(compute_throughputs) / len(
                    compute_throughputs
                )
            if memory_throughputs:
                profile.avg_memory_throughput_pct = sum(memory_throughputs) / len(
                    memory_throughputs
                )
            if occupancies:
                profile.avg_achieved_occupancy_pct = sum(occupancies) / len(occupancies)

            if (
                profile.avg_memory_throughput_pct
                > profile.avg_compute_throughput_pct + 10
            ):
                profile.bottleneck = "memory"
                profile.limiting_factor = "DRAM bandwidth"
            elif (
                profile.avg_compute_throughput_pct
                > profile.avg_memory_throughput_pct + 10
            ):
                profile.bottleneck = "compute"
                profile.limiting_factor = "SM throughput"
            elif profile.avg_achieved_occupancy_pct < 50:
                profile.bottleneck = "latency"
                profile.limiting_factor = "Low occupancy"
            else:
                profile.bottleneck = "balanced"
                profile.limiting_factor = "Well optimized"

        except Exception as e:
            profile.error = f"CSV parse error: {e}"

        profile.insights = self._generate_ncu_insights(profile)
        return profile

    def _parse_ncu_text_output(self, raw_output: str) -> NcuProfile:
        """Fallback parser for non-CSV ncu output."""
        profile = NcuProfile(success=True)
        lines = raw_output.split("\n")

        for line in lines:
            line_lower = line.lower()

            if "compute" in line_lower and "throughput" in line_lower:
                match = re.search(r"([\d.]+)\s*%", line)
                if match:
                    profile.avg_compute_throughput_pct = float(match.group(1))

            if "memory" in line_lower and "throughput" in line_lower:
                match = re.search(r"([\d.]+)\s*%", line)
                if match:
                    profile.avg_memory_throughput_pct = float(match.group(1))

            if "occupancy" in line_lower:
                match = re.search(r"([\d.]+)\s*%", line)
                if match:
                    profile.avg_achieved_occupancy_pct = float(match.group(1))

            if "registers" in line_lower:
                match = re.search(r"(\d+)", line)
                if match:
                    profile.max_registers_per_thread = int(match.group(1))

        if profile.avg_memory_throughput_pct > profile.avg_compute_throughput_pct + 10:
            profile.bottleneck = "memory"
        elif (
            profile.avg_compute_throughput_pct > profile.avg_memory_throughput_pct + 10
        ):
            profile.bottleneck = "compute"
        else:
            profile.bottleneck = "balanced"

        profile.insights = self._generate_ncu_insights(profile)
        return profile

    def _generate_ncu_insights(self, profile: NcuProfile) -> list[str]:
        """Generate actionable insights from ncu profile."""
        insights = []

        if profile.bottleneck == "memory":
            insights.append(
                "MEMORY BOUND: Optimize memory access patterns. "
                "Consider coalescing, shared memory caching, or reducing data movement."
            )
        elif profile.bottleneck == "compute":
            insights.append(
                "COMPUTE BOUND: Already well-optimized for memory. "
                "Consider algorithmic improvements or instruction-level optimizations."
            )
        elif profile.bottleneck == "latency":
            insights.append(
                "LATENCY BOUND: Low occupancy is limiting performance. "
                "Try reducing register usage or increasing block size."
            )

        if profile.avg_achieved_occupancy_pct < 30:
            insights.append(
                f"Very low occupancy ({profile.avg_achieved_occupancy_pct:.0f}%). "
                "Increase parallelism by using more threads or reducing resource usage."
            )
        elif profile.avg_achieved_occupancy_pct < 50:
            insights.append(
                f"Low occupancy ({profile.avg_achieved_occupancy_pct:.0f}%). "
                "Consider adjusting block size or reducing registers/shared memory."
            )

        if profile.max_registers_per_thread > 64:
            insights.append(
                f"High register usage ({profile.max_registers_per_thread}/thread). "
                "This limits occupancy. Consider using __launch_bounds__ or simplifying."
            )

        if profile.max_shared_mem_bytes > 48 * 1024:
            insights.append(
                f"High shared memory ({profile.max_shared_mem_bytes:,} bytes). "
                "This may limit blocks per SM. Consider reducing or using L2 cache."
            )

        if not insights:
            insights.append(
                "Kernel is reasonably well-optimized at the hardware level."
            )

        return insights

    # =========================================================================
    # Compute Sanitizer
    # =========================================================================

    def run_sanitizer(self, script_path: Path, workdir: Path) -> SanitizerResult:
        """Run compute-sanitizer for correctness checking."""
        if not self.enable_sanitizer:
            return SanitizerResult(success=False, error="compute-sanitizer disabled")

        result = SanitizerResult(success=True)

        # Run each sanitizer tool
        for tool in ["memcheck", "racecheck", "initcheck", "synccheck"]:
            try:
                proc = subprocess.run(
                    [
                        self.sanitizer_path,
                        f"--tool={tool}",
                        "--print-limit=10",
                        sys.executable,
                        str(script_path),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=self.sanitizer_timeout,
                    cwd=workdir,
                )

                output = proc.stdout + proc.stderr
                errors = self._parse_sanitizer_output(output, tool)

                if tool == "memcheck":
                    result.memcheck_errors = len(errors)
                    result.has_memory_errors = len(errors) > 0
                elif tool == "racecheck":
                    result.racecheck_errors = len(errors)
                    result.has_race_conditions = len(errors) > 0
                elif tool == "initcheck":
                    result.initcheck_errors = len(errors)
                    result.has_uninitialized_access = len(errors) > 0
                elif tool == "synccheck":
                    result.synccheck_errors = len(errors)
                    result.has_sync_errors = len(errors) > 0

                result.errors.extend(errors)

            except subprocess.TimeoutExpired:
                pass  # Timeout is OK, just skip this tool
            except Exception as e:
                pass  # Non-fatal

        return result

    def _parse_sanitizer_output(self, output: str, tool: str) -> list[dict]:
        """Parse compute-sanitizer output for errors."""
        errors = []
        lines = output.split("\n")

        for i, line in enumerate(lines):
            if "ERROR" in line.upper() or "HAZARD" in line.upper():
                error = {
                    "type": tool,
                    "message": line.strip()[:200],
                    "location": "",
                }
                # Try to get location from next lines
                if i + 1 < len(lines) and "at" in lines[i + 1].lower():
                    error["location"] = lines[i + 1].strip()[:100]
                errors.append(error)

        return errors

    # =========================================================================
    # torch.profiler
    # =========================================================================

    def run_torch_profiler(self, script_path: Path, workdir: Path) -> TorchProfile:
        """Run torch.profiler for PyTorch-level view."""
        if not self.enable_torch_profiler:
            return TorchProfile(success=False, error="torch.profiler disabled")

        # Create a wrapper script that uses torch.profiler
        profiler_script = workdir / "torch_profile_wrapper.py"
        profiler_output = workdir / "torch_profile.json"

        profiler_script.write_text(f'''
import sys
import json
import torch
from torch.profiler import profile, ProfilerActivity

# Run the original script first to warm up
exec(open("{script_path}").read())

# Import the model
import importlib.util
spec = importlib.util.spec_from_file_location("solution", "{script_path}")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# Get inputs if available
if hasattr(mod, 'get_inputs'):
    inputs = mod.get_inputs()
    inputs = [x.cuda() if hasattr(x, 'cuda') else x for x in inputs]
else:
    inputs = [torch.randn(16, 1024, device='cuda')]

if hasattr(mod, 'get_init_inputs'):
    init_inputs = mod.get_init_inputs()
else:
    init_inputs = []

model = mod.Model(*init_inputs).cuda().eval()

# Warmup
with torch.no_grad():
    for _ in range(5):
        model(*inputs)

torch.cuda.synchronize()

# Profile
results = {{}}
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=True,
) as prof:
    with torch.no_grad():
        for _ in range(10):
            model(*inputs)
    torch.cuda.synchronize()

# Extract metrics
key_averages = prof.key_averages()

operators = []
total_cpu = 0
total_cuda = 0

for item in key_averages:
    cpu_time = item.cpu_time_total
    cuda_time = item.cuda_time_total
    total_cpu += cpu_time
    total_cuda += cuda_time
    operators.append({{
        'name': item.key,
        'cpu_time_us': cpu_time,
        'cuda_time_us': cuda_time,
        'calls': item.count,
    }})

# Sort by CUDA time
operators.sort(key=lambda x: x['cuda_time_us'], reverse=True)

results = {{
    'total_cpu_time_us': total_cpu,
    'total_cuda_time_us': total_cuda,
    'top_operators': operators[:20],
    'peak_memory_bytes': torch.cuda.max_memory_allocated(),
    'memory_allocated_bytes': torch.cuda.memory_allocated(),
}}

with open("{profiler_output}", 'w') as f:
    json.dump(results, f)

print("TORCH_PROFILE_OK")
''')

        try:
            proc = subprocess.run(
                [sys.executable, str(profiler_script)],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=workdir,
            )

            if "TORCH_PROFILE_OK" not in proc.stdout:
                return TorchProfile(success=False, error=proc.stderr[:500])

            with open(profiler_output) as f:
                data = json.load(f)

            return TorchProfile(
                success=True,
                total_cpu_time_us=data.get("total_cpu_time_us", 0),
                total_cuda_time_us=data.get("total_cuda_time_us", 0),
                top_operators=data.get("top_operators", []),
                peak_memory_bytes=data.get("peak_memory_bytes", 0),
                memory_allocated_bytes=data.get("memory_allocated_bytes", 0),
            )

        except subprocess.TimeoutExpired:
            return TorchProfile(success=False, error="Timeout")
        except Exception as e:
            return TorchProfile(success=False, error=str(e))

    # =========================================================================
    # Assembly Analysis (PTX/SASS)
    # =========================================================================

    def run_assembly_analysis(
        self, script_path: Path, workdir: Path
    ) -> AssemblyAnalysis:
        """Extract and analyze PTX/SASS assembly."""
        if not self.enable_assembly or not self.cuobjdump_path:
            return AssemblyAnalysis(success=False, error="Assembly analysis disabled")

        result = AssemblyAnalysis(success=True)

        # First, we need to compile the kernel to a .cubin or get the PTX
        # This requires either a .cu file or extracting from the running process
        # For Triton kernels, we can get the PTX from triton.compile()

        # Create a script that extracts PTX from Triton kernels
        extract_script = workdir / "extract_ptx.py"
        ptx_output = workdir / "kernel.ptx"

        extract_script.write_text(f'''
import sys
import torch
import importlib.util

spec = importlib.util.spec_from_file_location("solution", "{script_path}")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# Try to find Triton kernels and get their PTX
ptx_code = ""

# Check if triton is used
try:
    import triton
    import triton.compiler

    # Look for @triton.jit decorated functions
    for name in dir(mod):
        obj = getattr(mod, name)
        if hasattr(obj, 'cache'):  # Triton JIT functions have cache
            try:
                # Try to get compiled kernel
                if hasattr(obj, 'run') and hasattr(obj.run, 'cache'):
                    for key, kernel in obj.run.cache.items():
                        if hasattr(kernel, 'asm'):
                            if 'ptx' in kernel.asm:
                                ptx_code += kernel.asm['ptx']
            except:
                pass
except ImportError:
    pass

# Also try to get PTX from torch/CUDA kernels via cuobjdump
# This requires the model to have been run at least once

with open("{ptx_output}", 'w') as f:
    f.write(ptx_code)

print(f"PTX_LINES:{{len(ptx_code.split(chr(10)))}}")
''')

        try:
            proc = subprocess.run(
                [sys.executable, str(extract_script)],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=workdir,
            )

            # Read PTX if generated
            if ptx_output.exists():
                ptx_code = ptx_output.read_text()
                result.ptx_snippet = ptx_code[:2000]  # First 2000 chars
                result.ptx_instructions = len(
                    [
                        l
                        for l in ptx_code.split("\n")
                        if l.strip() and not l.strip().startswith("//")
                    ]
                )

                # Analyze instruction mix
                for line in ptx_code.split("\n"):
                    line = line.strip().lower()
                    if any(op in line for op in ["ld.", "st.", "atom.", "red."]):
                        result.memory_instructions += 1
                    elif any(
                        op in line
                        for op in ["add", "mul", "fma", "sub", "div", "mad", "sqrt"]
                    ):
                        result.compute_instructions += 1
                    elif any(op in line for op in ["bra", "call", "ret", "setp", "@"]):
                        result.control_instructions += 1

                # Extract register count
                reg_match = re.search(r"\.reg\s+\.\w+\s+<(\d+)>", ptx_code)
                if reg_match:
                    result.ptx_registers = int(reg_match.group(1))

                # Detect patterns
                if "shfl" in ptx_code.lower():
                    result.patterns.append(
                        "Uses warp shuffle operations (good for reductions)"
                    )
                if "shared" in ptx_code.lower():
                    result.patterns.append("Uses shared memory")
                if "tex." in ptx_code.lower():
                    result.patterns.append("Uses texture memory")
                if ".f16" in ptx_code.lower() or "half" in ptx_code.lower():
                    result.patterns.append("Uses FP16 operations")
                if "wmma" in ptx_code.lower() or "mma" in ptx_code.lower():
                    result.patterns.append("Uses Tensor Cores (WMMA/MMA)")

        except Exception as e:
            result.error = str(e)

        return result

    # =========================================================================
    # Roofline Metrics
    # =========================================================================

    def compute_roofline(
        self, ncu_profile: NcuProfile, benchmark_time_us: float
    ) -> RooflineMetrics:
        """Compute roofline model metrics from NCU data."""
        if not self.enable_roofline:
            return RooflineMetrics(success=False, error="Roofline disabled")

        result = RooflineMetrics(success=True)

        # Get GPU specs
        result.peak_flops_tflops = self.gpu_specs["peak_tflops"]
        result.peak_bandwidth_gbps = self.gpu_specs["peak_bandwidth_gbps"]

        # Calculate ridge point (where compute and memory rooflines meet)
        # ridge_point = peak_flops / peak_bandwidth
        result.ridge_point = (result.peak_flops_tflops * 1e12) / (
            result.peak_bandwidth_gbps * 1e9
        )

        # Calculate arithmetic intensity from NCU data
        total_bytes = (
            ncu_profile.total_dram_bytes_read + ncu_profile.total_dram_bytes_written
        )
        if total_bytes > 0:
            # Estimate FLOPs from compute throughput
            # achieved_flops = peak_flops * (compute_throughput_pct / 100)
            achieved_flops = (
                result.peak_flops_tflops
                * 1e12
                * (ncu_profile.avg_compute_throughput_pct / 100)
            )
            result.achieved_flops_tflops = achieved_flops / 1e12

            # AI = FLOPs / bytes
            # Use benchmark time to estimate total FLOPs
            result.arithmetic_intensity = (
                achieved_flops * (benchmark_time_us / 1e6) / total_bytes
            )

        # Calculate achieved bandwidth
        if benchmark_time_us > 0:
            result.achieved_bandwidth_gbps = (
                total_bytes / (benchmark_time_us / 1e6) / 1e9
            )

        # Calculate efficiency
        if result.peak_flops_tflops > 0:
            result.compute_efficiency_pct = (
                result.achieved_flops_tflops / result.peak_flops_tflops
            ) * 100
        if result.peak_bandwidth_gbps > 0:
            result.memory_efficiency_pct = (
                result.achieved_bandwidth_gbps / result.peak_bandwidth_gbps
            ) * 100

        # Determine roofline bound
        if result.arithmetic_intensity < result.ridge_point:
            result.roofline_bound = "memory"
        else:
            result.roofline_bound = "compute"

        # Warp metrics from NCU
        result.warp_execution_efficiency_pct = ncu_profile.avg_achieved_occupancy_pct
        # Branch divergence would need additional NCU metrics
        result.branch_divergence_pct = (
            0.0  # Placeholder - would need specific NCU metric
        )

        return result


# Convenience function for one-shot profiling
def profile_kernel(
    solution_code: str,
    reference_code: str,
    device: str = "cuda:0",
    enable_nsys: bool = True,
    enable_ncu: bool = True,
    enable_sanitizer: bool = True,
    enable_torch_profiler: bool = True,
    enable_assembly: bool = True,
    enable_roofline: bool = True,
) -> dict:
    """
    Profile a kernel solution with all available profilers.

    Returns dict with all profiling results.
    """
    profiler = GPUProfiler(
        enable_nsys=enable_nsys,
        enable_ncu=enable_ncu,
        enable_sanitizer=enable_sanitizer,
        enable_torch_profiler=enable_torch_profiler,
        enable_assembly=enable_assembly,
        enable_roofline=enable_roofline,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        solution_path = tmpdir / "solution.py"
        reference_path = tmpdir / "reference.py"
        runner_path = tmpdir / "runner.py"

        solution_path.write_text(solution_code)
        reference_path.write_text(reference_code)

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

device = "{device}"

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

# Run for profiling
with torch.no_grad():
    for _ in range(10):
        model(*inputs)

torch.cuda.synchronize()
''')

        results = {
            "nsys": profiler.run_nsys(runner_path, tmpdir)
            if enable_nsys
            else NsysProfile(),
            "ncu": profiler.run_ncu(runner_path, tmpdir)
            if enable_ncu
            else NcuProfile(),
            "sanitizer": profiler.run_sanitizer(runner_path, tmpdir)
            if enable_sanitizer
            else SanitizerResult(),
            "torch_profile": profiler.run_torch_profiler(solution_path, tmpdir)
            if enable_torch_profiler
            else TorchProfile(),
            "assembly": profiler.run_assembly_analysis(solution_path, tmpdir)
            if enable_assembly
            else AssemblyAnalysis(),
        }

        # Compute roofline if we have NCU data
        if enable_roofline and results["ncu"].success:
            benchmark_time = (
                results["nsys"].total_gpu_time_us if results["nsys"].success else 1000.0
            )
            results["roofline"] = profiler.compute_roofline(
                results["ncu"], benchmark_time
            )
        else:
            results["roofline"] = RooflineMetrics()

        return results
