"""What ``spikeinfer doctor`` checks, and why each check exists.

Every entry here corresponds to a way this project has actually failed to run
on a fresh machine. Triton is the big one: on Windows it is a separate package
that JIT-compiles through MSVC, so a working install needs ``cl.exe`` on PATH
and a ``CUDA_HOME`` matching the torch build -- and when any of that is missing
the failure surfaces much later, as a slow engine or a compile error inside a
kernel launch, rather than as "your environment is wrong". The point of this
command is to move that diagnosis to the front.

Checks are ordered cheapest-first and never raise: a probe that blows up is
itself a finding. The exit code is 1 if anything is an outright failure, 0 if
everything is fine or merely a warning, so this can gate CI.
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Literal

Status = Literal["ok", "warn", "fail", "skip"]

_MARK = {"ok": "[ok]  ", "warn": "[warn]", "fail": "[FAIL]", "skip": "[--]  "}


@dataclass
class Check:
    name: str
    status: Status
    detail: str
    hint: str | None = None
    data: dict = field(default_factory=dict)

    def render(self) -> str:
        line = f"{_MARK[self.status]} {self.name:<26} {self.detail}"
        if self.hint and self.status in ("warn", "fail"):
            line += f"\n         -> {self.hint}"
        return line


class Report:
    def __init__(self) -> None:
        self.checks: list[Check] = []

    def add(self, *args, **kwargs) -> Check:
        check = Check(*args, **kwargs)
        self.checks.append(check)
        return check

    @property
    def failed(self) -> bool:
        return any(c.status == "fail" for c in self.checks)

    def render(self) -> str:
        body = "\n".join(c.render() for c in self.checks)
        counts = {s: sum(c.status == s for c in self.checks) for s in ("ok", "warn", "fail", "skip")}
        tail = (
            f"\n\n{counts['ok']} ok, {counts['warn']} warnings, "
            f"{counts['fail']} failures, {counts['skip']} skipped"
        )
        return body + tail

    def to_dict(self) -> dict:
        return {
            "ok": not self.failed,
            "checks": [
                {"name": c.name, "status": c.status, "detail": c.detail, **c.data}
                for c in self.checks
            ],
        }


def _check_python(report: Report) -> None:
    version = sys.version_info
    status: Status = "ok" if version >= (3, 10) else "fail"
    report.add(
        "python",
        status,
        f"{platform.python_version()} ({platform.system()} {platform.machine()})",
        hint="spikeinfer needs Python 3.10 or newer",
        data={"version": platform.python_version()},
    )


def _check_torch(report: Report):
    try:
        import torch
    except ImportError as exc:
        report.add("torch", "fail", str(exc), hint="pip install torch")
        return None
    build = getattr(torch.version, "cuda", None)
    report.add(
        "torch",
        "ok",
        f"{torch.__version__} (cuda build: {build or 'cpu-only'})",
        data={"version": torch.__version__, "cuda_build": build},
    )
    return torch


def _check_cuda(report: Report, torch) -> bool:
    from .sysinfo import cuda_is_usable, format_bytes

    if not cuda_is_usable():
        available = torch.cuda.is_available()
        detail = "no usable CUDA device"
        if available:
            # The exact trap sysinfo.cuda_is_usable exists for.
            detail += " (is_available() is True but device_count() is 0 -- CUDA_VISIBLE_DEVICES?)"
        report.add(
            "cuda device",
            "warn",
            detail,
            hint="CPU mode works: run with --device cpu. Expect single-digit tok/s.",
            data={"available": False},
        )
        return False

    index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(index)
    free, total = torch.cuda.mem_get_info()
    report.add(
        "cuda device",
        "ok",
        f"{props.name}, sm_{props.major}{props.minor}, "
        f"{format_bytes(free)} free of {format_bytes(total)}",
        data={"name": props.name, "free_bytes": free, "total_bytes": total},
    )
    supported = torch.cuda.is_bf16_supported()
    report.add(
        "bfloat16",
        "ok" if supported else "warn",
        "supported" if supported else "not supported -- dtype auto will pick float32",
        hint="float32 doubles the weight footprint; --dtype float16 is not "
        "recommended here (the T residual streams need the exponent range)",
        data={"bf16": supported},
    )
    return True


def _check_host_memory(report: Report) -> None:
    from .sysinfo import cpu_count, format_bytes, host_memory

    total, available = host_memory()
    report.add(
        "host memory",
        "ok" if available > 2 * 2**30 else "warn",
        f"{format_bytes(available)} available of {format_bytes(total)}, {cpu_count()} cores",
        hint="offloaded weights and the CPU KV cache both live here",
        data={"total_bytes": total, "available_bytes": available, "cores": cpu_count()},
    )


def _check_triton(report: Report, has_cuda: bool):
    try:
        import triton
    except ImportError:
        report.add(
            "triton",
            "warn",
            "not installed -- the LIF kernel, packing and paged attention fall "
            "back to pure PyTorch",
            hint="Linux: ships inside the torch wheel. Windows: pip install triton-windows",
            data={"installed": False},
        )
        return None
    report.add(
        "triton",
        "ok",
        getattr(triton, "__version__", "unknown"),
        data={"installed": True, "version": getattr(triton, "__version__", None)},
    )
    return triton


def _check_msvc(report: Report) -> None:
    """Windows only: Triton JITs through cl.exe, so it must be on PATH."""
    if sys.platform != "win32":
        report.add("msvc (cl.exe)", "skip", "not Windows")
        return
    cl = shutil.which("cl.exe")
    report.add(
        "msvc (cl.exe)",
        "ok" if cl else "fail",
        cl or "not on PATH -- Triton cannot compile kernels",
        hint=r"run tools\env.bat first, or install VS2022 Build Tools",
        data={"cl": cl},
    )


def _check_cuda_home(report: Report, torch) -> None:
    home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if not home:
        report.add(
            "CUDA_HOME",
            "warn" if sys.platform == "win32" else "skip",
            "not set",
            hint=r"Triton on Windows needs it to match the torch build; tools\env.bat sets it",
            data={"cuda_home": None},
        )
        return

    build = getattr(torch.version, "cuda", None) if torch else None
    nvcc = shutil.which("nvcc")
    release = None
    if nvcc:
        try:
            output = subprocess.run(
                [nvcc, "--version"], capture_output=True, text=True, timeout=20, check=False
            ).stdout
            for token in output.split():
                if token.startswith("V") and token[1:2].isdigit():
                    release = token[1:]
                    break
        except (OSError, subprocess.SubprocessError):  # pragma: no cover - env dependent
            release = None

    matches = bool(build and release and release.startswith(build))
    detail = f"{home}"
    if release:
        detail += f" (nvcc {release}, torch built against {build})"
    report.add(
        "CUDA_HOME",
        "ok" if matches or not release else "warn",
        detail,
        hint=f"nvcc {release} does not match the torch cu{(build or '').replace('.', '')} "
        "build; Triton compiles against the wrong headers",
        data={"cuda_home": home, "nvcc": release, "torch_cuda": build},
    )


def _check_pinned_memory(report: Report, torch, has_cuda: bool) -> None:
    """Offload needs pinned host buffers; allocation can fail under memory pressure."""
    if not has_cuda:
        report.add("pinned memory", "skip", "needs a CUDA device")
        return
    try:
        buffer = torch.empty(16 * 2**20, dtype=torch.uint8, pin_memory=True)
        del buffer
        report.add("pinned memory", "ok", "16 MiB allocated and released")
    except RuntimeError as exc:
        report.add(
            "pinned memory",
            "warn",
            f"could not pin 16 MiB: {exc}",
            hint="offload still works with --no-pin-memory, but copies cannot overlap compute",
        )


def _check_h2d_bandwidth(report: Report, torch, has_cuda: bool) -> None:
    """Measured PCIe throughput -- what `spikeinfer plan` costs offload against."""
    if not has_cuda:
        report.add("host-to-device", "skip", "needs a CUDA device")
        return
    try:
        size = 64 * 2**20
        host = torch.empty(size, dtype=torch.uint8, pin_memory=True)
        device = torch.empty(size, dtype=torch.uint8, device="cuda")
        for _ in range(3):
            device.copy_(host, non_blocking=True)
        torch.cuda.synchronize()

        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        for _ in range(10):
            device.copy_(host, non_blocking=True)
        end.record()
        torch.cuda.synchronize()

        seconds = start.elapsed_time(end) / 1000 / 10
        gbps = size / seconds / 1e9
        del host, device
        report.add(
            "host-to-device",
            "ok" if gbps > 2 else "warn",
            f"{gbps:.1f} GB/s pinned",
            hint="under 2 GB/s makes weight streaming slower than it should be; "
            "check the slot is x16 and not sharing lanes",
            data={"h2d_gb_s": round(gbps, 2)},
        )
    except RuntimeError as exc:  # pragma: no cover - depends on free VRAM
        report.add("host-to-device", "warn", f"could not measure: {exc}")


def _check_kernels(report: Report, torch, has_cuda: bool) -> None:
    """Actually compile and run the kernels, rather than trusting the imports.

    This is the check that catches a Triton install which imports but cannot
    reach a compiler -- the most common broken state on Windows, and one that
    otherwise only shows up on the first real request.
    """
    from .kernels.packing import pack_spikes, unpack_spikes

    device = "cuda" if has_cuda else "cpu"
    try:
        spikes = (torch.rand(4, 8, 64, device=device) < 0.5).to(torch.float32)
        packed = pack_spikes(spikes)
        if not torch.equal(unpack_spikes(packed, 64, torch.float32), spikes):
            report.add("packing round-trip", "fail", "pack/unpack is not the identity")
        else:
            report.add("packing round-trip", "ok", f"exact on {device}")
    except Exception as exc:
        report.add("packing round-trip", "fail", f"{type(exc).__name__}: {exc}")

    try:
        from .kernels.lif_triton import _HAS_TRITON, lif_multistep, lif_multistep_ref

        current = torch.randn(4, 64, 32, device=device)
        beta = torch.full((32,), 0.9, device=device)
        threshold = torch.full((32,), 1.0, device=device)
        fused = lif_multistep(current, beta, threshold)
        expected, _ = lif_multistep_ref(current, beta, threshold)
        agree = torch.equal(fused, expected)
        path = "triton" if (_HAS_TRITON and has_cuda) else "pytorch fallback"
        report.add(
            "lif kernel",
            "ok" if agree else "fail",
            f"{path}, bit-exact vs reference" if agree else f"{path}, DIVERGES from reference",
            hint="a mismatch here silently degrades model quality -- do not ignore it",
            data={"path": path, "exact": agree},
        )
    except Exception as exc:
        report.add(
            "lif kernel",
            "fail",
            f"{type(exc).__name__}: {exc}",
            hint=r"on Windows this is usually a missing cl.exe; run tools\env.bat",
        )


def run() -> Report:
    """Every check, in order."""
    report = Report()
    _check_python(report)
    torch = _check_torch(report)
    if torch is None:
        return report
    has_cuda = _check_cuda(report, torch)
    _check_host_memory(report)
    _check_triton(report, has_cuda)
    _check_msvc(report)
    _check_cuda_home(report, torch)
    _check_pinned_memory(report, torch, has_cuda)
    _check_h2d_bandwidth(report, torch, has_cuda)
    _check_kernels(report, torch, has_cuda)
    return report


def measured_h2d_bandwidth(report: Report) -> float | None:
    """Pull the measured PCIe rate back out, for the planner to use."""
    for check in report.checks:
        if "h2d_gb_s" in check.data:
            return check.data["h2d_gb_s"]
    return None
