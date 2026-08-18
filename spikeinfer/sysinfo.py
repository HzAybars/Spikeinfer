"""Host memory and CPU facts the engine needs but torch does not expose.

``torch`` reports GPU memory (``torch.cuda.mem_get_info``) and nothing about the
host, yet three places here have to size themselves against host RAM: the CPU KV
cache (:meth:`spikeinfer.engine.llm_engine.LLMEngine._size_cache`), the offload
planner (:mod:`spikeinfer.placement`) and ``spikeinfer doctor``.

``psutil`` is the obvious answer but is not worth a hard dependency for two
numbers, so it is used when present and the platform is asked directly when it
is not. Every function degrades to a conservative guess rather than raising --
a wrong RAM reading should shrink a cache, never crash a server.
"""
from __future__ import annotations

import os
import sys

import torch

_FALLBACK_TOTAL = 8 * 2**30
"""Assumed RAM when nothing can be measured. Deliberately small: under-sizing a
cache costs throughput, over-sizing it invites the OOM killer."""


def _psutil_memory() -> tuple[int, int] | None:
    try:
        import psutil
    except ImportError:
        return None
    vm = psutil.virtual_memory()
    return int(vm.total), int(vm.available)


def _windows_memory() -> tuple[int, int] | None:  # pragma: no cover - platform specific
    if sys.platform != "win32":
        return None
    import ctypes
    from ctypes import wintypes

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", wintypes.DWORD),
            ("dwMemoryLoad", wintypes.DWORD),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return int(status.ullTotalPhys), int(status.ullAvailPhys)


def _linux_memory() -> tuple[int, int] | None:  # pragma: no cover - platform specific
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            fields = {}
            for line in fh:
                key, _, rest = line.partition(":")
                fields[key] = int(rest.split()[0]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    total = fields.get("MemTotal")
    if total is None:
        return None
    # MemAvailable is the kernel's own estimate of what a new allocation can get,
    # which is what we want -- MemFree excludes reclaimable page cache.
    return total, fields.get("MemAvailable", fields.get("MemFree", total // 2))


def _posix_sysconf_memory() -> tuple[int, int] | None:  # pragma: no cover - platform specific
    try:
        total = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        available = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, ValueError, OSError):
        return None
    return int(total), int(available)


def host_memory() -> tuple[int, int]:
    """``(total_bytes, available_bytes)`` for host RAM, never raising."""
    for probe in (_psutil_memory, _windows_memory, _linux_memory, _posix_sysconf_memory):
        try:
            result = probe()
        except Exception:  # pragma: no cover - a broken probe must not be fatal
            result = None
        if result and result[0] > 0:
            total, available = result
            return total, max(0, min(available, total))
    return _FALLBACK_TOTAL, _FALLBACK_TOTAL // 2


def total_ram_bytes() -> int:
    return host_memory()[0]


def available_ram_bytes() -> int:
    return host_memory()[1]


def cpu_count() -> int:
    """Physical-ish core count: ``os.cpu_count()`` minus SMT when psutil knows."""
    try:
        import psutil

        physical = psutil.cpu_count(logical=False)
        if physical:
            return int(physical)
    except ImportError:
        pass
    return os.cpu_count() or 1


def format_bytes(n: float) -> str:
    """Human-readable size, used by every diagnostic that prints one."""
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or unit == "TiB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} TiB"  # pragma: no cover - unreachable, loop returns first


def cuda_is_usable() -> bool:
    """Whether a CUDA device can actually be used.

    Not ``torch.cuda.is_available()`` on its own: with ``CUDA_VISIBLE_DEVICES=""``
    that returns True while ``device_count()`` is 0 (observed on torch
    2.6.0+cu124/Windows), so anything that trusts it walks into "Invalid device
    id" on the first ``get_device_properties``. Masking the devices is the
    standard way to force a CPU run, and it has to work.
    """
    try:
        return torch.cuda.is_available() and torch.cuda.device_count() > 0
    except Exception:  # pragma: no cover - a broken driver is not our problem
        return False
