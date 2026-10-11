"""Lightweight hardware detection for local LLM workloads."""

import platform
import subprocess
from settings import APP_DATA_DIR

import psutil

try:
    import cpuinfo
except ImportError:
    cpuinfo = None


def _format_bytes(value):
    """Format a byte count using readable binary units."""
    if value is None:
        return "Unavailable"
    amount = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if amount < 1024 or unit == "TB":
            return f"{amount:.1f} {unit}" if unit != "B" else f"{amount:.0f} {unit}"
        amount /= 1024


def _windows_version():
    """Use the Windows build to distinguish Windows 11 from legacy product names."""
    try:
        result = subprocess.run(
            [
                "reg", "query",
                r"HKLM\SOFTWARE\Microsoft\Windows NT\CurrentVersion",
                "/v", "CurrentBuildNumber",
            ],
            capture_output=True, text=True, timeout=3, check=True,
        )
        build = int(next(
            line.split("REG_SZ", 1)[1].strip() for line in result.stdout.splitlines()
            if "CurrentBuildNumber" in line and "REG_SZ" in line
        ))
        display_version_result = subprocess.run(
            ["reg", "query", r"HKLM\SOFTWARE\Microsoft\Windows NT\CurrentVersion", "/v", "DisplayVersion"],
            capture_output=True, text=True, timeout=3, check=True,
        )
        display_version = next(
            (line.split("REG_SZ", 1)[1].strip() for line in display_version_result.stdout.splitlines()
             if "DisplayVersion" in line and "REG_SZ" in line),
            "",
        )
        windows_name = "Windows 11" if build >= 22000 else "Windows 10"
        return f"{windows_name} {display_version}".strip()
    except (OSError, subprocess.SubprocessError, IndexError, ValueError, StopIteration):
        return f"Windows {platform.release()}"


def _operating_system():
    if platform.system() == "Windows":
        return _windows_version()
    return f"{platform.system()} {platform.release()}"


def _cpu_details():
    info = {}
    if cpuinfo is not None:
        try:
            info = cpuinfo.get_cpu_info()
        except Exception:
            pass

    flags = set(info.get("flags") or [])
    cpu_name = info.get("brand_raw") or platform.processor() or "Unknown"
    if platform.system() == "Windows" and ("Family" in cpu_name or cpu_name in {"AMD64", "x86_64"}):
        try:
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name)"],
                capture_output=True, text=True, timeout=5, check=True,
            )
            cpu_name = result.stdout.strip() or cpu_name
        except (OSError, subprocess.SubprocessError):
            pass
    instruction_support = [name for name in ("AVX", "AVX2") if name.lower() in flags]
    if not instruction_support:
        instruction_support = ["Not reported"]

    try:
        physical_cores = psutil.cpu_count(logical=False) or "Unavailable"
        logical_cores = psutil.cpu_count(logical=True) or "Unavailable"
    except Exception:
        physical_cores = logical_cores = "Unavailable"
    return {
        "Model": cpu_name,
        "Physical cores": str(physical_cores),
        "Logical cores": str(logical_cores),
        "AVX / AVX2": ", ".join(instruction_support),
    }


def _gpu_details():
    """Detect display adapters; Windows CIM works without vendor-specific tools."""
    if platform.system() == "Windows":
        try:
            result = subprocess.run(
                [
                    "powershell", "-NoProfile", "-Command",
                    "Get-CimInstance -ClassName Win32_VideoController | "
                    "Select-Object Name,AdapterRAM | ConvertTo-Json -Compress",
                ],
                capture_output=True, text=True, timeout=5, check=True,
            )
            import json

            adapters = json.loads(result.stdout.strip()) if result.stdout.strip() else []
            if isinstance(adapters, dict):
                adapters = [adapters]
            gpus, memories = [], []
            for adapter in adapters or []:
                name = adapter.get("Name")
                if not name:
                    continue
                gpus.append(name)
                memory = adapter.get("AdapterRAM")
                if memory and memory > 0:
                    memories.append(f"{name}: {_format_bytes(memory)} reported")
            if gpus:
                cuda = "Unavailable"
                if any("nvidia" in gpu.lower() for gpu in gpus):
                    try:
                        subprocess.run(
                            ["nvidia-smi", "-L"], capture_output=True, text=True,
                            timeout=3, check=True,
                        )
                        cuda = "Available (NVIDIA GPU detected)"
                    except (OSError, subprocess.SubprocessError):
                        pass
                return {
                    "GPU": "; ".join(gpus),
                    "GPU memory": "; ".join(memories) if memories else "Unavailable",
                    "CUDA": cuda,
                }
        except (OSError, subprocess.SubprocessError, ValueError, TypeError):
            pass

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=3,
            check=True,
        )
        gpus = []
        for line in result.stdout.splitlines():
            parts = [part.strip() for part in line.split(",", 1)]
            if len(parts) == 2:
                gpus.append(f"{parts[0]} ({parts[1]} MiB VRAM)")
            elif parts and parts[0]:
                gpus.append(parts[0])
        if not gpus:
            return {"GPU": "GPU not detected", "GPU memory": "Unavailable", "CUDA": "Unavailable"}
        return {
            "GPU": "; ".join(line.split(",", 1)[0].strip() for line in gpus),
            "GPU memory": "; ".join(
                f"{line.split(',', 1)[0].strip()}: {line.split(',', 1)[1].strip()} MiB"
                for line in gpus if "," in line
            ) or "Unavailable",
            "CUDA": "Available (NVIDIA GPU detected)",
        }
    except (OSError, subprocess.SubprocessError):
        return {"GPU": "GPU not detected", "GPU memory": "Unavailable", "CUDA": "Unavailable"}


def detect_hardware():
    """Return hardware details grouped for display in the desktop UI."""
    try:
        memory = psutil.virtual_memory()
    except Exception:
        memory = None
    try:
        # Model downloads are per-user, so report the destination volume.
        disk_path = APP_DATA_DIR / "models"
        while not disk_path.exists():
            disk_path = disk_path.parent
        disk = psutil.disk_usage(str(disk_path))
        storage = _format_bytes(disk.free)
    except (OSError, PermissionError):
        storage = "Unavailable"

    return {
        "System": {
            "Operating system": _operating_system(),
            "CPU architecture": platform.machine() or "Unknown",
        },
        "CPU": _cpu_details(),
        "Memory": {
            "Total RAM": _format_bytes(memory.total) if memory else "Unavailable",
            "Available RAM": _format_bytes(memory.available) if memory else "Unavailable",
        },
        "GPU": _gpu_details(),
        "Storage": {"Available disk space": storage},
    }
