"""Conservative CPU-memory and disk checks for GGUF files."""

import psutil
from settings import APP_DATA_DIR


GIB = 1024 ** 3
MEMORY_MULTIPLIER = 1.3
RUNTIME_AND_CACHE_GB = 2
WINDOWS_RESERVE_GB = 4
RECOMMENDED_QUANTIZATIONS = {
    "Q4_K_M": 0,
    "Q5_K_M": 1,
    "Q4_K_S": 2,
    "Q5_K_S": 3,
    "Q6_K": 4,
    "Q8_0": 5,
    "Q4_0": 6,
    "Q5_0": 7,
}


def _format_size(size):
    return f"{size / GIB:.1f} GB" if size is not None else "Unknown"


def max_practical_gguf_size(total_ram=None):
    """Maximum file size under the existing 1.3x + 2 GiB estimate and 4 GiB reserve."""
    if total_ram is None:
        try:
            total_ram = psutil.virtual_memory().total
        except Exception:
            return 0
    safe_total = total_ram - WINDOWS_RESERVE_GB * GIB
    return max(0, int((safe_total - RUNTIME_AND_CACHE_GB * GIB) / MEMORY_MULTIPLIER))


def check_models(models):
    """Annotate variants using the conservative estimate; free RAM remains a live condition."""
    try:
        memory = psutil.virtual_memory()
        total_ram = memory.total
        available_ram = memory.available
    except Exception:
        total_ram = available_ram = None
    try:
        # Downloads use per-user app data; inspect that volume even before it exists.
        download_path = APP_DATA_DIR / "models"
        while not download_path.exists():
            download_path = download_path.parent
        available_disk = psutil.disk_usage(str(download_path)).free
    except (OSError, PermissionError):
        available_disk = None

    safe_total = total_ram - WINDOWS_RESERVE_GB * GIB if total_ram is not None else None
    checked_models = []
    for model in models:
        checked_model = {**model, "gguf_files": []}
        for variant in model.get("gguf_files", []):
            size = variant.get("size_bytes")
            if size is None or total_ram is None or available_ram is None:
                estimate = None
                status = "Unknown"
            else:
                estimate = int(size * MEMORY_MULTIPLIER + RUNTIME_AND_CACHE_GB * GIB)
                if estimate > safe_total:
                    status = "Too large"
                elif estimate > available_ram:
                    status = "Fits hardware, but insufficient RAM available now"
                else:
                    status = "Likely fits"

            if size is None or available_disk is None:
                disk_status = "Unknown"
            elif size > available_disk:
                disk_status = "Insufficient disk space"
            else:
                disk_status = "Enough disk space"

            checked_model["gguf_files"].append({
                **variant,
                "estimated_memory": _format_size(estimate),
                "estimated_memory_bytes": estimate,
                "compatibility": status,
                "disk_status": disk_status,
            })
        checked_models.append(checked_model)

    return checked_models

def is_potentially_compatible(variant):
    """Whether the variant fits memory estimates and currently available disk."""
    return (
        variant.get("compatibility") in {
            "Likely fits",
            "Fits hardware, but insufficient RAM available now",
        }
        and variant.get("disk_status") == "Enough disk space"
    )


def recommend_models(models, limit=5):
    """Choose a few distinct fit options; compatibility is checked before quality heuristics."""
    candidates = []
    for model in models:
        variants = [
            variant for variant in model.get("gguf_files", [])
            if variant.get("compatibility") in {
                "Likely fits",
                "Fits hardware, but insufficient RAM available now",
            }
            and variant.get("disk_status") == "Enough disk space"
            and variant.get("quantization", "").upper() in RECOMMENDED_QUANTIZATIONS
            and len(variant.get("files") or []) == 1
        ]
        if not variants:
            continue
        variant = min(
            variants,
            key=lambda item: (
                RECOMMENDED_QUANTIZATIONS[item["quantization"].upper()],
                item.get("estimated_memory_bytes") if item.get("estimated_memory_bytes") is not None else float("inf"),
                item.get("size_bytes") if item.get("size_bytes") is not None else float("inf"),
            ),
        )
        # Free RAM remains a live warning; it does not permanently lower a model's rank.
        parameter_count = _parameter_count(model.get("parameters"))
        candidates.append({
            "model": model,
            "variant": variant,
            "parameters": parameter_count,
            "chat_unverified": "unverified" in model.get("chat_evidence", "").lower(),
            "memory": variant.get("estimated_memory_bytes") or float("inf"),
            "size": variant.get("size_bytes") or float("inf"),
            "quant_rank": RECOMMENDED_QUANTIZATIONS[variant["quantization"].upper()],
        })

    if not candidates:
        return []

    # Quality benchmark values are not consistently comparable in current Hub metadata.
    # Use transparent parameter/memory heuristics for the three practical roles instead.
    candidates.sort(key=lambda item: (item["memory"], item["quant_rank"], item["size"]))
    chosen = []

    def model_identity(item):
        model = item["model"]
        base = model.get("base_model")
        if isinstance(base, str) and base.strip().lower() not in {"", "unknown", "none"}:
            return base.strip().lower()
        name = model.get("name")
        if isinstance(name, str) and name.strip().lower() not in {"", "unknown", "none"}:
            return name.strip().lower()
        return (model.get("gguf_repo") or "").strip().lower()

    def add(candidate, role, reason):
        if candidate is None or candidate in chosen:
            return
        model_name = model_identity(candidate)
        # Prefer a different underlying model where metadata provides a reliable identity.
        for existing in chosen:
            existing_name = model_identity(existing)
            if model_name and existing_name and (model_name == existing_name or model_name in existing_name or existing_name in model_name):
                return
        candidate["role"] = role
        candidate["role_reason"] = reason
        chosen.append(candidate)

    lightweight = candidates[0]
    add(lightweight, "Lightweight", "Lowest estimated memory use among the compatible options; likely the most responsive choice on CPU.")

    distinct = [item for item in candidates if item not in chosen]
    known_sizes = [item["parameters"] for item in distinct if item["parameters"] is not None]
    if known_sizes:
        midpoint = (min(known_sizes) + max(known_sizes)) / 2
        balanced = min(
            (item for item in distinct if item["parameters"] is not None),
            key=lambda item: abs(item["parameters"] - midpoint),
            default=None,
        )
    else:
        balanced = distinct[len(distinct) // 2] if distinct else None
    add(balanced, "Balanced", "A middle-ground compatible option, balancing model scale and estimated CPU memory needs.")

    distinct = [item for item in candidates if item not in chosen]
    capable = max(
        distinct,
        key=lambda item: (item["parameters"] is not None, item["parameters"] or item["size"], -item["memory"]),
        default=None,
    )
    add(capable, "More capable", "A larger compatible candidate that may offer more capacity; quality and speed have not been benchmarked.")

    # If a UI requests more than the three core choices, fill with different base models.
    for candidate in candidates:
        if len(chosen) >= limit:
            break
        add(candidate, "Alternative", "Another compatible option; no comparable quality benchmark is available.")

    recommendations = []
    for candidate in chosen[:limit]:
        model, variant = candidate["model"], candidate["variant"]
        if variant["compatibility"] == "Likely fits":
            memory_reason = "The conservative estimate fits total RAM and currently available RAM."
        else:
            memory_reason = "It fits the total-RAM estimate, but free RAM is low now; close other apps before loading."
        quant_reason = f"{variant['quantization']} is a practical moderate quantization; smaller files generally reduce CPU work."
        chat_reason = (
            "Chat/instruction tuning is unverified; the Hub only marks this repository conversational."
            if candidate["chat_unverified"] else "Hub metadata identifies this as an instruction/chat model."
        )
        if model.get("evaluation_entries"):
            quality_reason = (
                "The Hub lists evaluation entries, but no comparable benchmark values were verified."
            )
        else:
            quality_reason = (
                "No comparable quality benchmark was verified; this role uses practical-fit heuristics, not a quality score."
            )
        recommendations.append({
            "model": model,
            "variant": variant,
            "status": variant["compatibility"],
            "role": candidate["role"],
            "reasons": [candidate["role_reason"], memory_reason, quant_reason, quality_reason, chat_reason],
        })
    return recommendations


def _parameter_count(value):
    """Parse only explicit parameter counts; do not infer them from repository names."""
    if isinstance(value, (int, float)):
        return int(value)
    if not isinstance(value, str):
        return None
    normalized = value.strip().replace(",", "")
    try:
        return int(normalized)
    except ValueError:
        pass
    import re
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([BMK])", normalized, re.I)
    if not match:
        return None
    multiplier = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}[match.group(2).upper()]
    return int(float(match.group(1)) * multiplier)


def compatibility_summary(models):
    """Count distinct models that pass total-RAM, disk, quantization and single-file checks."""
    summary = {
        "hardware_compatible_models": 0,
        "insufficient_ram_available_now": 0,
        "hardware_rejections": {},
    }
    rejected = summary["hardware_rejections"]
    for model in models:
        variants = model.get("gguf_files", [])
        if not variants:
            rejected["no_standalone_variant"] = rejected.get("no_standalone_variant", 0) + 1
            continue
        single_file = [variant for variant in variants if len(variant.get("files") or []) == 1]
        if not single_file:
            rejected["multi_shard_only"] = rejected.get("multi_shard_only", 0) + 1
            continue
        practical = [
            variant for variant in single_file
            if variant.get("quantization", "").upper() in RECOMMENDED_QUANTIZATIONS
        ]
        if not practical:
            rejected["no_practical_quantization"] = rejected.get("no_practical_quantization", 0) + 1
            continue
        known_size = [variant for variant in practical if variant.get("size_bytes") is not None]
        if not known_size:
            rejected["file_size_unknown"] = rejected.get("file_size_unknown", 0) + 1
            continue
        disk_ok = [variant for variant in known_size if variant.get("disk_status") == "Enough disk space"]
        if not disk_ok:
            rejected["insufficient_disk_space"] = rejected.get("insufficient_disk_space", 0) + 1
            continue
        fits_total = [
            variant for variant in disk_ok
            if variant.get("compatibility") in {
                "Likely fits", "Fits hardware, but insufficient RAM available now"
            }
        ]
        if not fits_total:
            rejected["exceeds_total_ram_estimate"] = rejected.get("exceeds_total_ram_estimate", 0) + 1
            continue
        summary["hardware_compatible_models"] += 1
        if all(variant.get("compatibility") == "Fits hardware, but insufficient RAM available now" for variant in fits_total):
            summary["insufficient_ram_available_now"] += 1
    return summary
