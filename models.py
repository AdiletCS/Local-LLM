"""Find public GGUF versions of instruction/chat models on Hugging Face."""

import json
import os
import re
from pathlib import Path
import time

from huggingface_hub import HfApi
from settings import APP_DATA_DIR


DEFAULT_CANDIDATE_LIMIT = 100
MAX_CANDIDATE_LIMIT = 200
METADATA_CACHE = APP_DATA_DIR / "huggingface_metadata.json"
METADATA_CACHE_TTL = 24 * 60 * 60
METADATA_CACHE_LIMIT = 500

def _value(card, name):
    if isinstance(card, dict):
        return card.get(name)
    return getattr(card, name, None)


def _base_model(info):
    """Use only the Hub's explicit quantized-from relationship."""
    relation = getattr(info, "base_models", None)
    if isinstance(relation, dict) and relation.get("relation") == "quantized":
        models = relation.get("models") or []
        if len(models) == 1 and models[0].get("id"):
            return models[0]["id"]
    for tag in getattr(info, "tags", None) or []:
        if tag.startswith("base_model:quantized:"):
            return tag.split(":", 2)[2]
    return None


# Exclude only model types that are not ordinary text-generation models.
# Coding/uncensored labels describe specialization or policy, not incompatibility.
SPECIALIZED_TERMS = (
    "embedding", "reranker", "rerank", "speech", "audio",
    "vision", "vl", "classification", "image-text-to-text",
    "text-rewriting", "rewriting", "paraphrase", "style-transfer", "humanizer",
)


def _model_tags(info):
    tags = [tag.lower() for tag in (getattr(info, "tags", None) or [])]
    tags.extend(tag.lower() for tag in (_value(getattr(info, "card_data", None), "tags") or []))
    return tags


def _is_specialized_model(info, base_model=None):
    name = f"{base_model or ''} {getattr(info, 'id', '')}".lower()
    tags = _model_tags(info)
    pipeline = getattr(info, "pipeline_tag", None)
    has_specialized_term = lambda text: any(
        re.search(rf"(?:^|[/_.\-\s]){re.escape(term)}(?:$|[/_.\-\s])", text)
        for term in SPECIALIZED_TERMS
    )
    return (
        (pipeline is not None and pipeline not in {"text-generation", "conversational"})
        or any(has_specialized_term(tag) for tag in tags)
        or has_specialized_term(name)
    )


def _is_instruction_model(info, base_model, base_info=None):
    if _is_specialized_model(info, base_model):
        return False
    name = f"{base_model or ''} {getattr(info, 'id', '')}".lower()
    tags = _model_tags(info)
    if base_info:
        tags.extend(_model_tags(base_info))
    instruction_name = bool(
        re.search(r"(?:^|[/_.-])(?:instruct(?:ion)?|chat)(?:$|[_.-])", name)
        or re.search(r"(?:^|[-_])it(?:[-_0-9]|$)", name)
    )
    return instruction_name or any(
        tag in {"chat", "instruction", "instruct"}
        for tag in tags
    )


def is_standalone_gguf_file(filename):
    """Reject GGUF sidecars that are not standalone text-generation models."""
    path = Path(filename)
    name = path.stem
    if path.name == "model.gguf":
        try:
            metadata = json.loads((path.parent / "model.json").read_text(encoding="utf-8"))
            if isinstance(metadata, dict):
                name = Path(metadata.get("filename", path.name)).stem
        except (OSError, ValueError, TypeError):
            pass
    name = name.lower()
    auxiliary = (
        "mmproj", "projector", "vision-encoder", "vision_encoder", "image-encoder",
        "image_encoder", "clip", "adapter", "lora", "imatrix", "tokenizer",
        "vocab", "embedding", "mtp", "draft-model", "draft_model",
    )
    return not any(re.search(rf"(?:^|[-_.]){re.escape(term)}(?:$|[-_.])", name) for term in auxiliary)


def _format_size(size):
    return f"{size / (1024 ** 3):.2f} GB" if size is not None else "Unknown"


def _quantization(filename):
    match = re.search(r"(?:^|[-_.])(IQ\d+(?:_[A-Z0-9]+)*|Q\d+(?:_[A-Z0-9]+)*|BF16|F16|F32)(?=[-_.]|$)", filename, re.I)
    return match.group(1).upper() if match else "Unknown"


def _gguf_variants(files):
    """Combine the file sizes of shards that belong to one GGUF variant."""
    variants = {}
    for file in files:
        if not is_standalone_gguf_file(file.rfilename):
            continue
        stem = re.sub(r"[-_.]\d{5}-of-\d{5}$", "", Path(file.rfilename).stem, flags=re.I)
        variant = variants.setdefault(stem, {
            "name": stem,
            "quantization": _quantization(stem),
            "size_bytes": 0,
            "files": [],
        })
        variant["files"].append(file.rfilename)
        file_size = getattr(file, "size", None)
        if file_size is None:
            variant["size_bytes"] = None
        elif variant["size_bytes"] is not None:
            variant["size_bytes"] += file_size

    return [
        {**variant, "size": _format_size(variant["size_bytes"])}
        for variant in variants.values()
    ]


def _config_value(config, names):
    if not isinstance(config, dict):
        return None
    for section in (config, config.get("text_config", {})):
        if isinstance(section, dict):
            for name in names:
                value = section.get(name)
                if isinstance(value, (int, float)) and value > 0:
                    return value
    return None


def _parameter_count(config, *model_infos):
    value = _config_value(config, ("num_parameters", "n_parameters"))
    if value:
        return f"{value:,}"
    for model_info in model_infos:
        if model_info is None:
            continue
        value = _config_value(getattr(model_info, "config", None), ("num_parameters", "n_parameters"))
        if value:
            return f"{value:,}"
        parameter_sizes = _value(getattr(model_info, "safetensors", None), "parameters")
        if isinstance(parameter_sizes, dict):
            counts = [count for count in parameter_sizes.values() if isinstance(count, (int, float))]
            if counts:
                return f"{int(sum(counts)):,}"
        for tag in getattr(model_info, "tags", None) or []:
            match = re.fullmatch(r"params[-_](\d+(?:\.\d+)?[BMK])", tag, re.I)
            if match:
                return match.group(1).upper()
    return "Unknown"


def _context_length(config):
    value = _config_value(
        config,
        ("max_position_embeddings", "max_sequence_length", "max_seq_len", "n_positions", "model_max_length"),
    )
    return f"{value:,} tokens" if value and value < 10_000_000 else "Unknown"


def _license(info, quantized_info):
    for model_info in (info, quantized_info):
        if model_info is None:
            continue
        license_name = _value(getattr(model_info, "card_data", None), "license")
        if license_name:
            return license_name
        for tag in getattr(model_info, "tags", None) or []:
            if tag.startswith("license:"):
                return tag.split(":", 1)[1]
    return "Unknown"


def _chat_evidence(info, base_model=None):
    """Return a concise Hub signal, preserving uncertainty instead of guessing."""
    if _is_instruction_model(info, base_model):
        return "Hub name or tags identify an instruction/chat model."
    if not _is_specialized_model(info, base_model):
        pipeline = getattr(info, "pipeline_tag", None)
        if "conversational" in _model_tags(info) or pipeline == "conversational":
            return "Hub metadata marks this as conversational; instruction tuning is unverified."
    return None


def _load_metadata_cache():
    try:
        value = json.loads(METADATA_CACHE.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def _cached_model(cache, repo_id):
    entry = cache.get(repo_id.lower())
    if not isinstance(entry, dict):
        return None
    try:
        age = time.time() - float(entry.get("cached_at", 0))
    except (TypeError, ValueError):
        return None
    if age > METADATA_CACHE_TTL:
        return None
    model = entry.get("model")
    return model if isinstance(model, dict) else None


def _save_metadata_cache(cache):
    now = time.time()
    fresh = {}
    for key, value in cache.items():
        if not isinstance(value, dict):
            continue
        try:
            if now - float(value.get("cached_at", 0)) <= METADATA_CACHE_TTL:
                fresh[key] = value
        except (TypeError, ValueError):
            continue
    if len(fresh) > METADATA_CACHE_LIMIT:
        fresh = dict(sorted(fresh.items(), key=lambda pair: float(pair[1]["cached_at"]))[-METADATA_CACHE_LIMIT:])
    try:
        APP_DATA_DIR.mkdir(parents=True, exist_ok=True)
        temporary = METADATA_CACHE.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(fresh, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, METADATA_CACHE)
    except OSError:
        # Metadata caching is optional; a read-only profile should still discover models.
        pass


def _matches_conversion_slug(source_id, candidate_id):
    source = source_id.rsplit("/", 1)[-1].lower()
    candidate = candidate_id.rsplit("/", 1)[-1].lower()
    if candidate == source:
        return True
    if not candidate.startswith(source):
        return False
    suffix = candidate[len(source):]
    return bool(re.match(r"^[-_.](?:gguf|q\d|f16|bf16|instruct|chat|it)(?:$|[-_.0-9])", suffix))


def _model_info_id(info):
    return getattr(info, "id", None)


def public_gguf_repository_count():
    """Return an exact API total when the Hub provides one; otherwise return None."""
    from huggingface_hub.utils import get_session

    response = get_session().get(
        "https://huggingface.co/api/models",
        params={
            "filter": "gguf",
            "gated": "false",
            "limit": 1,
        },
        timeout=15,
    )
    response.raise_for_status()
    value = response.headers.get("X-Total-Count") or response.headers.get("x-total-count")
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def find_models(candidate_limit=DEFAULT_CANDIDATE_LIMIT, return_stats=False, progress=None):
    """Inspect a bounded, diverse set of public GGUF repositories without downloading weights."""
    from collections import Counter
    import time

    started = time.perf_counter()
    api = HfApi()
    try:
        limit = max(1, min(int(candidate_limit), MAX_CANDIDATE_LIMIT))
    except (TypeError, ValueError):
        limit = DEFAULT_CANDIDATE_LIMIT
    repository_count = None
    count_error = None
    try:
        repository_count = public_gguf_repository_count()
    except Exception as error:
        count_error = str(error)

    expand = [
        "siblings", "tags", "cardData", "baseModels", "lastModified", "gated",
        "downloads", "likes", "evalResults", "pipeline_tag",
    ]
    search_errors = {}

    def search(label, **parameters):
        results = []
        try:
            for item in api.list_models(**parameters):
                results.append(item)
                if progress and (len(results) == 1 or len(results) % 10 == 0 or len(results) == parameters.get("limit")):
                    progress(f"Found {len(results)} {label} candidates…")
        except Exception as error:
            search_errors[label] = str(error)
        return results

    if progress:
        progress(
            f"Searching up to {limit} repositories using downloads, trending, text-generation, and instruction signals…"
        )
    # HfApi's iterator follows Hub pagination; each source is bounded by the requested pool size.
    metadata_search_limit = max(1, (limit + 2) // 3)
    sources = [
        ("popular GGUF", search(
            "popular GGUF", filter="gguf", gated=False, sort="downloads", limit=limit,
            expand=expand, token=False,
        )),
        ("trending GGUF", search(
            "trending GGUF", filter="gguf", gated=False, sort="trending_score", limit=metadata_search_limit,
            expand=expand, token=False,
        )),
        ("text-generation GGUF", search(
            "text-generation GGUF", filter="gguf", pipeline_tag="text-generation", gated=False,
            sort="downloads", limit=metadata_search_limit, expand=expand, token=False,
        )),
        ("instruction-tuned GGUF", search(
            "instruction-tuned GGUF", filter="gguf", search="instruct", gated=False,
            sort="downloads", limit=metadata_search_limit, expand=expand, token=False,
        )),
    ]

    # Use a few popular, publicly tagged text-generation originals as dynamic search seeds.
    # For each seed, look up its most downloaded GGUF conversions; no publishers or families are fixed.
    seed_limit = min(8, max(1, (limit + 19) // 20))
    base_seeds = search(
        "text-generation source models", pipeline_tag="text-generation", gated=False,
        sort="downloads", limit=seed_limit,
        expand=["tags", "cardData", "baseModels", "gated", "downloads", "likes", "pipeline_tag"],
        token=False,
    )
    conversions = []
    seed_evidence = {}
    ambiguous_seed_repositories = set()
    for seed in base_seeds:
        if _chat_evidence(seed) is None or _is_specialized_model(seed):
            continue
        seed_id = _model_info_id(seed)
        if not seed_id:
            continue
        slug = seed_id.rsplit("/", 1)[-1]
        try:
            found = list(api.list_models(
                filter="gguf", search=slug, gated=False, sort="downloads", limit=min(2, limit),
                expand=expand, token=False,
            ))
        except Exception as error:
            search_errors[f"GGUF conversions for {seed_id}"] = str(error)
            continue
        if progress:
            progress(f"Checking GGUF conversions for {seed_id}…")
        for item in found:
            relation = _base_model(item)
            # Only inherit source-model identity/chat evidence when the explicit Hub
            # relation matches or the full source slug is present in the conversion id.
            if relation and relation.lower() == seed_id.lower():
                linked = True
            else:
                linked = _matches_conversion_slug(seed_id, item.id)
            if linked:
                key = item.id.lower()
                existing_seed = seed_evidence.get(key)
                if existing_seed and existing_seed["base_model"].lower() != seed_id.lower():
                    ambiguous_seed_repositories.add(key)
                    seed_evidence.pop(key, None)
                elif key not in ambiguous_seed_repositories:
                    seed_evidence[key] = {"base_model": seed_id, "evidence": _chat_evidence(seed)}
            conversions.append(item)
    sources.append(("popular model conversions", conversions))

    # Interleave all sources and inspect no more unique GGUF repositories than requested.
    pool = []
    source_indexes = [0] * len(sources)
    seen_repositories = set()
    duplicate_search_repositories = 0
    while len(pool) < limit:
        progressed = False
        for source_index, (_label, results) in enumerate(sources):
            index = source_indexes[source_index]
            if index >= len(results):
                continue
            source_indexes[source_index] += 1
            item = results[index]
            key = item.id.lower()
            if key in seen_repositories:
                duplicate_search_repositories += 1
                continue
            seen_repositories.add(key)
            pool.append(item)
            progressed = True
            if len(pool) >= limit:
                break
        if not progressed:
            break
    if not pool:
        details = "; ".join(f"{name}: {message}" for name, message in search_errors.items())
        raise RuntimeError("Could not retrieve public GGUF models from Hugging Face." + (f" {details}" if details else ""))

    rejections = Counter()
    candidates_by_base = {}
    for index, item in enumerate(pool, 1):
        if progress:
            progress(f"Screening repository {index} of {len(pool)}…")
        seed = seed_evidence.get(item.id.lower())
        base_model = _base_model(item) or (seed["base_model"] if seed else None)
        identity = (base_model or item.id).lower()
        siblings = item.siblings or []
        gguf_files = [file.rfilename for file in siblings if file.rfilename.lower().endswith(".gguf")]
        if not gguf_files:
            rejections["missing_gguf_file"] += 1
            continue
        standalone_files = [filename for filename in gguf_files if is_standalone_gguf_file(filename)]
        if not standalone_files:
            rejections["auxiliary_gguf_only"] += 1
            continue
        if _is_specialized_model(item, base_model):
            rejections["clearly_specialized"] += 1
            continue
        evidence = _chat_evidence(item, base_model) or (seed["evidence"] if seed else None)
        if evidence is None:
            rejections["no_chat_signal"] += 1
            continue
        candidate = {
            "info": item,
            "base_model": base_model,
            "identity": identity,
            "standalone_files": standalone_files,
            "chat_evidence": evidence,
        }
        previous = candidates_by_base.get(identity)
        if previous is None:
            candidates_by_base[identity] = candidate
        else:
            rejections["duplicate_base_model"] += 1
            # Prefer a repository that has a moderate quantization filename; retain
            # the earlier (more popular/trending) repo when the choices are equivalent.
            preferred = {"Q4_K_M": 0, "Q5_K_M": 1, "Q4_K_S": 2, "Q5_K_S": 3, "Q6_K": 4, "Q8_0": 5}
            def quant_rank(entry):
                ranks = [
                    preferred.get(_quantization(Path(filename).stem), 99)
                    for filename in entry["standalone_files"]
                ]
                return min(ranks, default=99)
            if quant_rank(candidate) < quant_rank(previous):
                candidates_by_base[identity] = candidate

    distinct_candidates = list(candidates_by_base.values())
    models = []
    metadata_unavailable = Counter()
    detail_requests = 0
    metadata_cache = _load_metadata_cache()
    cache_hits = 0
    for index, candidate in enumerate(distinct_candidates, 1):
        if progress:
            progress(f"Inspecting model metadata {index} of {len(distinct_candidates)}…")
        base_model = candidate["base_model"]
        gguf_repo = candidate["info"].id
        cached = _cached_model(metadata_cache, gguf_repo)
        if cached:
            models.append(cached)
            cache_hits += 1
            continue
        try:
            quantized_info = api.model_info(gguf_repo, files_metadata=True, token=False)
            detail_requests += 1
            files = [file for file in (quantized_info.siblings or []) if file.rfilename.lower().endswith(".gguf")]
            if not files:
                files = [
                    file for file in (candidate["info"].siblings or [])
                    if file.rfilename.lower().endswith(".gguf")
                ]
                metadata_unavailable["gguf_file_metadata_unavailable"] += 1
        except Exception:
            quantized_info = None
            files = [
                file for file in (candidate["info"].siblings or [])
                if file.rfilename.lower().endswith(".gguf")
            ]
            metadata_unavailable["gguf_file_metadata_unavailable"] += 1

        base_info = None
        if base_model:
            try:
                base_info = api.model_info(
                    base_model, expand=["cardData", "config", "tags", "safetensors"], token=False
                )
                detail_requests += 1
            except Exception:
                metadata_unavailable["base_model_metadata_unavailable"] += 1

        config = getattr(base_info, "config", None) if base_info else {}
        quantized_card = getattr(quantized_info, "card_data", None)
        base_card = getattr(base_info, "card_data", None) if base_info else None
        display_name = (
            _value(base_card, "model_name")
            or _value(quantized_card, "model_name")
            or base_model
            or gguf_repo
        )
        model_record = {
            "name": display_name,
            "base_model": base_model or "Unknown",
            "gguf_repo": gguf_repo,
            "chat_evidence": candidate["chat_evidence"],
            "parameters": _parameter_count(config, base_info, quantized_info),
            "context_length": _context_length(config),
            "license": _license(base_info, quantized_info),
            "downloads": getattr(candidate["info"], "downloads", None),
            "likes": getattr(candidate["info"], "likes", None),
            "evaluation_entries": len(getattr(candidate["info"], "eval_results", None) or []) or None,
            "gguf_files": _gguf_variants(files),
        }
        models.append(model_record)
        metadata_cache[gguf_repo.lower()] = {"cached_at": time.time(), "model": model_record}
    _save_metadata_cache(metadata_cache)

    stats = {
        "repositories_inspected": len(pool),
        "candidate_limit": limit,
        "detailed_candidate_repositories": len(distinct_candidates),
        "detailed_metadata_requests": detail_requests,
        "metadata_cache_hits": cache_hits,
        "public_gguf_repository_count": repository_count,
        "repository_count_error": count_error,
        "trending_search_error": search_errors.get("trending GGUF"),
        "search_errors": search_errors,
        "duplicate_search_repositories": duplicate_search_repositories,
        "distinct_models_found": len(models),
        "rejections": dict(rejections),
        "metadata_unavailable": dict(metadata_unavailable),
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }
    return (models, stats) if return_stats else models
