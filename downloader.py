"""Download, migrate, and safely manage standalone GGUF models."""

import hashlib
import json
import re
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import hf_hub_download
from tqdm import tqdm

from models import is_standalone_gguf_file
from settings import APP_DATA_DIR


MODELS_DIR = APP_DATA_DIR / "models"
APP_DIRECTORY = Path(__file__).resolve().parent
LEGACY_MODELS_DIR = APP_DIRECTORY / "models"
MODEL_FILENAME = "model.gguf"
METADATA_FILENAME = "model.json"


def _model_roots():
    """New per-user storage plus the previous app-folder location for existing models."""
    roots = [MODELS_DIR]
    if LEGACY_MODELS_DIR.resolve() != MODELS_DIR.resolve():
        roots.append(LEGACY_MODELS_DIR)
    return [root for root in roots if not root.is_symlink()]


def read_model_metadata(path):
    """Read metadata for one app-managed model file, if present."""
    path = Path(path)
    try:
        metadata_path = path.parent / METADATA_FILENAME
        if metadata_path.is_symlink():
            return None
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if isinstance(metadata, dict) and metadata.get("managed_by") == "local-llm" and metadata.get("repo_id"):
            return metadata
    except (OSError, ValueError, TypeError):
        pass
    return None


def model_display_name(path):
    metadata = read_model_metadata(path)
    original_name = metadata.get("filename") if metadata else None
    return Path(original_name).stem if original_name else Path(path).stem


def _safe_folder_name(filename):
    name = Path(filename).stem
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip(" .-_")[:90]
    if not name or name.upper() in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}:
        name = f"model-{name}" if name else "model"
    return name


def _source_key(repo_id, filename):
    return hashlib.sha256(f"{repo_id}\n{filename}".encode("utf-8")).hexdigest()


def _metadata_matches(path, repo_id, filename):
    metadata = read_model_metadata(path / MODEL_FILENAME)
    return bool(metadata and metadata.get("repo_id") == repo_id and metadata.get("filename") == filename)


def _choose_model_folder(repo_id, filename, root=None):
    """Choose a readable unique folder; add a short hash only on collision."""
    base = _safe_folder_name(filename)
    key = _source_key(repo_id, filename)
    candidate_names = [base] + [f"{base}-{key[:length]}" for length in (8, 12, 16, 24, 64)]
    for name in candidate_names:
        folder = (root or MODELS_DIR) / name
        if not folder.exists() or _metadata_matches(folder, repo_id, filename):
            return folder
    raise ValueError(f"Could not find a safe folder name for {filename}.")


def migrate_legacy_models():
    """Move old models/<publisher>/<repo>/<file>.gguf downloads in place."""
    migrated = []
    for root in _model_roots():
        if not root.exists():
            continue
        for source in list(root.rglob("*.gguf")):
            try:
                relative = source.relative_to(root)
                if len(relative.parts) != 3 or source.name == MODEL_FILENAME or source.is_symlink():
                    continue
                if source.parent.is_symlink() or source.parent.parent.is_symlink():
                    continue
                if not source.is_file() or not is_standalone_gguf_file(source.name):
                    continue
                with source.open("rb") as model_file:
                    if model_file.read(4) != b"GGUF":
                        continue

                publisher, repository, filename = relative.parts
                repo_id = f"{publisher}/{repository}"
                folder = _choose_model_folder(repo_id, filename, root)
                target = folder / MODEL_FILENAME
                if _metadata_matches(folder, repo_id, filename) and target.is_file():
                    continue
                if folder.exists():
                    continue

                folder.mkdir(parents=True)
                metadata = {
                    "managed_by": "local-llm",
                    "format_version": 1,
                    "repo_id": repo_id,
                    "filename": filename,
                    "size_bytes": source.stat().st_size,
                }
                try:
                    shutil.move(str(source), str(target))
                    (folder / METADATA_FILENAME).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
                except OSError:
                    if target.exists() and not source.exists():
                        shutil.move(str(target), str(source))
                    if folder.exists() and not any(folder.iterdir()):
                        folder.rmdir()
                    continue

                migrated.append(target)
                for parent in (source.parent, source.parent.parent):
                    try:
                        parent.rmdir()
                    except OSError:
                        pass
            except (OSError, ValueError):
                continue
        _preserve_legacy_hub_caches(root)
    return migrated


def _preserve_legacy_hub_caches(root):
    """Move old Hugging Face local-dir caches aside without discarding them."""
    for publisher in list(root.iterdir()):
        if not publisher.is_dir() or publisher.is_symlink() or publisher.name.startswith("."):
            continue
        for repository in list(publisher.iterdir()):
            if not repository.is_dir() or repository.is_symlink():
                continue
            cache = repository / ".cache"
            hub_cache = cache / "huggingface"
            if {item.name for item in repository.iterdir()} != {".cache"}:
                continue
            if not (hub_cache / "CACHEDIR.TAG").is_file() or not (hub_cache / ".gitignore").is_file():
                continue
            archived_cache = root / ".legacy-cache" / publisher.name / repository.name / ".cache"
            if archived_cache.exists():
                continue
            try:
                archived_cache.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(cache), str(archived_cache))
                repository.rmdir()
                if not any(publisher.iterdir()):
                    publisher.rmdir()
            except OSError:
                continue


def managed_models():
    """Return app-managed model files directly under their dedicated folders."""
    paths = []
    seen_sources = set()
    for root in _model_roots():
        if not root.exists():
            continue
        for folder in root.iterdir():
            if not folder.is_dir() or folder.is_symlink():
                continue
            path = folder / MODEL_FILENAME
            metadata = read_model_metadata(path)
            if path.is_file() and not path.is_symlink() and metadata and is_standalone_gguf_file(metadata.get("filename", "")):
                source = (metadata.get("repo_id"), metadata.get("filename"))
                if source in seen_sources:
                    continue
                seen_sources.add(source)
                paths.append(path)
    return sorted(paths, key=lambda path: model_display_name(path).casefold())


def model_source(path):
    """Return (repository, original filename) from metadata or a legacy path."""
    metadata = read_model_metadata(path)
    if metadata:
        return metadata.get("repo_id"), metadata.get("filename")
    for root in _model_roots():
        try:
            relative = Path(path).relative_to(root)
            if len(relative.parts) == 3:
                return f"{relative.parts[0]}/{relative.parts[1]}", Path(path).name
        except ValueError:
            continue
    return None, None


def local_models():
    """List managed models plus valid legacy files if migration was blocked."""
    paths = managed_models()
    known_sources = {model_source(path) for path in paths}
    for root in _model_roots():
        if root.exists():
            for path in root.rglob("*.gguf"):
                try:
                    relative = path.relative_to(root)
                    if len(relative.parts) != 3 or path.is_symlink() or not path.is_file():
                        continue
                    if not is_standalone_gguf_file(path.name):
                        continue
                    with path.open("rb") as model_file:
                        if model_file.read(4) != b"GGUF":
                            continue
                    if model_source(path) not in known_sources:
                        paths.append(path)
                except OSError:
                    continue
    return sorted(paths, key=lambda path: model_display_name(path).casefold())


def model_folder_size(path):
    """Return the bytes in a validated managed model folder."""
    folder = _validated_model_folder(path)
    return sum(item.stat().st_size for item in folder.iterdir() if item.is_file())


def _validated_model_folder(path):
    path = Path(path)
    folder = path.parent
    allowed_roots = {root.resolve() for root in _model_roots()}
    if path.name != MODEL_FILENAME or folder.is_symlink() or folder.resolve().parent not in allowed_roots:
        raise ValueError("This is not a model stored in a dedicated Local LLM folder.")
    metadata = read_model_metadata(path)
    if not metadata or not path.is_file() or path.is_symlink():
        raise ValueError("The model folder is not recognized as app-managed; it was left unchanged.")
    contents = {item.name for item in folder.iterdir()}
    if contents != {MODEL_FILENAME, METADATA_FILENAME}:
        raise ValueError("The model folder contains unrecognized files; nothing was deleted.")
    return folder


def delete_managed_model(path):
    """Delete only model.gguf and model.json from a validated dedicated folder."""
    folder = _validated_model_folder(path)
    freed = sum(item.stat().st_size for item in folder.iterdir() if item.is_file())
    (folder / MODEL_FILENAME).unlink()
    (folder / METADATA_FILENAME).unlink()
    folder.rmdir()
    return freed


def inspect_download(repo_id, variant):
    """Validate a single-file variant and check its destination and disk space."""
    if MODELS_DIR.is_symlink():
        raise ValueError("The Local LLM model directory is a symbolic link; downloads were stopped for safety.")
    files = variant.get("files") or []
    if len(files) != 1:
        raise ValueError("This GGUF variant requires multiple shards. Multi-shard downloads are not supported yet.")

    size = variant.get("size_bytes")
    if size is None:
        raise ValueError("The GGUF file size is unknown, so available disk space cannot be checked safely.")

    filename = files[0]
    if not is_standalone_gguf_file(filename):
        raise ValueError(
            "This GGUF is an auxiliary component, not a standalone language model. "
            "Select a normal model quantization such as Q4_K_M instead."
        )

    migrate_legacy_models()
    folder = _choose_model_folder(repo_id, filename)
    destination = folder / MODEL_FILENAME
    if _metadata_matches(folder, repo_id, filename) and destination.is_file():
        if destination.stat().st_size != size:
            raise ValueError("The existing model file size does not match Hugging Face metadata; delete it before retrying.")
        return {"already_downloaded": True, "path": destination, "size": size, "free_space": None}
    if folder.exists():
        raise ValueError(f"The model folder already exists but is not a complete matching download: {folder}")

    existing_parent = MODELS_DIR
    while not existing_parent.exists():
        existing_parent = existing_parent.parent
    free_space = shutil.disk_usage(existing_parent).free
    if free_space < size:
        raise ValueError(
            f"Insufficient disk space. File size: {size / (1024 ** 3):.2f} GB; "
            f"available: {free_space / (1024 ** 3):.2f} GB."
        )

    return {"already_downloaded": False, "path": destination, "size": size, "free_space": free_space}


def download_variant(repo_id, variant, progress=None):
    """Download a GGUF into models/<readable model and quantization>/."""
    plan = inspect_download(repo_id, variant)
    if plan["already_downloaded"]:
        return plan

    filename = variant["files"][0]
    destination = Path(plan["path"])
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    class DownloadProgress(tqdm):
        def __init__(self, *arguments, **options):
            super().__init__(*arguments, **options)
            self.transferred_bytes = options.get("initial", 0)

        def update(self, amount=1):
            result = super().update(amount)
            self.transferred_bytes += amount
            if progress:
                progress(self.transferred_bytes, self.total if self.total is not None else -1)
            return result

    with tempfile.TemporaryDirectory(prefix=".download-", dir=MODELS_DIR) as temporary:
        downloaded = Path(hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=temporary,
            cache_dir=str(Path(temporary) / ".hf-cache"),
            tqdm_class=DownloadProgress,
        ))
        if not downloaded.is_file() or downloaded.stat().st_size != plan["size"]:
            raise ValueError("The downloaded file size does not match Hugging Face metadata.")
        with downloaded.open("rb") as model_file:
            if model_file.read(4) != b"GGUF":
                raise ValueError("The downloaded file does not have a valid GGUF header.")

        destination.parent.mkdir(parents=True, exist_ok=False)
        metadata = {
            "managed_by": "local-llm",
            "format_version": 1,
            "repo_id": repo_id,
            "filename": filename,
            "size_bytes": plan["size"],
        }
        try:
            try:
                downloaded.resolve().relative_to(Path(temporary).resolve())
                staged_local_file = not downloaded.is_symlink()
            except ValueError:
                staged_local_file = False
            if staged_local_file:
                shutil.move(str(downloaded), str(destination))
            else:
                shutil.copyfile(downloaded, destination)
            (destination.parent / METADATA_FILENAME).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        except OSError:
            if destination.exists():
                destination.unlink()
            if destination.parent.exists() and not any(destination.parent.iterdir()):
                destination.parent.rmdir()
            raise
    return {**plan, "path": destination, "already_downloaded": False}
