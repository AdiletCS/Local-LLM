"""Start and communicate with one local llama.cpp server."""

import json
import os
import platform
import shutil
import socket
import struct
import subprocess
import threading
import time
import tempfile
import zipfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from models import is_standalone_gguf_file
from conversations import MAX_OUTPUT_TOKENS
from settings import APP_DATA_DIR, CONTEXT_OPTIONS, effective_threads


INSTALL_ROOT = APP_DATA_DIR
LLAMA_DIR = INSTALL_ROOT / "llama.cpp"
APP_DIRECTORY = Path(__file__).resolve().parent
LOCAL_LLAMA_DIR = APP_DIRECTORY / "tools" / "llama.cpp"
REQUIRED_DLLS = (
    "llama-server-impl.dll",
    "llama-common.dll",
    "llama.dll",
    "ggml.dll",
    "ggml-base.dll",
    "libomp.dll",
)
GIB = 1024 ** 3
RESPONSE_RESERVE = MAX_OUTPUT_TOKENS
TOKEN_SAFETY_MARGIN = 64
SYSTEM_RAM_RESERVE = 4 * GIB
RUNTIME_RAM_RESERVE = 2 * GIB


def _valid_install(folder):
    """Find a server with the core DLLs from the official Windows archive."""
    try:
        if not folder.exists():
            return None
        for executable in folder.rglob("llama-server.exe"):
            parent = executable.parent
            required = [parent / name for name in REQUIRED_DLLS]
            cpu_libraries = list(parent.glob("ggml-cpu-*.dll"))
            if executable.is_file() and executable.stat().st_size:
                if all(path.is_file() and path.stat().st_size for path in required):
                    if any(path.stat().st_size for path in cpu_libraries):
                        return executable
    except PermissionError:
        return None
    return None


def find_llama_server():
    """Return the installed server path, or None if it is not installed."""
    return _valid_install(LLAMA_DIR) or _valid_install(LOCAL_LLAMA_DIR)


def _server_process_is_running(executable):
    """Check whether this exact server executable is currently in use."""
    try:
        import psutil
    except ImportError as error:
        raise RuntimeError("Cannot safely check for a running llama-server process because psutil is unavailable.") from error

    expected = os.path.normcase(os.path.abspath(executable))
    for process in psutil.process_iter(["name"]):
        try:
            if (process.info.get("name") or "").lower() != "llama-server.exe":
                continue
            running_path = process.exe()
            if os.path.normcase(os.path.abspath(running_path)) == expected:
                return True
        except psutil.NoSuchProcess:
            continue
        except psutil.AccessDenied:
            raise RuntimeError(
                "Windows would not let the installer verify a running llama-server process. "
                "Close the server before repairing the installation."
            )
    return False


def _validate_archive(folder):
    executable = _valid_install(folder)
    if executable:
        return executable
    raise RuntimeError(
        "The downloaded archive is incomplete: llama-server.exe or one of its required DLL files is missing."
    )


def install_llama_cpp(progress=None):
    """Stage, validate, and install the latest official Windows x64 CPU release."""
    if platform.system() != "Windows" or platform.machine().lower() not in {"amd64", "x86_64"}:
        raise RuntimeError("Automatic installation currently supports Windows x64 only.")

    existing = find_llama_server()
    if existing:
        if progress:
            progress(f"Using existing llama.cpp installation: {existing}")
        return existing, "already installed"

    try:
        import cpuinfo

        flags = set(cpuinfo.get_cpu_info().get("flags") or [])
        cpu_support = "AVX2 detected" if "avx2" in flags else "AVX2 not reported; using the x64 CPU build"
    except Exception:
        cpu_support = "CPU features unavailable; using the x64 CPU build"

    def report(message):
        if progress:
            progress(message)

    report(f"Checking latest official release ({cpu_support})...")
    api_request = Request(
        "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=20",
        headers={"User-Agent": "Local-LLM"},
    )
    try:
        with urlopen(api_request, timeout=30) as response:
            releases = json.loads(response.read().decode("utf-8"))
        release = None
        asset = None
        for candidate in releases:
            matching = next(
                (item for item in candidate.get("assets", [])
                 if item["name"].startswith("llama-") and item["name"].endswith("-bin-win-cpu-x64.zip")),
                None,
            )
            if matching:
                release, asset = candidate, matching
                break
        if not release:
            raise StopIteration("No recent Windows x64 CPU archive was published.")
    except (HTTPError, URLError, TimeoutError, OSError, ValueError, KeyError, IndexError, StopIteration) as error:
        raise RuntimeError(f"Could not find the latest Windows x64 CPU release: {error}") from error

    try:
        INSTALL_ROOT.mkdir(parents=True, exist_ok=True)
    except PermissionError as error:
        raise RuntimeError(
            f"Windows denied access to the per-user install folder {INSTALL_ROOT}: {error}. "
            "Check that your Windows account can write to Local AppData."
        ) from error
    report(f"Downloading llama.cpp {release['tag_name']} to a temporary folder...")
    try:
        with tempfile.TemporaryDirectory(prefix="llama.cpp-", dir=INSTALL_ROOT) as temporary:
            stage = Path(temporary)
            archive_path = stage / asset["name"]
            extracted = stage / "extracted"
            extracted.mkdir()
            download_request = Request(asset["browser_download_url"], headers={"User-Agent": "Local-LLM"})
            with urlopen(download_request, timeout=60) as response, archive_path.open("wb") as archive:
                total = int(response.headers.get("Content-Length") or asset.get("size") or 0)
                received = 0
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    archive.write(chunk)
                    received += len(chunk)
                    if total:
                        report(f"Downloading llama.cpp {release['tag_name']}... {received * 100 // total}%")

            report("Extracting and checking llama.cpp files...")
            with zipfile.ZipFile(archive_path) as archive:
                for entry in archive.infolist():
                    member = Path(entry.filename)
                    if member.is_absolute() or ".." in member.parts:
                        raise RuntimeError("The llama.cpp archive contains an unsafe file path.")
                archive.extractall(extracted)
            _validate_archive(extracted)

            if LLAMA_DIR.exists():
                for current_executable in LLAMA_DIR.rglob("llama-server.exe"):
                    if _server_process_is_running(current_executable):
                        raise RuntimeError(
                            f"Cannot update llama.cpp because this server is running: {current_executable}. "
                            "Stop that model before repairing the installation."
                        )

            report("Installing llama.cpp outside the project folder...")
            backup = INSTALL_ROOT / f"llama.cpp.previous-{os.getpid()}"
            if backup.exists():
                raise RuntimeError(f"Cannot safely replace the partial install because this backup already exists: {backup}")
            moved_old_install = False
            try:
                if LLAMA_DIR.exists():
                    os.replace(LLAMA_DIR, backup)
                    moved_old_install = True
                os.replace(extracted, LLAMA_DIR)
            except OSError as error:
                if moved_old_install and not LLAMA_DIR.exists():
                    try:
                        os.replace(backup, LLAMA_DIR)
                    except OSError as rollback_error:
                        raise RuntimeError(
                            f"Installation failed ({error}) and the old folder could not be restored. "
                            f"Its files remain at {backup}."
                        ) from rollback_error
                if isinstance(error, PermissionError):
                    raise RuntimeError(
                        f"Windows denied access while placing llama.cpp in {LLAMA_DIR}: {error}. "
                        "Check that no llama-server from this folder is running and that your Windows account can "
                        "write to Local AppData."
                    ) from error
                raise RuntimeError(f"Could not safely place llama.cpp in {LLAMA_DIR}: {error}") from error
            if moved_old_install:
                shutil.rmtree(backup, ignore_errors=True)
    except PermissionError as error:
        raise RuntimeError(
            f"Windows denied access during temporary download or extraction: {error}. "
            f"The install is staged under {INSTALL_ROOT}; check Local AppData permissions."
        ) from error
    except (HTTPError, URLError, TimeoutError, OSError, zipfile.BadZipFile) as error:
        raise RuntimeError(f"Could not download or extract llama.cpp: {error}") from error

    executable = find_llama_server()
    if not executable:
        raise RuntimeError(f"llama.cpp was placed in {LLAMA_DIR}, but required files did not validate.")
    report(f"Installed llama.cpp {release['tag_name']}.")
    return executable, release["tag_name"]


def read_gguf_metadata(model_path):
    """Read the small set of scalar GGUF metadata values used for context/KV estimates."""
    wanted_suffixes = {
        ".context_length", ".block_count", ".embedding_length",
        ".attention.head_count", ".attention.head_count_kv",
    }
    values = {}
    scalar_formats = {
        0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
        6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d",
    }

    try:
        with Path(model_path).open("rb") as file:
            file_size = Path(model_path).stat().st_size
            if file.read(4) != b"GGUF":
                return {}

            def read(fmt):
                data = file.read(struct.calcsize(fmt))
                if len(data) != struct.calcsize(fmt):
                    raise ValueError("Incomplete GGUF metadata")
                return struct.unpack(fmt, data)[0]

            def read_string():
                length = read("<Q")
                if length > file_size - file.tell():
                    raise ValueError("Invalid GGUF string length")
                return file.read(length).decode("utf-8", errors="replace")

            def skip_value(value_type):
                if value_type in scalar_formats:
                    file.seek(struct.calcsize(scalar_formats[value_type]), 1)
                elif value_type == 8:
                    length = read("<Q")
                    file.seek(length, 1)
                elif value_type == 9:
                    element_type = read("<I")
                    count = read("<Q")
                    if count > 10_000_000:
                        raise ValueError("Unreasonable GGUF array length")
                    if element_type in scalar_formats:
                        file.seek(struct.calcsize(scalar_formats[element_type]) * count, 1)
                    elif element_type == 8:
                        for _ in range(count):
                            file.seek(read("<Q"), 1)
                    else:
                        raise ValueError("Unsupported GGUF array type")
                else:
                    raise ValueError("Unsupported GGUF metadata type")
                if file.tell() > file_size:
                    raise ValueError("GGUF metadata extends past the file")

            _version = read("<I")
            _tensor_count = read("<Q")
            metadata_count = read("<Q")
            if metadata_count > 100_000:
                return {}
            for _ in range(metadata_count):
                key = read_string()
                value_type = read("<I")
                if key == "general.architecture" and value_type == 8:
                    values["architecture"] = read_string()
                elif any(key.endswith(suffix) for suffix in wanted_suffixes) and value_type in scalar_formats:
                    value = read(scalar_formats[value_type])
                    if isinstance(value, (int, float)):
                        values[key] = int(value)
                else:
                    skip_value(value_type)
    except (OSError, ValueError, struct.error, OverflowError):
        return {}
    return values


def choose_context_size(model_path, total_ram=None, available_ram=None, preferred_context=None):
    """Choose a supported context using GGUF context/KV data and conservative RAM reserves."""
    metadata = read_gguf_metadata(model_path)
    architecture = metadata.get("architecture")
    context_limit = metadata.get(f"{architecture}.context_length") if architecture else None
    context_known = bool(context_limit and 128 <= context_limit <= 1_000_000)
    if not context_known:
        context_limit = 2048

    candidates = [size for size in CONTEXT_OPTIONS if size <= context_limit]
    if preferred_context is not None:
        if preferred_context not in CONTEXT_OPTIONS or preferred_context > context_limit:
            raise ValueError(f"The requested {preferred_context}-token context is not supported by this model.")
        candidates = [preferred_context]
    if not candidates:
        candidates = [context_limit]
    smallest = candidates[0]
    kv_per_token = None
    if architecture:
        layers = metadata.get(f"{architecture}.block_count")
        embedding = metadata.get(f"{architecture}.embedding_length")
        heads = metadata.get(f"{architecture}.attention.head_count")
        kv_heads = metadata.get(f"{architecture}.attention.head_count_kv")
        if layers and embedding and heads and kv_heads and embedding % heads == 0:
            kv_per_token = layers * (embedding // heads) * kv_heads * 4  # K and V, F16 each.

    # Without KV dimensions, stay at the established small context rather than guessing.
    if kv_per_token is None:
        if preferred_context is not None and preferred_context > 2048:
            raise ValueError(
                "This model does not expose enough GGUF metadata to estimate memory for a manual context above "
                "2,048 tokens. Choose Auto or use a smaller context."
            )
        conservative_context = min(2048, context_limit)
        candidates = [conservative_context]
        smallest = conservative_context

    try:
        import psutil

        memory = psutil.virtual_memory()
        total_ram = memory.total if total_ram is None else total_ram
        available_ram = memory.available if available_ram is None else available_ram
    except Exception:
        total_ram = total_ram or 0
        available_ram = available_ram or 0

    size_bytes = Path(model_path).stat().st_size
    model_and_runtime = int(size_bytes * 1.3) + RUNTIME_RAM_RESERVE
    total_budget = max(0, total_ram - SYSTEM_RAM_RESERVE)
    safe_budget = min(total_budget, available_ram)
    selected = smallest
    fallback_kv_per_token = 1024 * 1024  # Conservative 2 GiB at the 2048-token fallback.
    selected_kv = (kv_per_token or fallback_kv_per_token) * smallest
    selected_estimate = model_and_runtime + selected_kv
    fits_available = selected_estimate <= safe_budget
    for candidate in candidates:
        kv_bytes = (kv_per_token or fallback_kv_per_token) * candidate
        estimate = model_and_runtime + kv_bytes
        if estimate <= safe_budget:
            selected, selected_estimate, selected_kv, fits_available = candidate, estimate, kv_bytes, True
        else:
            break

    warning = ""
    if not fits_available:
        warning = (
            f"The estimated model and KV-cache requirement at {selected:,} tokens is "
            f"{selected_estimate / GIB:.1f} GB, above currently available memory. "
            "Loading may fail; close other apps and retry."
        )
    if not context_known:
        context_note = "GGUF context metadata was unavailable; using a conservative 2048-token fallback."
    elif kv_per_token is None:
        context_note = (
            f"GGUF reports support for {context_limit:,} tokens, but its KV-cache dimensions were incomplete; "
            "using a conservative 2048-token context."
        )
    else:
        context_note = (
            f"GGUF reports up to {context_limit:,} tokens; KV-cache memory is estimated from its architecture metadata."
        )
    return {
        "context_size": selected,
        "supported_context": context_limit if context_known else None,
        "memory_estimate_bytes": selected_estimate,
        "kv_cache_bytes": selected_kv,
        "metadata_reliable": context_known and kv_per_token is not None,
        "context_note": context_note,
        "fits_available": fits_available,
        "warning": warning,
    }


class LlamaRuntime:
    def __init__(self):
        self.process = None
        self.port = None
        self.log_file = None
        self.context_size = None
        self.context_info = None
        self._stream_lock = threading.Lock()
        self._stream_cancel = None
        self._stream_response = None
        self.last_finish_reason = None
        self.last_usage = None
        self.last_timings = None

    def start(
        self, executable, model_path, threads=None, batch_size=None,
        microbatch_size=None, context_size=None, cancel_event=None,
    ):
        """Start llama-server and wait for its local health endpoint."""
        if not is_standalone_gguf_file(model_path):
            raise RuntimeError(
                "This GGUF file is an auxiliary component (such as a vision projector or adapter), "
                "not a standalone language model. Select a language-model GGUF."
            )
        if not Path(model_path).is_file():
            raise RuntimeError(f"The selected model file does not exist: {model_path}")
        self.stop()
        try:
            context_info = choose_context_size(model_path, preferred_context=context_size)
            selected_context = context_info["context_size"]
            if threads is None:
                threads = effective_threads()
            command = [executable, "-m", str(model_path)]
            if batch_size is not None:
                command.extend(["-b", str(batch_size)])
            if microbatch_size is not None:
                command.extend(["-ub", str(microbatch_size)])
            INSTALL_ROOT.mkdir(parents=True, exist_ok=True)
            options = {
                "cwd": str(Path(executable).resolve().parent),
                "stderr": subprocess.STDOUT,
            }
            if hasattr(subprocess, "CREATE_NO_WINDOW"):
                options["creationflags"] = subprocess.CREATE_NO_WINDOW
            command.extend(["--host", "127.0.0.1", "-c", str(selected_context), "-t", str(threads), "-ngl", "0", "--jinja"])
        except (OSError, ValueError) as error:
            self.stop()
            raise RuntimeError(f"Could not start llama-server: {error}") from error

        for port_attempt in range(3):
            try:
                self.context_info = context_info
                self.context_size = selected_context
                with socket.socket() as sock:
                    sock.bind(("127.0.0.1", 0))
                    self.port = sock.getsockname()[1]
                self.log_file = (INSTALL_ROOT / "llama-server.log").open("w", encoding="utf-8")
                self.process = subprocess.Popen(
                    command + ["--port", str(self.port)], stdout=self.log_file, **options
                )
            except (OSError, ValueError) as error:
                self.stop()
                raise RuntimeError(f"Could not start llama-server: {error}") from error
            deadline = time.monotonic() + 180
            retry_port = False
            while time.monotonic() < deadline:
                if cancel_event is not None and cancel_event.is_set():
                    self.stop()
                    raise RuntimeError("Model loading was cancelled.")
                if self.process.poll() is not None:
                    exit_code = self.process.returncode
                    self.stop()
                    details = self._read_server_log()
                    lower_log = details.lower()
                    if any(term in lower_log for term in ("address already in use", "failed to bind", "bind failed")) and port_attempt < 2:
                        retry_port = True
                        break
                    if "clip cannot be used as main model" in lower_log:
                        message = (
                            "This GGUF is an auxiliary vision projector, not a standalone language model. "
                            "Select a language-model GGUF instead."
                        )
                    elif "address already in use" in lower_log or "failed to bind" in lower_log or "bind failed" in lower_log:
                        message = f"llama-server could not use a local port after {port_attempt + 1} attempts. Check for a port conflict."
                    else:
                        message = (
                            f"llama-server could not load this model (exit code {exit_code}). "
                            f"See the short server log at {INSTALL_ROOT / 'llama-server.log'}."
                        )
                    raise RuntimeError(message)
                try:
                    with urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2) as response:
                        if response.status == 200:
                            return
                except HTTPError as error:
                    if error.code != 503:
                        self.stop()
                        raise RuntimeError(f"llama-server health check failed: HTTP {error.code}") from error
                except (URLError, TimeoutError, OSError):
                    pass
                time.sleep(0.5)
            if retry_port:
                continue
            self.stop()
            raise RuntimeError("Timed out waiting for llama-server to load the model.")
        self.stop()
        raise RuntimeError("Could not find an available localhost port for llama-server.")

    def _read_server_log(self):
        path = INSTALL_ROOT / "llama-server.log"
        try:
            return path.read_text(encoding="utf-8", errors="replace")[-6000:].strip()
        except OSError:
            return "No server log was available."

    def count_prompt_tokens(self, messages, max_tokens=RESPONSE_RESERVE):
        """Count a templated chat prompt with llama.cpp, falling back conservatively if needed."""
        if not self.process or self.process.poll() is not None:
            raise RuntimeError("The model server is not running.")

        def post_json(endpoint, body):
            request = Request(
                f"http://127.0.0.1:{self.port}{endpoint}",
                data=json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urlopen(request, timeout=15) as response:
                return json.loads(response.read().decode("utf-8"))

        try:
            result = post_json(
                "/v1/chat/completions/input_tokens",
                {"messages": messages, "max_tokens": max_tokens},
            )
            count = result.get("input_tokens")
            if isinstance(count, int) and count >= 0:
                return count, "llama.cpp chat tokenizer"
        except (HTTPError, URLError, TimeoutError, OSError, ValueError):
            pass

        try:
            templated = post_json("/apply-template", {"messages": messages})
            prompt = templated.get("prompt")
            if isinstance(prompt, str):
                result = post_json(
                    "/tokenize",
                    {"content": prompt, "add_special": False, "parse_special": True},
                )
                tokens = result.get("tokens")
                if isinstance(tokens, list):
                    return len(tokens), "llama.cpp template + tokenizer"
        except (HTTPError, URLError, TimeoutError, OSError, ValueError):
            pass

        # Content bytes are an intentionally high-side estimate for common tokenizers;
        # fixed costs cover per-message and chat-template markers.
        estimate = 128 + sum(len(item.get("content", "").encode("utf-8")) + 16 for item in messages)
        return estimate, "conservative UTF-8 estimate"

    def generate(self, prompt):
        """Return a complete response, using the same real stream as the UI."""
        return "".join(self.generate_stream(prompt))

    def generate_stream(self, prompt, max_tokens=MAX_OUTPUT_TOKENS, temperature=0.7, cache_prompt=True):
        """Yield real text chunks from llama.cpp's OpenAI-compatible SSE endpoint."""
        if not self.process or self.process.poll() is not None:
            raise RuntimeError("The model server is not running.")
        messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
        self.last_finish_reason = None
        self.last_usage = None
        self.last_timings = None
        body = json.dumps({
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
            # Some reasoning-tuned chat templates emit no user-visible answer by default.
            "chat_template_kwargs": {"enable_thinking": False},
            "cache_prompt": cache_prompt,
        }).encode("utf-8")
        request = Request(
            f"http://127.0.0.1:{self.port}/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        cancelled = threading.Event()
        response = None
        with self._stream_lock:
            self._stream_cancel = cancelled
        try:
            response = urlopen(request, timeout=600)
            with self._stream_lock:
                self._stream_response = response
            finished = False
            while not cancelled.is_set():
                line = response.readline()
                if not line:
                    break
                if not line.startswith(b"data:"):
                    continue
                payload = line[5:].strip()
                if payload == b"[DONE]":
                    break
                try:
                    event = json.loads(payload.decode("utf-8"))
                except (UnicodeDecodeError, ValueError) as error:
                    raise RuntimeError("llama.cpp sent an invalid streaming response.") from error
                if event.get("error"):
                    detail = event["error"]
                    message = detail.get("message", str(detail)) if isinstance(detail, dict) else str(detail)
                    raise RuntimeError(f"llama.cpp error: {message}")
                if isinstance(event.get("usage"), dict):
                    self.last_usage = event["usage"]
                if isinstance(event.get("timings"), dict):
                    self.last_timings = event["timings"]
                choices = event.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                delta = choice.get("delta") or {}
                text = delta.get("content")
                if isinstance(text, str) and text:
                    yield text
                if choice.get("finish_reason") is not None:
                    self.last_finish_reason = choice["finish_reason"]
                    finished = True
            if cancelled.is_set():
                self.last_finish_reason = "cancelled"
            elif not finished:
                raise RuntimeError("llama.cpp ended the stream without a finish reason; the response may be incomplete.")
            elif self.last_finish_reason not in {"stop", "length"}:
                raise RuntimeError(
                    f"llama.cpp ended generation with finish reason {self.last_finish_reason!r}; "
                    "the response was not treated as complete."
                )
        except HTTPError as error:
            try:
                detail = error.read().decode("utf-8", errors="replace")
            except OSError:
                detail = str(error)
            raise RuntimeError(f"Generation failed: {detail or error}") from error
        except (URLError, TimeoutError, OSError, ValueError) as error:
            if not cancelled.is_set():
                raise RuntimeError(f"Generation failed: {error}") from error
        finally:
            if cancelled.is_set():
                self.last_finish_reason = "cancelled"
            with self._stream_lock:
                if self._stream_response is response:
                    self._stream_response = None
                if self._stream_cancel is cancelled:
                    self._stream_cancel = None
            if response:
                try:
                    response.close()
                except (OSError, ValueError):
                    pass

    def cancel_generation(self):
        """Cancel the active HTTP stream without stopping the model server."""
        with self._stream_lock:
            cancelled = self._stream_cancel
            response = self._stream_response
            if cancelled:
                cancelled.set()
        if response:
            try:
                response.close()
            except (OSError, ValueError):
                pass

    def stop(self):
        """Stop the current server, if one is running."""
        process = self.process
        self.process = None
        self.port = None
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if self.log_file:
            self.log_file.close()
            self.log_file = None
        self.context_size = None
        self.context_info = None
