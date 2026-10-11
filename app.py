"""Desktop interface for managing and chatting with local language models."""

import os
import sys
import time
from pathlib import Path

import psutil
from PySide6.QtCore import QEvent, QTimer, Qt, QThread, Signal
from PySide6.QtGui import QFont, QFontDatabase, QFontMetrics
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
    QSpinBox,
)

from downloader import (
    delete_managed_model,
    download_variant,
    inspect_download,
    local_models,
    migrate_legacy_models,
    model_display_name,
    model_folder_size,
    model_source,
    read_model_metadata,
)
import conversations
from hardware import detect_hardware
from models import DEFAULT_CANDIDATE_LIMIT, MAX_CANDIDATE_LIMIT, find_models, is_standalone_gguf_file
from runtime import LlamaRuntime, find_llama_server, install_llama_cpp
from selector import check_models, compatibility_summary, max_practical_gguf_size, recommend_models
from settings import automatic_batch_sizes, automatic_response_tokens


class HardwareWorker(QThread):
    completed = Signal(object)
    failed = Signal(str)

    def run(self):
        try:
            self.completed.emit(detect_hardware())
        except Exception as error:
            self.failed.emit(str(error))


class DiscoveryWorker(QThread):
    completed = Signal(object)
    failed = Signal(str)
    progress = Signal(str)

    def __init__(self, candidate_limit):
        super().__init__()
        self.candidate_limit = candidate_limit

    def run(self):
        try:
            started = time.perf_counter()
            self.progress.emit("Detecting hardware…")
            hardware = detect_hardware()
            raw, stats = find_models(
                candidate_limit=self.candidate_limit,
                return_stats=True,
                progress=self.progress.emit,
            )
            self.progress.emit("Checking memory and disk compatibility…")
            checked = check_models(raw)
            stats.update(compatibility_summary(checked))
            top = recommend_models(checked, limit=10)
            stats["max_practical_file_bytes"] = max_practical_gguf_size()
            stats["compatible_recommendations"] = len(top)
            stats["elapsed_seconds"] = round(time.perf_counter() - started, 1)
            self.progress.emit(
                f"Found {stats.get('hardware_compatible_models', 0)} compatible models; preparing recommendations…"
            )
            self.completed.emit({"models": [item["model"] for item in top], "stats": stats, "hardware": hardware})
        except Exception as error:
            self.failed.emit(str(error))


class DownloadWorker(QThread):
    progress = Signal(int, int)
    completed = Signal(str)
    failed = Signal(str)

    def __init__(self, repo_id, variant):
        super().__init__()
        self.repo_id = repo_id
        self.variant = variant

    def run(self):
        try:
            result = download_variant(self.repo_id, self.variant, self.progress.emit)
            self.completed.emit(str(result["path"]))
        except Exception as error:
            self.failed.emit(str(error))


class RuntimeWorker(QThread):
    completed = Signal(str)
    failed = Signal(str)

    def __init__(self, operation, runtime, *arguments):
        super().__init__()
        self.operation = operation
        self.runtime = runtime
        self.arguments = arguments

    def run(self):
        try:
            result = getattr(self.runtime, self.operation)(*self.arguments)
            self.completed.emit(result if isinstance(result, str) else "")
        except Exception as error:
            self.failed.emit(str(error))


class GenerationWorker(QThread):
    chunk = Signal(int, str)
    status = Signal(int, str)
    context_ready = Signal(int, bool)
    completed = Signal(int, bool, str)
    failed = Signal(int, str)

    def __init__(self, request_id, runtime, saved_messages, latest_user, max_response_tokens, conversation_id):
        super().__init__()
        self.request_id = request_id
        self.runtime = runtime
        self.saved_messages = saved_messages
        self.latest_user = latest_user
        self.max_response_tokens = max_response_tokens
        self.conversation_id = conversation_id
        self.cancelled = False
        self.text = ""
        self.result_status = "generating"
        self.error = None
        self.finish_reason = None

    def cancel(self):
        self.cancelled = True
        self.runtime.cancel_generation()

    def run(self):
        try:
            context_size = self.runtime.context_size or conversations.CONTEXT_SIZE
            token_counter = lambda items: self.runtime.count_prompt_tokens(items, self.max_response_tokens)
            summary_state = conversations.get_summary(self.conversation_id)

            def build():
                return conversations.build_context(
                    self.saved_messages,
                    self.latest_user,
                    context_size,
                    token_counter,
                    response_reserve=self.max_response_tokens,
                    summary=summary_state["content"],
                    summary_through_id=summary_state["through_message_id"],
                    return_ids=True,
                )

            messages, omitted, token_count, token_method, included_ids = build()
            if omitted and not self.cancelled:
                batch = conversations.summarization_batch(
                    self.saved_messages, included_ids, summary_state["content"],
                    summary_state["through_message_id"], context_size, token_counter,
                )
                if batch:
                    self.status.emit(self.request_id, "Preparing conversation…")
                    try:
                        summary_text = "".join(self.runtime.generate_stream(
                            batch["messages"], max_tokens=min(256, max(128, context_size // 8)), temperature=0.2,
                        )).strip()
                        if self.cancelled:
                            self.result_status = "interrupted"
                            self.finish_reason = "cancelled"
                            self.completed.emit(self.request_id, True, self.finish_reason)
                            return
                        if self.runtime.last_finish_reason == "stop" and summary_text:
                            previous_summary = summary_state
                            summary_state = {
                                "content": summary_text,
                                "through_message_id": batch["through_message_id"],
                            }
                            try:
                                prepared = build()
                                conversations.save_summary(
                                    self.conversation_id, summary_text, batch["through_message_id"]
                                )
                            except Exception:
                                # Keep the earlier summary/recent context if the new memory will not fit.
                                summary_state = previous_summary
                            else:
                                messages, omitted, token_count, token_method, included_ids = prepared
                    except Exception:
                        # A summary is an optimization. Keep the existing recent-turn fallback.
                        if self.cancelled:
                            self.result_status = "interrupted"
                            self.finish_reason = "cancelled"
                            self.completed.emit(self.request_id, True, self.finish_reason)
                            return

            self.context_ready.emit(self.request_id, omitted)
            for text in self.runtime.generate_stream(messages, max_tokens=self.max_response_tokens):
                if self.cancelled:
                    break
                self.text += text
                self.chunk.emit(self.request_id, text)
            self.finish_reason = self.runtime.last_finish_reason
            if self.cancelled:
                self.result_status = "interrupted"
            elif self.finish_reason == "stop":
                self.result_status = "complete"
            elif self.finish_reason == "length":
                self.result_status = "limit"
            else:
                self.result_status = "error"
                self.error = "Generation ended without a reliable completion reason."
                self.failed.emit(self.request_id, self.error)
                return
            self.completed.emit(self.request_id, self.cancelled, self.finish_reason or "unknown")
        except Exception as error:
            if self.cancelled:
                self.result_status = "interrupted"
                self.finish_reason = "cancelled"
                self.completed.emit(self.request_id, True, self.finish_reason)
            else:
                self.result_status = "error"
                self.error = str(error)
                self.failed.emit(self.request_id, str(error))


class InstallWorker(QThread):
    progress = Signal(str)
    completed = Signal(str)
    failed = Signal(str)

    def run(self):
        try:
            executable, _version = install_llama_cpp(self.progress.emit)
            self.completed.emit(str(executable))
        except Exception as error:
            self.failed.emit(str(error))


class HardwareWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Local LLM")
        self.resize(1120, 780)
        self.setMinimumSize(900, 620)
        self.setStyleSheet("""
            QMainWindow { background: #181a1f; }
            QWidget { color: #e5e7eb; }
            QLabel { color: #e5e7eb; }
            QLabel#muted { color: #a1a7b2; }
            QLabel#title { font-size: 20px; font-weight: 700; }
            QLabel#section { font-size: 14px; font-weight: 650; color: #cbd0d8; }
            QFrame#sidebar, QFrame#modelCard, QFrame#chatPanel { background: #202329; border: 1px solid #30343c; border-radius: 10px; }
            QPushButton { background: #2a2e35; border: 1px solid #3a4049; border-radius: 7px; padding: 6px 10px; color: #e3e6eb; }
            QPushButton:hover { background: #353b44; }
            QPushButton:disabled { color: #777e89; background: #25282e; border-color: #30343b; }
            QPushButton#primary { color: white; background: #3b67b8; border: none; font-weight: 650; }
            QPushButton#primary:hover { background: #4979cf; }
            QPlainTextEdit, QListWidget, QSpinBox { background: #25282f; color: #e5e7eb; border: 1px solid #383d46; border-radius: 7px; padding: 8px; selection-background-color: #3b67b8; }
            QLabel#userBubble { background: #303740; color: #f0f2f5; border-radius: 14px; padding: 10px 14px; }
            QLabel#assistantText { background: transparent; color: #e5e7eb; padding: 6px 12px; }
            QProgressBar { border: none; background: #30343b; border-radius: 4px; height: 7px; }
            QProgressBar::chunk { background: #5281d2; border-radius: 4px; }
            QScrollArea { border: none; background: transparent; }
            QScrollBar:vertical { background: #202329; width: 10px; margin: 2px; }
            QScrollBar::handle:vertical { background: #454b55; border-radius: 4px; min-height: 24px; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
        """)

        self.runtime = LlamaRuntime()
        self.runtime_ready = False
        self.model_state = "Stopped"
        self.generation_worker = None
        self.generation_id = 0
        self.after_generation = None
        self.assistant_widget = None
        self.assistant_text = ""
        self.assistant_message_id = None
        self.current_conversation_id = None
        self.pending_conversation_action = None
        self.selected_model_path = None
        self.selected_model_name = ""
        self.pending = None
        self.busy = False
        self.recommendations = []
        self._checked_models = []
        self.workers = []
        self.installed_paths = []
        conversations.initialize()
        saved_conversations = conversations.list_conversations()
        self.current_conversation_id = (
            saved_conversations[0]["id"] if saved_conversations else conversations.create_conversation()
        )
        try:
            migrated_models = migrate_legacy_models()
            migration_error = ""
        except OSError as error:
            migrated_models = []
            migration_error = f"Older model folders could not be reorganized: {error}"

        central = QWidget()
        page = QHBoxLayout(central)
        page.setContentsMargins(16, 14, 16, 14)
        page.setSpacing(14)

        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(345)
        side = QVBoxLayout(sidebar)
        side.setContentsMargins(14, 13, 14, 13)
        side.setSpacing(9)
        discover_title = QLabel("Discover Models")
        discover_title.setObjectName("title")
        side.addWidget(discover_title)
        self.repository_count_label = QLabel("Public GGUF repository count: not checked")
        self.repository_count_label.setObjectName("muted")
        self.repository_count_label.setWordWrap(True)
        side.addWidget(self.repository_count_label)
        search_row = QHBoxLayout()
        search_row.addWidget(QLabel("Candidate repositories to examine"), 1)
        self.candidate_count = QSpinBox()
        self.candidate_count.setRange(1, MAX_CANDIDATE_LIMIT)
        self.candidate_count.setValue(DEFAULT_CANDIDATE_LIMIT)
        self.candidate_count.setToolTip(f"Choose between 1 and {MAX_CANDIDATE_LIMIT} repositories.")
        search_row.addWidget(self.candidate_count)
        side.addLayout(search_row)
        self.find_button = QPushButton("Find Models for My PC")
        self.find_button.setObjectName("primary")
        self.find_button.clicked.connect(self.find_models)
        side.addWidget(self.find_button)
        self.sidebar_status = QLabel("Installed models are available offline.")
        self.sidebar_status.setObjectName("muted")
        self.sidebar_status.setWordWrap(True)
        side.addWidget(self.sidebar_status)
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.progress_bar.hide()
        side.addWidget(self.progress_bar)

        self.sidebar_scroll = QScrollArea()
        self.sidebar_scroll.setWidgetResizable(True)
        self.sidebar_scroll.viewport().setStyleSheet("background:#202329;")
        content = QWidget()
        content.setStyleSheet("background:#202329;")
        self.sidebar_layout = QVBoxLayout(content)
        self.sidebar_layout.setContentsMargins(0, 0, 0, 0)
        self.sidebar_layout.setSpacing(7)
        installed_title = QLabel("My Models")
        installed_title.setObjectName("section")
        self.sidebar_layout.addWidget(installed_title)
        self.installed_layout = QVBoxLayout()
        self.installed_layout.setSpacing(6)
        self.sidebar_layout.addLayout(self.installed_layout)
        recommended_title = QLabel("Recommended for your PC")
        recommended_title.setObjectName("section")
        self.sidebar_layout.addWidget(recommended_title)
        self.recommendations_layout = QVBoxLayout()
        self.recommendations_layout.setSpacing(6)
        self.sidebar_layout.addLayout(self.recommendations_layout)
        self.sidebar_layout.addStretch()
        self.sidebar_scroll.setWidget(content)
        side.addWidget(self.sidebar_scroll, 1)
        history_title = QLabel("Conversations")
        history_title.setObjectName("section")
        side.addWidget(history_title)
        history_buttons = QHBoxLayout()
        self.new_chat_button = QPushButton("New Chat")
        self.new_chat_button.clicked.connect(self.new_conversation)
        history_buttons.addWidget(self.new_chat_button)
        self.delete_chat_button = QPushButton("Delete")
        self.delete_chat_button.clicked.connect(self.delete_conversation)
        history_buttons.addWidget(self.delete_chat_button)
        side.addLayout(history_buttons)
        self.conversation_list = QListWidget()
        self.conversation_list.setMaximumHeight(115)
        self.conversation_list.itemClicked.connect(self.select_conversation)
        side.addWidget(self.conversation_list)
        page.addWidget(sidebar)

        main = QVBoxLayout()
        main.setSpacing(10)
        chat_title = QLabel("Chat")
        chat_title.setObjectName("section")
        main.addWidget(chat_title)
        active_row = QHBoxLayout()
        self.active_title = QLabel("Choose a model from My Models")
        self.active_title.setObjectName("title")
        active_row.addWidget(self.active_title, 1)
        self.state_label = QLabel("Stopped")
        self.state_label.setStyleSheet("background:#30343b; color:#cbd5e1; border-radius:10px; padding:4px 10px; font-weight:600;")
        active_row.addWidget(self.state_label)
        self.generating_label = QLabel("Generating…")
        self.generating_label.setObjectName("muted")
        self.generating_label.hide()
        active_row.addWidget(self.generating_label)
        self.stop_generation_button = QPushButton("Stop Generating")
        self.stop_generation_button.hide()
        self.stop_generation_button.clicked.connect(self.cancel_generation)
        active_row.addWidget(self.stop_generation_button)
        self.stop_button = QPushButton("Unload")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_model)
        active_row.addWidget(self.stop_button)
        main.addLayout(active_row)

        self.hardware_summary = QLabel("Detecting hardware…")
        self.hardware_summary.setObjectName("muted")
        self.hardware_summary.setWordWrap(True)
        main.addWidget(self.hardware_summary)
        self.operation_label = QLabel("")
        self.operation_label.setObjectName("muted")
        self.operation_label.setWordWrap(True)
        main.addWidget(self.operation_label)
        self.memory_warning = QLabel("")
        self.memory_warning.setStyleSheet("background:#493f25; color:#f5d78a; border-radius:7px; padding:8px;")
        self.memory_warning.setWordWrap(True)
        self.memory_warning.hide()
        main.addWidget(self.memory_warning)

        chat_panel = QFrame()
        chat_panel.setObjectName("chatPanel")
        chat_layout = QVBoxLayout(chat_panel)
        chat_layout.setContentsMargins(12, 12, 12, 12)
        chat_layout.setSpacing(8)
        self.chat_scroll = QScrollArea()
        self.chat_scroll.setWidgetResizable(True)
        self.chat_scroll.viewport().setStyleSheet("background:#202329;")
        self.chat_messages = QWidget()
        self.chat_messages.setStyleSheet("background:#202329;")
        self.chat_messages_layout = QVBoxLayout(self.chat_messages)
        self.chat_messages_layout.setContentsMargins(10, 12, 10, 14)
        self.chat_messages_layout.setSpacing(12)
        self.chat_messages_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.chat_messages_layout.addStretch(1)
        self.chat_scroll.setWidget(self.chat_messages)
        chat_layout.addWidget(self.chat_scroll, 1)
        prompt_row = QHBoxLayout()
        self.prompt_input = QPlainTextEdit()
        self.prompt_input.setPlaceholderText("Load a model to start chatting…")
        self.prompt_input.setFixedHeight(58)
        self.prompt_input.setEnabled(False)
        self.prompt_input.installEventFilter(self)
        prompt_row.addWidget(self.prompt_input, 1)
        self.send_button = QPushButton("Send")
        self.send_button.setObjectName("primary")
        self.send_button.setEnabled(False)
        self.send_button.clicked.connect(self.send_message)
        prompt_row.addWidget(self.send_button)
        chat_layout.addLayout(prompt_row)
        main.addWidget(chat_panel, 1)
        page.addLayout(main, 1)
        self.setCentralWidget(central)

        self.refresh_installed_models()
        self.refresh_conversations()
        self._display_conversation(self.current_conversation_id)
        if migrated_models:
            self.sidebar_status.setText(f"Organized {len(migrated_models)} existing model(s) into dedicated folders.")
        elif migration_error:
            self.sidebar_status.setText(migration_error)
        self.hardware_worker = HardwareWorker()
        self.hardware_worker.completed.connect(self.show_hardware)
        self.hardware_worker.failed.connect(lambda message: self.hardware_summary.setText(f"Hardware unavailable: {message}"))
        self.hardware_worker.start()

    def _remember_worker(self, worker):
        self.workers.append(worker)
        worker.finished.connect(lambda: self.workers.remove(worker) if worker in self.workers else None)

    def eventFilter(self, watched, event):
        if watched is self.prompt_input and event.type() == QEvent.Type.KeyPress:
            if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                    return False
                self.send_message()
                return True
        return super().eventFilter(watched, event)

    def _set_state(self, state, message=""):
        self.model_state = state
        self.state_label.setText(state)
        colors = {
            "Stopped": ("#30343b", "#cbd5e1"),
            "Loading": ("#493f25", "#f5d78a"),
            "Ready": ("#1f3e2e", "#a7f3c2"),
            "Error": ("#4b2429", "#fca5a5"),
        }
        background, foreground = colors.get(state, colors["Stopped"])
        self.state_label.setStyleSheet(
            f"background:{background}; color:{foreground}; border-radius:10px; padding:4px 10px; font-weight:600;"
        )
        if message:
            self.operation_label.setText(message)
        self.active_title.setText(self.selected_model_name or "Choose a model from My Models")
        info = self.runtime.context_info or {}
        if state == "Ready" and self.runtime.context_size:
            warning = info.get("warning", "")
            if warning:
                self.memory_warning.setText(warning)
                self.memory_warning.show()
            elif self.memory_warning.text().startswith("The estimated model and KV-cache"):
                self.memory_warning.hide()
        elif state in {"Stopped", "Error", "Loading"}:
            if self.memory_warning.text().startswith("The estimated model and KV-cache"):
                self.memory_warning.hide()
        can_stop = self.runtime.process is not None and (not self.busy or self.generation_worker is not None)
        self.stop_button.setEnabled(can_stop)
        self.refresh_installed_models()

    def show_hardware(self, details):
        cpu = details.get("CPU", {}).get("Model", "Unknown")
        memory = details.get("Memory", {})
        gpu = details.get("GPU", {}).get("GPU", "Unknown")
        self.hardware_summary.setText(
            f"CPU: {cpu}  ·  RAM: {memory.get('Total RAM', 'Unknown')} "
            f"(available {memory.get('Available RAM', 'Unknown')})  ·  GPU: {gpu}"
        )

    def refresh_installed_models(self):
        while self.installed_layout.count():
            item = self.installed_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.installed_paths = [
            path for path in local_models()
            if path.stat().st_size > 0
        ]

        if not self.installed_paths:
            label = QLabel("No downloaded models yet.")
            label.setObjectName("muted")
            self.installed_layout.addWidget(label)
        for path in self.installed_paths:
            card = QFrame()
            card.setObjectName("modelCard")
            row = QHBoxLayout(card)
            row.setContentsMargins(9, 7, 9, 7)
            labels = QVBoxLayout()
            name = QLabel(model_display_name(path))
            name.setStyleSheet("font-weight:600;")
            name.setWordWrap(True)
            size = QLabel(f"{path.stat().st_size / (1024 ** 3):.2f} GB")
            size.setObjectName("muted")
            labels.addWidget(name)
            labels.addWidget(size)
            row.addLayout(labels, 1)
            actions = QVBoxLayout()
            chat = QPushButton("Chat")
            chat.setObjectName("primary")
            chat.setStyleSheet(
                "background:#3b67b8; color:#ffffff; border:none; border-radius:6px; "
                "padding:6px 10px; font-weight:600;"
            )
            chat.setEnabled(not self.busy)
            chat.clicked.connect(lambda checked=False, p=path: self.chat_installed(p))
            delete = QPushButton("Delete")
            delete.setEnabled(
                not self.busy and self.runtime.process is None and read_model_metadata(path) is not None
            )
            if not read_model_metadata(path):
                delete.setToolTip("This older folder could not be migrated; it is kept safe and remains runnable.")
            delete.clicked.connect(lambda checked=False, p=path: self.delete_model(p))
            actions.addWidget(chat)
            actions.addWidget(delete)
            row.addLayout(actions)
            self.installed_layout.addWidget(card)

    def refresh_conversations(self):
        self.conversation_list.blockSignals(True)
        self.conversation_list.clear()
        selected = None
        for conversation in conversations.list_conversations():
            self.conversation_list.addItem(conversation["title"])
            item = self.conversation_list.item(self.conversation_list.count() - 1)
            item.setData(Qt.ItemDataRole.UserRole, conversation["id"])
            if conversation["id"] == self.current_conversation_id:
                selected = item
        if selected:
            self.conversation_list.setCurrentItem(selected)
        self.conversation_list.blockSignals(False)
        self.delete_chat_button.setEnabled(self.conversation_list.count() > 0)

    def _clear_conversation_view(self):
        while self.chat_messages_layout.count() > 1:
            item = self.chat_messages_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.assistant_widget = None
        self.assistant_text = ""

    def _display_conversation(self, conversation_id):
        self.current_conversation_id = conversation_id
        self._clear_conversation_view()
        for message in conversations.get_messages(conversation_id):
            if message["role"] == "user":
                self._add_user_message(message["content"])
            else:
                self._begin_assistant_message()
                self._append_to_assistant(message["content"])
                if message["status"] == "interrupted":
                    self._append_to_assistant("\n\n[Response interrupted]")
                elif message["status"] == "error":
                    self._append_to_assistant("\n\n[Response ended with an error]")
                elif message["status"] == "limit":
                    self._append_to_assistant("\n\n[Response reached the token limit]")
                self.assistant_widget = None
                self.assistant_text = ""
        self.refresh_conversations()
        self._scroll_chat(True)

    def select_conversation(self, item):
        if not item:
            return
        conversation_id = item.data(Qt.ItemDataRole.UserRole)
        if conversation_id == self.current_conversation_id:
            return
        if self.generation_worker is not None:
            self.pending_conversation_action = ("open", conversation_id)
            self.sidebar_status.setText("Stopping the current response before opening that conversation…")
            self._cancel_active_generation()
            return
        if self.busy:
            self.refresh_conversations()
            return
        self._display_conversation(conversation_id)

    def new_conversation(self):
        if self.generation_worker is not None:
            self.pending_conversation_action = ("new", None)
            self.sidebar_status.setText("Stopping the current response before starting a new chat…")
            self._cancel_active_generation()
            return
        if self.busy:
            return
        conversation_id = conversations.create_conversation()
        self._display_conversation(conversation_id)
        self.sidebar_status.setText("New conversation. The loaded model is unchanged.")

    def delete_conversation(self):
        item = self.conversation_list.currentItem()
        if not item:
            return
        conversation_id = item.data(Qt.ItemDataRole.UserRole)
        title = item.text()
        answer = QMessageBox.question(
            self,
            "Delete conversation?",
            f"Delete ‘{title}’ and all of its messages? This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        if self.generation_worker is not None and conversation_id == self.current_conversation_id:
            self.pending_conversation_action = ("delete", conversation_id)
            self._cancel_active_generation()
            return
        conversations.delete_conversation(conversation_id)
        if conversation_id == self.current_conversation_id:
            remaining = conversations.list_conversations()
            next_id = remaining[0]["id"] if remaining else conversations.create_conversation()
            self._display_conversation(next_id)
        else:
            self.refresh_conversations()

    def find_models(self):
        if self.busy:
            return
        self.busy = True
        self.find_button.setEnabled(False)
        self.candidate_count.setEnabled(False)
        self.sidebar_status.setText("Checking hardware, then searching Hugging Face…")
        self._indeterminate_progress()
        worker = DiscoveryWorker(self.candidate_count.value())
        worker.progress.connect(self.sidebar_status.setText)
        worker.completed.connect(self.show_recommendations)
        worker.failed.connect(self.discovery_failed)
        self._remember_worker(worker)
        worker.start()

    def show_recommendations(self, result):
        # Discovery is complete before buttons are created so they are usable.
        self.busy = False
        models = result.get("models", []) if isinstance(result, dict) else result
        stats = result.get("stats", {}) if isinstance(result, dict) else {}
        self._checked_models = models
        self.recommendations = recommend_models(models, limit=5)
        repository_count = stats.get("public_gguf_repository_count")
        if isinstance(repository_count, int):
            count_text = f"{repository_count:,} public GGUF repositories"
        else:
            count_text = "Public GGUF repository total unavailable from the Hub API"
        self.repository_count_label.setText(
            f"{count_text} · repository count, not unique underlying models"
        )
        self._clear_layout(self.recommendations_layout)
        if not self.recommendations:
            label = QLabel("No suitable models passed the current checks in this search. Try examining more candidates.")
            label.setObjectName("muted")
            label.setWordWrap(True)
            self.recommendations_layout.addWidget(label)
        elif len(self.recommendations) < 3:
            label = QLabel(
                f"Only {len(self.recommendations)} distinct options passed the checks in this search. "
                "Examine more candidates to look for additional choices."
            )
            label.setObjectName("muted")
            label.setWordWrap(True)
            self.recommendations_layout.addWidget(label)
        for recommendation in self.recommendations:
            model = recommendation["model"]
            variant = recommendation["variant"]
            card = QFrame()
            card.setObjectName("modelCard")
            row = QVBoxLayout(card)
            row.setContentsMargins(9, 8, 9, 8)
            role = QLabel(recommendation.get("role", "Recommended"))
            role.setObjectName("section")
            display_name = model.get("name") or model.get("base_model")
            if display_name == "Unknown":
                display_name = model["gguf_repo"]
            name = QLabel(display_name)
            name.setStyleSheet("font-weight:600;")
            name.setWordWrap(True)
            params = model.get("parameters", "Unknown")
            params = f"{params} parameters" if params != "Unknown" else "Parameters unknown"
            meta = QLabel(f"{params} · {variant['quantization']} · {variant['size']} · est. {variant['estimated_memory']}")
            meta.setObjectName("muted")
            meta.setWordWrap(True)
            source = QLabel(
                f"Original base: {model.get('base_model', 'Unknown')} · "
                f"GGUF repository: {model['gguf_repo']}"
            )
            source.setObjectName("muted")
            source.setWordWrap(True)
            reason = QLabel(
                f"{recommendation['status']}. " + " ".join(recommendation["reasons"])
            )
            reason.setObjectName("muted")
            reason.setWordWrap(True)
            button = QPushButton("Use Model")
            button.setObjectName("primary")
            button.setEnabled(not self.busy and len(variant.get("files") or []) == 1)
            button.clicked.connect(lambda checked=False, item=recommendation: self.use_recommended(item))
            row.addWidget(name)
            row.addWidget(role)
            row.addWidget(meta)
            row.addWidget(source)
            row.addWidget(reason)
            row.addWidget(button)
            self.recommendations_layout.addWidget(card)
        self.find_button.setEnabled(True)
        self.candidate_count.setEnabled(True)
        self._hide_progress()
        if stats:
            self.sidebar_status.setText(
                f"Examined {stats.get('repositories_inspected', '?')} of {stats.get('candidate_limit', '?')} repositories; "
                f"{stats.get('hardware_compatible_models', '?')} compatible models found."
            )
        else:
            self.sidebar_status.setText(f"Showing {len(self.recommendations)} recommendations.")
        self.refresh_installed_models()

    @staticmethod
    def _clear_layout(layout):
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def discovery_failed(self, message):
        self.busy = False
        self.find_button.setEnabled(True)
        self.candidate_count.setEnabled(True)
        self._hide_progress()
        self.sidebar_status.setText(f"Could not find models: {message}")

    def chat_installed(self, path):
        path = Path(path)
        if not path.is_file() or not is_standalone_gguf_file(path.name):
            self._model_error("This file is missing or is not a standalone language model.")
            return
        self._begin_model(path, model_display_name(path), "installed")

    def use_recommended(self, recommendation):
        if self.busy and self.generation_worker is None:
            return
        model = recommendation["model"]
        variant = recommendation["variant"]
        existing = self._existing_variant_file(variant, model["gguf_repo"])
        if existing:
            path = existing
        else:
            try:
                plan = inspect_download(model["gguf_repo"], variant)
                path = plan["path"]
            except (ValueError, OSError) as error:
                self._model_error(str(error))
                return
        self._begin_model(
            path,
            model.get("name") if model.get("name") and model.get("name") != "Unknown" else model["gguf_repo"],
            "recommended",
            model,
            variant,
        )

    def _begin_model(self, path, name, source, model=None, variant=None):
        if self.busy and self.generation_worker is None:
            return
        path = Path(path)
        if self.runtime_ready and self.selected_model_path and path.resolve() == Path(self.selected_model_path).resolve():
            self._set_state("Ready", "This model is already ready to chat.")
            return
        self.pending = {"path": path, "name": name, "source": source, "model": model, "variant": variant}
        self.selected_model_path = str(path)
        self.selected_model_name = name
        self.busy = True
        self._set_chat_enabled(False)
        self._set_state("Loading", "Preparing model…")
        self._indeterminate_progress()
        if self.generation_worker is not None:
            self.after_generation = "load"
            self._cancel_active_generation()
            self.sidebar_status.setText("Stopping the current response before switching models…")
            return
        self._stop_old_or_continue()

    def _stop_old_or_continue(self):
        if self.runtime.process:
            self._run_runtime("stop", (), "switch_stop")
        else:
            self._continue_loading()

    def _continue_loading(self):
        pending = self.pending
        path = pending["path"]
        try:
            available = psutil.virtual_memory().available
        except Exception:
            available = None
        self._update_available_memory(available)
        self.memory_warning.hide()

        if pending["source"] == "recommended":
            model, variant = pending["model"], pending["variant"]
            local_copy = self._existing_variant_file(variant, model["gguf_repo"])
            fresh = check_models([{**model, "gguf_files": [variant]}])[0]["gguf_files"][0]
            if fresh["compatibility"] == "Too large" or (
                fresh["disk_status"] == "Insufficient disk space" and not local_copy
            ):
                self._model_error(
                    f"The refreshed estimate is {fresh['compatibility']}; {fresh['disk_status']}. "
                    "Choose a smaller model or free disk space."
                )
                return
            if fresh["compatibility"] == "Fits hardware, but insufficient RAM available now":
                self.memory_warning.setText(
                    f"Only {available / (1024 ** 3):.1f} GB RAM is currently available; "
                    f"this model is estimated to need {fresh['estimated_memory']}. Loading may fail or slow the PC."
                )
                self.memory_warning.show()
            elif fresh["compatibility"] == "Unknown":
                self.memory_warning.setText("Available memory could not be checked. Loading may fail if RAM is low.")
                self.memory_warning.show()
            if local_copy:
                path = local_copy
            else:
                plan = inspect_download(model["gguf_repo"], variant)
                if plan["already_downloaded"]:
                    path = plan["path"]
                else:
                    self.pending["path"] = plan["path"]
                    self.selected_model_path = str(plan["path"])
                    self.sidebar_status.setText(f"Downloading {model.get('name') or model['gguf_repo']}…")
                    self.progress_bar.setRange(0, 100)
                    self.progress_bar.setValue(0)
                    self.progress_bar.show()
                    worker = DownloadWorker(model["gguf_repo"], variant)
                    worker.progress.connect(self.download_progress)
                    worker.completed.connect(self.download_complete)
                    worker.failed.connect(lambda message: self._model_error(f"Download failed: {message}"))
                    self._remember_worker(worker)
                    worker.start()
                    return
        else:
            if not path.is_file():
                self._model_error(f"Downloaded model file was not found: {path}")
                return
            estimate_model = {"gguf_files": [{"size_bytes": path.stat().st_size}]}
            estimate = check_models([estimate_model])[0]["gguf_files"][0]
            if estimate["compatibility"] in {"Fits hardware, but insufficient RAM available now", "Too large", "Unknown"}:
                if available is None:
                    warning = "Available memory could not be checked. Loading may fail if RAM is low."
                else:
                    warning = (
                        f"Only {available / (1024 ** 3):.1f} GB RAM is currently available; "
                        f"this file is estimated to need {estimate['estimated_memory']}. Loading may fail or slow the PC."
                    )
                self.memory_warning.setText(warning)
                self.memory_warning.show()

        self.pending["path"] = path
        self.selected_model_path = str(path)
        self._start_engine(path)

    def _existing_variant_file(self, variant, repo_id=None):
        files = variant.get("files") or []
        size = variant.get("size_bytes")
        if len(files) != 1 or size is None:
            return None
        matches = [
            path for path in local_models()
            if model_source(path)[1] == Path(files[0]).name
            and path.stat().st_size == size
            and (repo_id is None or model_source(path)[0] == repo_id)
        ]
        return matches[0] if len(matches) == 1 else None

    def download_progress(self, received, total):
        if total <= 0:
            self.progress_bar.setRange(0, 0)
            self.sidebar_status.setText(f"Downloading… {received / (1024 ** 2):.0f} MB received")
        else:
            self.progress_bar.setRange(0, 1000)
            self.progress_bar.setValue(min(1000, received * 1000 // total))
            self.sidebar_status.setText(
                f"Downloading… {received / (1024 ** 3):.2f} / {total / (1024 ** 3):.2f} GB"
            )

    def download_complete(self, model_path):
        self.pending["path"] = Path(model_path)
        self.selected_model_path = str(model_path)
        self.refresh_installed_models()
        self.sidebar_status.setText("Download complete. Loading model…")
        self._start_engine(Path(model_path))

    def _start_engine(self, model_path):
        executable = find_llama_server()
        if executable:
            self.sidebar_status.setText("Loading model…")
            batch_size, microbatch_size = automatic_batch_sizes()
            arguments = (
                str(executable), str(model_path), None,
                batch_size, microbatch_size, None,
            )
            self._run_runtime("start", arguments, "load")
            return
        self.sidebar_status.setText("Installing the inference engine…")
        worker = InstallWorker()
        worker.progress.connect(self.sidebar_status.setText)
        worker.completed.connect(lambda exe: self._engine_installed(exe, model_path))
        worker.failed.connect(lambda message: self._model_error(f"Could not install llama.cpp: {message}"))
        self._remember_worker(worker)
        worker.start()

    def _engine_installed(self, executable, model_path):
        self.sidebar_status.setText("Loading model…")
        batch_size, microbatch_size = automatic_batch_sizes()
        arguments = (
            executable, str(model_path), None,
            batch_size, microbatch_size, None,
        )
        self._run_runtime("start", arguments, "load")

    def _run_runtime(self, operation, arguments, purpose):
        self.runtime_purpose = purpose
        worker = RuntimeWorker(operation, self.runtime, *arguments)
        worker.completed.connect(lambda result: self.runtime_succeeded(purpose, result))
        worker.failed.connect(lambda message: self.runtime_failed(purpose, message))
        self._remember_worker(worker)
        worker.start()

    def runtime_succeeded(self, purpose, result):
        if purpose == "switch_stop":
            self.runtime_ready = False
            self._continue_loading()
        elif purpose == "load":
            self.runtime_ready = True
            self._set_chat_enabled(True)
            self.busy = False
            self._hide_progress()
            self.sidebar_status.setText("Ready to chat")
            self._set_state("Ready", "The model is ready. Model fit and speed are estimates, not guarantees.")
        elif purpose == "stop":
            self.runtime_ready = False
            self.busy = False
            self.memory_warning.hide()
            self._set_chat_enabled(False)
            self._hide_progress()
            self.sidebar_status.setText("Model stopped. Select Chat to load it again.")
            self._set_state("Stopped", "The model is still installed and can be loaded again.")
        self.refresh_installed_models()

    def runtime_failed(self, purpose, message):
        self.runtime_ready = self.runtime.process is not None and self.runtime.process.poll() is None
        self._model_error(f"Could not load model: {message}")

    def stop_model(self):
        if self.generation_worker is not None:
            self.after_generation = "stop"
            self.busy = True
            self._set_state("Loading", "Stopping response before unloading the model…")
            self._cancel_active_generation()
            self.sidebar_status.setText("Stopping response before unloading the model…")
            return
        if self.busy or not self.runtime.process:
            return
        self.busy = True
        self._set_chat_enabled(False)
        self._set_state("Loading", "Stopping model and releasing memory…")
        self._run_runtime("stop", (), "stop")

    def send_message(self):
        prompt = self.prompt_input.toPlainText().strip()
        if not prompt or not self.runtime_ready or self.busy or self.generation_worker is not None:
            return

        self._add_user_message(prompt)
        conversations.add_message(self.current_conversation_id, "user", prompt)
        self.refresh_conversations()
        saved_messages = conversations.get_messages(self.current_conversation_id)
        self._begin_assistant_message()
        self.assistant_message_id = conversations.add_message(
            self.current_conversation_id, "assistant", "", self.selected_model_name, "generating"
        )
        self.context_was_trimmed = False
        self.prompt_input.clear()

        self.generation_id += 1
        request_id = self.generation_id
        self.busy = True
        self._set_chat_enabled(False)
        self.sidebar_status.setText("Preparing context and generating response…")
        self.operation_label.setText("Counting this conversation with the loaded model…")
        self._indeterminate_progress()
        self.generating_label.show()
        self.stop_generation_button.setEnabled(True)
        self.stop_generation_button.show()
        worker = GenerationWorker(
            request_id, self.runtime, saved_messages, prompt,
            automatic_response_tokens(context_size=self.runtime.context_size),
            self.current_conversation_id,
        )
        self.generation_worker = worker
        worker.chunk.connect(self.append_generation_chunk)
        worker.status.connect(self.generation_status)
        worker.context_ready.connect(self.generation_context_ready)
        worker.completed.connect(self.generation_completed)
        worker.failed.connect(self.generation_failed)
        self._remember_worker(worker)
        self.stop_button.setEnabled(self.runtime.process is not None)
        worker.start()

    def generation_status(self, request_id, message):
        if request_id != self.generation_id or self.generation_worker is None:
            return
        self.sidebar_status.setText(message)
        self.operation_label.setText(message)

    def generation_context_ready(self, request_id, omitted):
        if request_id != self.generation_id or self.generation_worker is None:
            return
        self.context_was_trimmed = omitted
        self.operation_label.setText(
            "Some earlier messages may not be reflected in this reply."
            if omitted else ""
        )
        self.sidebar_status.setText("Generating response…")

    def append_generation_chunk(self, request_id, text):
        if request_id != self.generation_id or self.generation_worker is None:
            return
        self._append_to_assistant(text)

    def _add_user_message(self, text):
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addStretch(1)
        bubble = QFrame()
        bubble.setStyleSheet("background-color:#303740; border-radius:14px;")
        bubble_layout = QVBoxLayout(bubble)
        bubble_layout.setContentsMargins(13, 9, 13, 9)
        message = QLabel(text)
        message.setWordWrap(True)
        message.setTextFormat(Qt.TextFormat.PlainText)
        message.setStyleSheet("background:transparent; color:#f0f2f5;")
        bubble_layout.addWidget(message)
        max_width = max(220, int(self.chat_scroll.viewport().width() * 0.72))
        metrics = QFontMetrics(message.font())
        longest_line = max(text.splitlines() or [text], key=metrics.horizontalAdvance)
        bubble_width = min(max_width, max(160, metrics.horizontalAdvance(longest_line) + 28))
        bubble.setFixedWidth(bubble_width)
        layout.addWidget(bubble)
        self.chat_messages_layout.insertWidget(self.chat_messages_layout.count() - 1, row)
        self._scroll_chat(True)

    def _begin_assistant_message(self):
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        self.assistant_widget = QLabel("")
        self.assistant_widget.setObjectName("assistantText")
        self.assistant_widget.setWordWrap(True)
        self.assistant_widget.setTextFormat(Qt.TextFormat.PlainText)
        self.assistant_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.assistant_widget.setStyleSheet("background:transparent; color:#e5e7eb; padding:6px 12px;")
        layout.addWidget(self.assistant_widget)
        self.chat_messages_layout.insertWidget(self.chat_messages_layout.count() - 1, row)
        self.assistant_text = ""
        self._scroll_chat(True)

    def _append_to_assistant(self, text):
        if self.assistant_widget is None:
            return
        self.assistant_text += text
        self.assistant_widget.setText(self.assistant_text)
        self._scroll_chat()

    def _scroll_chat(self, force_bottom=False):
        scrollbar = self.chat_scroll.verticalScrollBar()
        old_value = scrollbar.value()
        follow_output = force_bottom or old_value >= scrollbar.maximum() - 2

        def restore_position():
            if follow_output:
                scrollbar.setValue(scrollbar.maximum())
            else:
                scrollbar.setValue(min(old_value, scrollbar.maximum()))

        QTimer.singleShot(0, restore_position)

    def _finish_assistant_message(self, error=None, status="complete"):
        if self.assistant_widget is not None:
            if error:
                self._append_to_assistant(f"\n\nResponse error: {error}")
            if self.assistant_message_id is not None:
                conversations.update_message(self.assistant_message_id, self.assistant_text, status)
                self.assistant_message_id = None
            self.assistant_widget = None
            self.assistant_text = ""
            self._scroll_chat()
            self.refresh_conversations()

    def _cancel_active_generation(self):
        worker = self.generation_worker
        if worker is None:
            return
        self.generation_id += 1  # Ignore any chunks already queued for the UI.
        worker.cancel()
        self.stop_generation_button.setEnabled(False)
        self.generating_label.setText("Stopping response…")

    def cancel_generation(self):
        if self.generation_worker is None:
            return
        self.after_generation = None
        self.sidebar_status.setText("Stopping response…")
        self._cancel_active_generation()

    def generation_completed(self, request_id, cancelled, finish_reason):
        if self.generation_worker is None or self.generation_worker.request_id != request_id:
            return
        worker = self.generation_worker
        self.assistant_text = worker.text
        if self.assistant_widget is not None:
            self.assistant_widget.setText(self.assistant_text)
        result_status = "interrupted" if cancelled else ("limit" if finish_reason == "length" else "complete")
        self._finish_assistant_message(status=result_status)
        self.generation_worker = None
        self.generating_label.hide()
        self.generating_label.setText("Generating…")
        self.stop_generation_button.hide()
        self._hide_progress()

        next_action = self.after_generation
        self.after_generation = None
        conversation_action = self.pending_conversation_action
        self.pending_conversation_action = None
        if conversation_action:
            action, conversation_id = conversation_action
            if action == "open":
                self._display_conversation(conversation_id)
            elif action == "new":
                self._display_conversation(conversations.create_conversation())
            elif action == "delete":
                conversations.delete_conversation(conversation_id)
                remaining = conversations.list_conversations()
                next_id = remaining[0]["id"] if remaining else conversations.create_conversation()
                self._display_conversation(next_id)
            self.busy = False
            self.runtime_ready = self.runtime.process is not None and self.runtime.process.poll() is None
            self._set_chat_enabled(self.runtime_ready)
            if self.runtime_ready:
                self.sidebar_status.setText("Response stopped. Ready to chat.")
                self._set_state("Ready")
            else:
                self.sidebar_status.setText("Response stopped. Select Chat to load a model.")
                self._set_state("Stopped")
            return
        if next_action == "load":
            self._stop_old_or_continue()
            return
        if next_action == "stop":
            self._run_runtime("stop", (), "stop")
            return

        self.busy = False
        self.runtime_ready = self.runtime.process is not None and self.runtime.process.poll() is None
        if self.runtime_ready:
            self._set_chat_enabled(True)
            self.sidebar_status.setText("Response stopped. Ready to chat." if cancelled else "Ready to chat")
            if cancelled:
                self.operation_label.setText("Response stopped; the partial answer is preserved.")
            elif finish_reason == "length":
                self.operation_label.setText("Response reached the token limit; this answer may be incomplete.")
            else:
                self.operation_label.setText(
                    "Some earlier messages may not be reflected in this reply."
                    if self.context_was_trimmed else ""
                )
            self._set_state("Ready")
        else:
            self._set_chat_enabled(False)
            self.sidebar_status.setText("The model server stopped. Select Chat to load it again.")
            self._set_state("Error", "The model server stopped during generation.")
        self.refresh_installed_models()

    def generation_failed(self, request_id, message):
        if self.generation_worker is None or self.generation_worker.request_id != request_id:
            return
        self._finish_assistant_message(message, "error")
        self.generation_worker = None
        self.generating_label.hide()
        self.generating_label.setText("Generating…")
        self.stop_generation_button.hide()
        self._hide_progress()
        self.after_generation = None
        self.busy = False
        self.runtime_ready = self.runtime.process is not None and self.runtime.process.poll() is None
        self.sidebar_status.setText(f"Response failed: {message}")
        if self.runtime_ready:
            self._set_chat_enabled(True)
            self._set_state("Ready", "The response failed. You can try another message.")
        else:
            self._set_chat_enabled(False)
            self._set_state("Error", f"Response failed: {message}")
        self.refresh_installed_models()

    def _set_chat_enabled(self, enabled):
        active = enabled and self.runtime_ready
        self.prompt_input.setEnabled(active)
        self.send_button.setEnabled(active)
        can_stop = self.runtime.process is not None and (not self.busy or self.generation_worker is not None)
        self.stop_button.setEnabled(can_stop)

    def _model_error(self, message):
        self.busy = False
        if self.runtime.process is None:
            self.runtime_ready = False
        self._hide_progress()
        self._set_chat_enabled(self.runtime_ready)
        self.sidebar_status.setText(message)
        self._set_state("Error", message)
        self.find_button.setEnabled(True)
        self.refresh_installed_models()

    def _update_available_memory(self, available):
        if available is None:
            return
        summary = self.hardware_summary.text()
        if "available " in summary:
            start = summary.find("available ") + len("available ")
            end = summary.find(")", start)
            if end >= 0:
                self.hardware_summary.setText(
                    summary[:start] + f"{available / (1024 ** 3):.1f} GB" + summary[end:]
                )

    def _indeterminate_progress(self):
        self.progress_bar.setRange(0, 0)
        self.progress_bar.show()

    def _hide_progress(self):
        self.progress_bar.hide()
        self.progress_bar.setRange(0, 100)

    def delete_model(self, path):
        if self.busy or self.runtime.process is not None:
            QMessageBox.information(
                self,
                "Model deletion unavailable",
                "Finish the current operation and unload the model before deleting downloaded models.",
            )
            return
        path = Path(path).resolve()
        try:
            freed = model_folder_size(path)
        except (OSError, ValueError) as error:
            self.sidebar_status.setText(f"Could not delete model: {error}")
            return
        answer = QMessageBox.question(
            self,
            "Delete model?",
            f"Delete {model_display_name(path)} and its dedicated folder?\n\n"
            f"About {freed / (1024 ** 3):.2f} GB will be freed.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            delete_managed_model(path)
            if self.selected_model_path and path == Path(self.selected_model_path).resolve():
                self.selected_model_path = None
                self.selected_model_name = ""
                self._set_state("Stopped", "Model deleted.")
            self.refresh_installed_models()
        except (OSError, ValueError) as error:
            self.sidebar_status.setText(f"Could not delete model: {error}")

    def closeEvent(self, event):
        generation = self.generation_worker
        if generation is not None:
            generation.cancel()
        self.runtime.stop()
        for worker in self.workers:
            if worker.isRunning():
                worker.wait()
        if generation is not None and self.assistant_message_id is not None:
            conversations.update_message(
                self.assistant_message_id,
                generation.text,
                "interrupted" if generation.result_status == "generating" else generation.result_status,
            )
            self.assistant_message_id = None
        if self.hardware_worker.isRunning():
            self.hardware_worker.wait()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    windows_root = os.environ.get("WINDIR") or os.environ.get("SystemRoot") or str(Path.home())
    font_path = Path(windows_root) / "Fonts" / "segoeui.ttf"
    if font_path.is_file():
        QFontDatabase.addApplicationFont(str(font_path))
        app.setFont(QFont("Segoe UI", 10))
    window = HardwareWindow()
    window.show()
    sys.exit(app.exec())
