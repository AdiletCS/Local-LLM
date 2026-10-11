"""Small SQLite store for local chat conversations."""

import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone

from settings import APP_DATA_DIR


APP_DATA = APP_DATA_DIR
DATABASE = APP_DATA / "conversations.sqlite3"
CONTEXT_SIZE = 2048
MAX_OUTPUT_TOKENS = 512
RESPONSE_RESERVE = MAX_OUTPUT_TOKENS
TOKEN_SAFETY_MARGIN = 64


@contextmanager
def _connect():
    APP_DATA.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def initialize():
    with _connect() as db:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY, title TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                role TEXT NOT NULL, content TEXT NOT NULL, model TEXT,
                status TEXT NOT NULL DEFAULT 'complete', created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS conversation_summaries (
                conversation_id TEXT PRIMARY KEY REFERENCES conversations(id) ON DELETE CASCADE,
                content TEXT NOT NULL, through_message_id INTEGER NOT NULL, updated_at TEXT NOT NULL
            );
        """)
        db.execute("UPDATE messages SET status = 'interrupted' WHERE status = 'generating'")


def create_conversation():
    conversation_id = str(uuid.uuid4())
    now = _now()
    with _connect() as db:
        db.execute(
            "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (conversation_id, "New chat", now, now),
        )
    return conversation_id


def list_conversations():
    with _connect() as db:
        return [dict(row) for row in db.execute(
            "SELECT id, title, updated_at FROM conversations ORDER BY updated_at DESC"
        )]


def get_messages(conversation_id):
    with _connect() as db:
        return [dict(row) for row in db.execute(
            "SELECT id, role, content, model, status, created_at FROM messages "
            "WHERE conversation_id = ? ORDER BY id", (conversation_id,)
        )]


def add_message(conversation_id, role, content, model=None, status="complete"):
    now = _now()
    with _connect() as db:
        cursor = db.execute(
            "INSERT INTO messages (conversation_id, role, content, model, status, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (conversation_id, role, content, model, status, now),
        )
        if role == "user":
            conversation = db.execute(
                "SELECT title FROM conversations WHERE id = ?", (conversation_id,)
            ).fetchone()
            if conversation and conversation["title"] == "New chat":
                title = " ".join(content.split())[:52] or "New chat"
                db.execute("UPDATE conversations SET title = ? WHERE id = ?", (title, conversation_id))
        db.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conversation_id))
        return cursor.lastrowid


def update_message(message_id, content, status):
    with _connect() as db:
        row = db.execute(
            "SELECT conversation_id FROM messages WHERE id = ?", (message_id,)
        ).fetchone()
        db.execute("UPDATE messages SET content = ?, status = ? WHERE id = ?", (content, status, message_id))
        if row:
            db.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (_now(), row["conversation_id"]))


def delete_conversation(conversation_id):
    with _connect() as db:
        db.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))


def get_summary(conversation_id):
    with _connect() as db:
        row = db.execute(
            "SELECT content, through_message_id FROM conversation_summaries WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
    return dict(row) if row else {"content": "", "through_message_id": 0}


def save_summary(conversation_id, content, through_message_id):
    with _connect() as db:
        db.execute(
            "INSERT INTO conversation_summaries (conversation_id, content, through_message_id, updated_at) "
            "VALUES (?, ?, ?, ?) ON CONFLICT(conversation_id) DO UPDATE SET "
            "content = excluded.content, through_message_id = excluded.through_message_id, "
            "updated_at = excluded.updated_at",
            (conversation_id, content, through_message_id, _now()),
        )


def _complete_turns(messages, after_id=0):
    prior = [item for item in messages if item.get("role") in {"user", "assistant"}]
    turns = []
    index = 0
    while index + 1 < len(prior):
        user, assistant = prior[index], prior[index + 1]
        if user.get("role") == "user" and assistant.get("role") == "assistant":
            if assistant.get("status") == "complete" and assistant.get("id", 0) > after_id:
                turns.append([user, assistant])
            index += 2
        else:
            index += 1
    return turns


def summarization_batch(messages, selected_ids, summary, through_message_id, context_size, token_counter=None):
    """Return a bounded batch of omitted, complete turns to merge into the saved summary."""
    latest_user = messages[-1].get("content", "") if messages and messages[-1].get("role") == "user" else ""
    selected_ids = set(selected_ids)
    omitted_turns = [
        turn for turn in _complete_turns(messages[:-1], through_message_id)
        if turn[-1].get("id") not in selected_ids
    ]
    if not omitted_turns:
        return None

    source_budget = max(256, context_size // 2)
    chosen = []

    def prompt_for(turns):
        previous = f"Existing summary:\n{summary}\n\n" if summary else ""
        source = "\n\n".join(
            f"User: {turn[0]['content']}\nAssistant: {turn[1]['content']}" for turn in turns
        )
        return [
            {"role": "system", "content": (
                "Update a concise memory of an ongoing conversation. Merge the existing summary with the "
                "older complete turns. Preserve names, preferences, decisions, facts, and unresolved requests. "
                "Do not invent details. Return only the updated summary."
            )},
            {"role": "user", "content": previous + "Older complete turns to incorporate:\n" + source},
        ]

    for turn in omitted_turns:
        proposed = chosen + [turn]
        prompt = prompt_for(proposed)
        try:
            count = token_counter(prompt) if token_counter else (
                128 + sum(len(item["content"].encode("utf-8")) + 16 for item in prompt),
                "conservative UTF-8 estimate",
            )
            token_count = count[0] if isinstance(count, tuple) else count
        except Exception:
            token_count = sum(len(item["content"].encode("utf-8")) for item in prompt)
        if token_count > source_budget:
            break
        chosen = proposed
    if not chosen:
        return None
    return {
        "messages": prompt_for(chosen),
        "through_message_id": chosen[-1][-1].get("id", through_message_id),
    }


def build_context(
    messages,
    latest_user,
    context_size=CONTEXT_SIZE,
    token_counter=None,
    response_reserve=RESPONSE_RESERVE,
    summary="",
    summary_through_id=0,
    return_ids=False,
):
    """Keep the latest user message and newest complete turns within the model context."""
    def fallback_count(request_messages):
        return 128 + sum(
            len(item.get("content", "").encode("utf-8")) + 16
            for item in request_messages
        ), "conservative UTF-8 estimate"

    def count(request_messages):
        if token_counter:
            try:
                return token_counter(request_messages)
            except Exception:
                pass
        return fallback_count(request_messages)

    summary_message = [{"role": "system", "content": f"Earlier conversation summary:\n{summary}"}] if summary else []
    latest = [{"role": "user", "content": latest_user}]
    latest_cost, token_method = count(summary_message + latest)
    latest_cost += TOKEN_SAFETY_MARGIN
    if latest_cost > context_size - response_reserve:
        raise ValueError(
            f"This message and its chat formatting need about {latest_cost} tokens, "
            f"but only {context_size - response_reserve} are available before the response reserve. "
            "Shorten it and try again."
        )

    prior = messages[:-1] if messages and messages[-1].get("role") == "user" else messages
    turns = _complete_turns(prior, summary_through_id)

    selected = []
    used = latest_cost
    selected_prompt = summary_message + latest
    for turn in reversed(turns):
        proposed_turns = [turn] + list(reversed(selected))
        proposed_messages = [
            {"role": item["role"], "content": item["content"]}
            for prior_turn in proposed_turns for item in prior_turn
        ]
        proposed_messages = summary_message + proposed_messages + latest
        prompt_count, method = count(proposed_messages)
        if prompt_count + TOKEN_SAFETY_MARGIN > context_size - response_reserve:
            break
        selected.append(turn)
        used = prompt_count + TOKEN_SAFETY_MARGIN
        selected_prompt = proposed_messages
        token_method = method
    selected.reverse()
    request_messages = selected_prompt
    included_ids = {item["id"] for turn in selected for item in turn if "id" in item}
    omitted = any(
        item.get("id", 0) > summary_through_id and item.get("id") not in included_ids
        for item in prior
    )
    result = (request_messages, omitted, max(0, used - TOKEN_SAFETY_MARGIN), token_method)
    return result + (included_ids,) if return_ids else result
