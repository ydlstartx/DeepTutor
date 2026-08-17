#!/usr/bin/env python3
"""Repair message parent links corrupted by the optimistic-id collision bug.

Bug context: the web chat used to mint optimistic message ids with
``-Date.now()``. A turn's user row and assistant placeholder, dispatched in
the same tick, often shared one id; the turn-end reconcile then stamped both
rows with the *user* message's persisted id. The next send picked that shared
id as its parent, so a user message ended up parented to the *previous user
message* instead of the assistant reply — hiding the reply from the visible
branch path and from the LLM context chain.

Repair rule (the corruption signature): a ``user`` message whose parent is
also a ``user`` message. The correct parent is the assistant reply to that
parent (the assistant row whose ``parent_message_id`` is the parent's id;
when several exist — regenerate branches — the latest one wins).

Usage:
    python scripts/repair_message_parents.py [DB_PATH]          # dry-run
    python scripts/repair_message_parents.py [DB_PATH] --apply  # write fixes

Defaults to data/user/chat_history.db. Safe to re-run: repaired rows no
longer match the signature.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sqlite3
import sys

DEFAULT_DB = Path("data/user/chat_history.db")


def find_repairs(conn: sqlite3.Connection) -> list[tuple[int, int, int]]:
    """Return (message_id, wrong_parent_id, correct_parent_id) triples."""
    rows = conn.execute(
        "SELECT id, session_id, role, parent_message_id FROM messages ORDER BY id"
    ).fetchall()
    by_id = {row[0]: row for row in rows}

    # parent_id -> assistant child ids (ascending; last = latest branch tip)
    assistant_children: dict[int, list[int]] = {}
    for msg_id, _session, role, parent_id in rows:
        if role == "assistant" and parent_id is not None:
            assistant_children.setdefault(parent_id, []).append(msg_id)

    repairs: list[tuple[int, int, int]] = []
    for msg_id, _session, role, parent_id in rows:
        if role != "user" or parent_id is None:
            continue
        parent = by_id.get(parent_id)
        if parent is None or parent[2] != "user":
            continue  # parent missing or not a user row — not our signature
        candidates = assistant_children.get(parent_id, [])
        if not candidates:
            print(
                f"  ! message {msg_id}: parent user row {parent_id} has no "
                "assistant reply; leaving unchanged"
            )
            continue
        repairs.append((msg_id, parent_id, candidates[-1]))
    return repairs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "db",
        nargs="?",
        default=str(DEFAULT_DB),
        help=f"path to chat_history.db (default: {DEFAULT_DB})",
    )
    parser.add_argument("--apply", action="store_true", help="write the repairs (default: dry-run)")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.is_file():
        print(f"error: database not found: {db_path}", file=sys.stderr)
        return 1

    conn = sqlite3.connect(str(db_path), timeout=5)
    try:
        repairs = find_repairs(conn)
        if not repairs:
            print("no corrupted parent links found")
            return 0

        for msg_id, wrong_parent, correct_parent in repairs:
            print(f"  message {msg_id}: parent {wrong_parent} -> {correct_parent}")

        if not args.apply:
            print(f"\ndry-run: {len(repairs)} repair(s) pending; re-run with --apply")
            return 0

        with conn:
            conn.executemany(
                "UPDATE messages SET parent_message_id = ? WHERE id = ?",
                [(correct, msg_id) for msg_id, _wrong, correct in repairs],
            )
        print(f"\napplied {len(repairs)} repair(s)")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
