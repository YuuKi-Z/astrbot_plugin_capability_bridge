"""Administrator policies and short-lived, caller-bound execution approvals."""

import copy
import json
import secrets
import time

POLICY_MODES = {"direct", "confirm", "admin_confirm"}
CONFIRMATION_TTL = 120


def call_policy(state: dict, capability_id: str, fingerprint: str) -> tuple[str, bool]:
    """Trust only dashboard decisions made against the current plugin source."""
    saved = state.get("policies", {}).get(capability_id, {})
    if (
        isinstance(saved, dict)
        and saved.get("fingerprint") == fingerprint
        and isinstance(saved.get("mode"), str)
        and saved.get("mode") in POLICY_MODES
    ):
        return saved["mode"], True
    if isinstance(saved, dict) and saved.get("mode") == "admin_confirm":
        # Source changes revoke prior allowances without removing restrictions.
        return "admin_confirm", False
    return "confirm", False


def caller_key(event) -> tuple[str, str]:
    """Bind approvals to the real caller and the exact conversation."""
    sender, origin = event.get_sender_id(), event.unified_msg_origin
    if not sender or not origin:
        raise ValueError("无法确认当前调用者或会话，请使用原插件指令")
    return str(sender), str(origin)


class PendingCalls:
    """Keep one-use approvals in memory; restarts invalidate all pending calls."""

    def __init__(self, now=time.monotonic):
        self.now = now
        self.items = {}

    def prune(self):
        """Drop expired requests without retaining user data on disk."""
        current = self.now()
        for code in list(self.items):
            if self.items[code]["expires"] <= current:
                self.items.pop(code)

    def request(self, event, capability_id, arguments, snapshot) -> str:
        """Reuse identical pending calls and bound storage for busy group chats."""
        self.prune()
        owner = caller_key(event)
        # JSON rejects non-finite values and ensures an immutable comparison shape.
        frozen = json.loads(json.dumps(snapshot, ensure_ascii=False, allow_nan=False))
        for code, item in self.items.items():
            if item["owner"] == owner and item["snapshot"] == frozen:
                return code
        owned = [code for code, item in self.items.items() if item["owner"] == owner]
        if len(owned) >= 4:
            self.items.pop(owned[0])
        if len(self.items) >= 256:
            self.items.pop(next(iter(self.items)))
        code = secrets.token_urlsafe(6)
        while code in self.items:
            code = secrets.token_urlsafe(6)
        self.items[code] = {
            "owner": owner,
            "capability_id": capability_id,
            "arguments": copy.deepcopy(arguments),
            "snapshot": frozen,
            "expires": self.now() + CONFIRMATION_TTL,
        }
        return code

    def consume(self, event, code: str) -> dict:
        """Consume only the matching caller's request, before executing anything."""
        self.prune()
        item = self.items.get(code)
        if not item or item["owner"] != caller_key(event):
            raise ValueError("确认码无效、已过期，或不属于你在当前会话发起的操作")
        return self.items.pop(code)
