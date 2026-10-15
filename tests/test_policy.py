"""Approval replay, expiry, conversation isolation, and policy invalidation."""

import unittest
from types import SimpleNamespace

from ..policy import PendingCalls, call_policy


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.clock = 100.0
        self.pending = PendingCalls(now=lambda: self.clock)
        self.event = self.event_for("caller", "platform:group:one")
        self.snapshot = {"command": "report", "parameters": {"account": "Sample#1"}}

    @staticmethod
    def event_for(sender, origin):
        return SimpleNamespace(get_sender_id=lambda: sender, unified_msg_origin=origin)

    def test_other_user_or_conversation_cannot_consume_approval(self):
        code = self.pending.request(self.event, "cap_example", {}, self.snapshot)
        for event in (
            self.event_for("other", "platform:group:one"),
            self.event_for("caller", "platform:group:two"),
        ):
            with self.assertRaises(ValueError):
                self.pending.consume(event, code)
        self.assertEqual(
            self.pending.consume(self.event, code)["snapshot"], self.snapshot
        )
        with self.assertRaises(ValueError):
            self.pending.consume(self.event, code)

    def test_expired_approval_cannot_execute(self):
        code = self.pending.request(self.event, "cap_example", {}, self.snapshot)
        self.clock += 120
        with self.assertRaises(ValueError):
            self.pending.consume(self.event, code)
        self.assertEqual(self.pending.items, {})

    def test_duplicate_preview_does_not_extend_lifetime_or_change_arguments(self):
        arguments = {"query": "original"}
        code = self.pending.request(self.event, "cap_example", arguments, self.snapshot)
        arguments["query"] = "mutated"
        self.snapshot["parameters"]["account"] = "mutated"
        self.clock += 100
        frozen = {"command": "report", "parameters": {"account": "Sample#1"}}
        duplicate = self.pending.request(self.event, "cap_example", {}, frozen)
        self.assertEqual(duplicate, code)
        self.assertEqual(self.pending.items[code]["arguments"]["query"], "original")
        self.clock += 20
        with self.assertRaises(ValueError):
            self.pending.consume(self.event, code)

    def test_source_changes_reset_explicit_allowance(self):
        state = {"policies": {"cap_example": {"mode": "direct", "fingerprint": "v1"}}}
        self.assertEqual(call_policy(state, "cap_example", "v1"), ("direct", True))
        self.assertEqual(call_policy(state, "cap_example", "v2"), ("confirm", False))
        self.assertEqual(call_policy({}, "cap_example", "v1"), ("confirm", False))

    def test_approval_storage_is_bounded_per_caller(self):
        codes = [
            self.pending.request(self.event, "cap_example", {}, {"value": i})
            for i in range(5)
        ]
        with self.assertRaises(ValueError):
            self.pending.consume(self.event, codes[0])
        self.assertEqual(len(self.pending.items), 4)

    def test_source_update_keeps_admin_restriction_until_re_review(self):
        state = {
            "policies": {"cap_example": {"mode": "admin_confirm", "fingerprint": "v1"}}
        }
        self.assertEqual(
            call_policy(state, "cap_example", "v2"), ("admin_confirm", False)
        )


if __name__ == "__main__":
    unittest.main()
