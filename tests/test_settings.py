"""Scope precedence, model validation, and confirmation setting boundaries."""

import unittest
from types import SimpleNamespace

from ..settings import access_error, read_settings, settings_revision, validate_update


class SettingsTests(unittest.TestCase):
    def event(self, group="group-one", sender="caller", platform="platform-one"):
        return SimpleNamespace(
            get_sender_id=lambda: sender,
            get_group_id=lambda: group,
            get_platform_name=lambda: platform,
            plugins_name=None,
            is_admin=lambda: True,
        )

    def test_enabled_empty_whitelist_denies_all_groups_even_for_admins(self):
        settings = read_settings({"group_whitelist_enabled": True})
        self.assertIsNotNone(access_error(settings, self.event(), "bridge"))
        settings["group_whitelist"] = ["group-one"]
        self.assertIsNone(access_error(settings, self.event(), "bridge"))

    def test_blacklist_takes_precedence_and_platform_ids_do_not_collide(self):
        settings = read_settings(
            {
                "group_whitelist_enabled": True,
                "group_whitelist": ["group-one"],
                "group_blacklist": ["platform-one:group-one"],
            }
        )
        self.assertIsNotNone(access_error(settings, self.event(), "bridge"))
        self.assertIsNone(
            access_error(settings, self.event(platform="platform-two"), "bridge")
        )

    def test_user_blacklist_applies_to_private_and_group_messages(self):
        settings = read_settings({"user_blacklist": ["platform-one:caller"]})
        for group in ("group-one", ""):
            self.assertIsNotNone(
                access_error(settings, self.event(group=group), "bridge")
            )
        self.assertIsNone(
            access_error(settings, self.event(sender="someone-else"), "bridge")
        )

    def test_private_chat_scope_is_separate_from_group_scope(self):
        settings = read_settings({"allow_private_chat": False})
        self.assertIsNotNone(access_error(settings, self.event(group=""), "bridge"))
        self.assertIsNone(access_error(settings, self.event(), "bridge"))
        settings.update(allow_private_chat=True, group_whitelist_enabled=True)
        self.assertIsNone(access_error(settings, self.event(group=""), "bridge"))

    def test_profile_or_plugin_disable_cannot_be_overridden_by_whitelisting(self):
        event = self.event()
        event.plugins_name = []
        self.assertIsNotNone(access_error(read_settings({}), event, "bridge"))
        event.plugins_name = ["*"]
        self.assertIsNone(access_error(read_settings({}), event, "bridge"))
        self.assertIsNotNone(
            access_error(read_settings({"enabled": False}), event, "bridge")
        )

    def test_partial_model_updates_preserve_scope_and_allow_follow_current(self):
        current = read_settings(
            {"group_blacklist": ["group-one"], "execution_provider_id": "chat-one"}
        )
        updated = validate_update(
            current,
            {"analysis_provider_id": "analysis-one"},
            {"analysis-one", "chat-one"},
        )
        self.assertEqual(updated["group_blacklist"], ["group-one"])
        self.assertEqual(updated["execution_provider_id"], "chat-one")
        updated = validate_update(updated, {"execution_provider_id": ""}, set())
        self.assertEqual(updated["execution_provider_id"], "")
        with self.assertRaises(ValueError):
            validate_update(current, {"analysis_provider_id": "invented"}, {"chat-one"})

    def test_malformed_controls_are_rejected_instead_of_becoming_truthy(self):
        for update in (
            {"confirmation_enabled": "false"},
            {"group_whitelist": "one"},
            {"user_blacklist": [False]},
            {"group_blacklist": ["one two"]},
            {"unrecognized": True},
        ):
            with self.assertRaises(ValueError):
                validate_update(read_settings({}), update, set())

    def test_analysis_model_changes_do_not_invalidate_runtime_approvals(self):
        first = read_settings({})
        second = read_settings({"analysis_provider_id": "analysis-two"})
        self.assertNotEqual(settings_revision(first), settings_revision(second))
        self.assertEqual(
            settings_revision(first, controls_only=True),
            settings_revision(second, controls_only=True),
        )
        second["analysis_timeout"] = 900
        self.assertEqual(
            settings_revision(first, controls_only=True),
            settings_revision(second, controls_only=True),
        )
        second["confirmation_enabled"] = False
        self.assertNotEqual(
            settings_revision(first, controls_only=True),
            settings_revision(second, controls_only=True),
        )

    def test_analysis_timeout_validation(self):
        self.assertEqual(read_settings({})["analysis_timeout"], 600)
        for value in (False, "600", 0, 29, 1801, 60.5):
            with self.assertRaises(ValueError):
                validate_update(read_settings({}), {"analysis_timeout": value}, set())
        for value in (30, 900, 1800):
            self.assertEqual(
                read_settings({"analysis_timeout": value})["analysis_timeout"], value
            )


if __name__ == "__main__":
    unittest.main()
