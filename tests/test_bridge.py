"""Behavioral checks using real AstrBot filters and in-memory plugin fixtures."""

import asyncio
import copy
import functools
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from astrbot.api.message_components import At
from astrbot.core.message.message_event_result import MessageChain, MessageEventResult
from astrbot.core.star.filter.command import CommandFilter
from astrbot.core.star.filter.command_group import CommandGroupFilter
from astrbot.core.star.filter.permission import PermissionType, PermissionTypeFilter

from ..analysis import generate
from ..main import CapabilityBridge
from ..plans import validate_plan
from ..policy import PendingCalls
from ..runtime import CapabilityError, execute, resolve_member
from ..scanner import (
    _collect,
    analysis_batches,
    fingerprint,
    inspect_helpers,
    parameter_schema,
)
from ..settings import read_settings, settings_revision


class FixturePlugin:
    """A generic account-backed plugin, deliberately unrelated to any game."""

    def __init__(self):
        self.calls = []
        self.bindings = {"111111": "Caller#1", "222222": "Target#2"}

    def _get_account(self, user_id):
        return self.bindings.get(str(user_id))

    def _missing_hint(self):
        return "请先使用 /account bind <ID> 绑定账号"

    def _unsafe_hint(self):
        self.calls.append("bad")
        return "unsafe"

    def _get_mutating(self, user_id):
        self.calls.append(user_id)
        return "bad"

    async def report(self, event, account: str | None = None):
        self.calls.append((event.get_sender_id(), account))
        yield MessageEventResult().message("report: " + str(account))
        await event.send(MessageChain().message("native image placeholder"))

    async def text_query(self, incoming):
        keyword = incoming.get_message_str().replace("catalog find", "", 1).strip()
        self.calls.append(
            (incoming.get_sender_id(), keyword, incoming.message_obj.message_str)
        )
        yield MessageEventResult().message("query: " + keyword)


class FakeEvent:
    """Capture messages without connecting to any chat platform."""

    def __init__(self):
        self._extras = {}
        self._has_send_oper = False
        self.plugins_name = None
        self.unified_msg_origin = "test:GroupMessage:1"
        self.message_str = "帮我查群友的报告"
        self.message_obj = SimpleNamespace(message=[], message_str=self.message_str)
        self.sent = []
        self.admin = False
        self.bot = SimpleNamespace(
            call_action=AsyncMock(
                return_value=[
                    {"user_id": 111111, "nickname": "小一"},
                    {"user_id": 222222, "nickname": "目标", "card": "小二"},
                    {"user_id": 333333, "nickname": "未绑定"},
                ]
            )
        )

    def get_sender_id(self):
        return "111111"

    def get_sender_name(self):
        return "小一"

    def get_group_id(self):
        return "888888"

    def get_platform_name(self):
        return "aiocqhttp"

    def is_admin(self):
        return self.admin

    def get_extra(self, key=None, default=None):
        return self._extras if key is None else self._extras.get(key, default)

    def set_extra(self, key, value):
        self._extras[key] = value

    def get_messages(self):
        return self.message_obj.message

    def get_message_str(self):
        return self.message_str

    def clear_result(self):
        self._result = None

    def stop_event(self):
        self.stopped = True

    def plain_result(self, text):
        return MessageEventResult().message(text)

    async def send(self, chain):
        self.sent.append(chain.get_plain_text())
        self._has_send_oper = True


class BridgeTests(unittest.IsolatedAsyncioTestCase):
    """Verify target identity, permission boundaries, stale plans, and output."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        (root / "main.py").write_text("version=1", encoding="utf-8")
        self.plugin = FixturePlugin()
        self.event = FakeEvent()
        self.context = SimpleNamespace(get_config=lambda **kwargs: {})
        command_filter = CommandFilter("report")
        handler = SimpleNamespace(
            enabled=True, handler=self.plugin.report, event_filters=[command_filter]
        )
        self.record = {
            "name": "fixture",
            "active": True,
            "root": root,
            "metadata": SimpleNamespace(
                activated=True, star_cls=self.plugin, name="fixture", version="1"
            ),
            "helpers": inspect_helpers(self.plugin)[0],
            "commands": {
                "fixture.report": {
                    "id": "fixture.report",
                    "command": "account report",
                    "permission": "member",
                    "enabled": True,
                    "parameters": parameter_schema(self.plugin.report),
                    "handler": handler,
                }
            },
        }
        self.context.get_registered_star = lambda name: self.record["metadata"]
        self.digest = fingerprint(root, "1")
        self.record["fingerprint"] = self.digest
        self.raw = {
            "name": "账号报告",
            "description": "按绑定账号查询目标成员报告",
            "examples": ["帮我查小二的账号报告"],
            "command_id": "fixture.report",
            "input_schema": {
                "type": "object",
                "properties": {
                    "target_user": {"type": "string", "default": "我"},
                    "account": {"type": "string"},
                },
            },
            "argument_map": {"account": {"input": "account"}},
            "subject_binding": {
                "input": "target_user",
                "lookup_method": "_get_account",
                "target_parameter": "account",
                "missing_method": "_missing_hint",
            },
        }
        self.cap = validate_plan(self.raw, self.record)
        self.session_patch = patch(
            "astrbot.core.star.session_plugin_manager.SessionPluginManager.filter_handlers_by_session",
            AsyncMock(side_effect=lambda event, handlers: handlers),
        )
        self.session_patch.start()
        self.addCleanup(self.session_patch.stop)

    async def test_other_member_keeps_caller_identity_and_delivers_native_output(self):
        result = await execute(
            self.context,
            self.event,
            self.cap,
            self.record,
            {"target_user": "小二"},
            self.digest,
        )
        self.assertEqual(self.plugin.calls, [("111111", "Target#2")])
        self.assertEqual(len(self.event.sent), 2)
        self.assertTrue(result["do_not_repeat"])
        self.assertEqual(self.event.message_str, "帮我查群友的报告")

    async def test_missing_binding_uses_original_hint(self):
        result = await execute(
            self.context,
            self.event,
            self.cap,
            self.record,
            {"target_user": "未绑定"},
            self.digest,
        )
        self.assertEqual(result["status"], "missing_binding")
        self.assertIn(self.plugin._missing_hint(), self.event.sent[0])
        self.assertEqual(self.plugin.calls, [])

    async def test_self_and_explicit_account(self):
        await execute(self.context, self.event, self.cap, self.record, {}, self.digest)
        await execute(
            self.context,
            self.event,
            self.cap,
            self.record,
            {"target_user": "不存在", "account": "Explicit#9"},
            self.digest,
        )
        self.assertEqual(
            self.plugin.calls, [("111111", "Caller#1"), ("111111", "Explicit#9")]
        )

    async def test_permissions_and_session_scope(self):
        entry = self.record["commands"]["fixture.report"]
        entry["parents"] = [
            SimpleNamespace(
                enabled=True, event_filters=[PermissionTypeFilter(PermissionType.ADMIN)]
            )
        ]
        with self.assertRaises(CapabilityError):
            await execute(
                self.context, self.event, self.cap, self.record, {}, self.digest
            )
        self.assertEqual(self.event.sent, [])
        self.event.admin = True
        self.event.plugins_name = []
        with self.assertRaises(CapabilityError):
            await execute(
                self.context, self.event, self.cap, self.record, {}, self.digest
            )

    async def test_custom_filter_before_missing_binding(self):
        rule = SimpleNamespace(filter=lambda event, cfg: False)
        self.record["commands"]["fixture.report"]["handler"].event_filters.append(rule)
        with self.assertRaises(CapabilityError):
            await execute(
                self.context,
                self.event,
                self.cap,
                self.record,
                {"target_user": "未绑定"},
                self.digest,
            )
        self.assertEqual(self.event.sent, [])

    async def test_stale_source_and_invalid_parameters(self):
        with self.assertRaises(CapabilityError):
            await execute(
                self.context,
                self.event,
                self.cap,
                self.record,
                {"arbitrary": 1},
                self.digest,
            )
        (self.record["root"] / "main.py").write_text("version=222", encoding="utf-8")
        with self.assertRaises(CapabilityError):
            await execute(
                self.context, self.event, self.cap, self.record, {}, self.digest
            )

    async def test_ambiguous_and_nonmember_targets(self):
        self.event.bot.call_action.return_value.append(
            {"user_id": 444444, "nickname": "小二"}
        )
        with self.assertRaises(CapabilityError):
            await resolve_member(self.event, "小二")
        with self.assertRaises(CapabilityError):
            await resolve_member(self.event, "999999")

    async def test_runtime_deduplicates_a_request(self):
        bridge = object.__new__(CapabilityBridge)
        bridge.context = self.context
        bridge.config = {"enabled": True}
        bridge.dirty = False
        bridge.catalog = {"fixture": self.record}
        bridge.state = {
            "policies": {
                self.cap["id"]: {"mode": "direct", "fingerprint": self.digest}
            },
            "adaptations": {
                "fixture": {"fingerprint": self.digest, "capabilities": [self.cap]}
            },
        }
        a = await bridge.capability_run(self.event, self.cap["id"], {})
        b = await bridge.capability_run(self.event, self.cap["id"], {})
        self.assertEqual(a, b)
        self.assertEqual(len(self.plugin.calls), 1)

    def make_bridge(self, mode=None):
        bridge = object.__new__(CapabilityBridge)
        bridge.context = self.context
        bridge.config = {"enabled": True}
        bridge.dirty = False
        bridge.catalog = {"fixture": self.record}
        bridge.state = {
            "adaptations": {
                "fixture": {"fingerprint": self.digest, "capabilities": [self.cap]}
            },
            "policies": {},
        }
        if mode:
            bridge.state["policies"][self.cap["id"]] = {
                "mode": mode,
                "fingerprint": self.digest,
            }
        bridge.confirmations = PendingCalls()
        bridge.refresh = AsyncMock()
        return bridge

    def confirmation_event(self, bridge, code):
        event = FakeEvent()
        event.set_extra(
            "activated_handlers",
            [
                SimpleNamespace(
                    handler_name="confirm_execution",
                    handler=bridge.confirm_execution,
                    handler_full_name="fixture_confirm_execution",
                )
            ],
        )
        event.set_extra(
            "handlers_parsed_params",
            {"fixture_confirm_execution": {"confirmation_code": code}},
        )
        return event

    async def test_final_target_account_is_checked_before_execution(self):
        observed = []

        def restrict_target(event, config):
            account = event.get_extra("parsed_params", {}).get("account")
            observed.append((account, event.message_str))
            return account != "Target#2"

        self.record["commands"]["fixture.report"]["handler"].event_filters.append(
            SimpleNamespace(filter=restrict_target)
        )
        with self.assertRaises(CapabilityError):
            await execute(
                self.context,
                self.event,
                self.cap,
                self.record,
                {"target_user": "小二"},
                self.digest,
            )
        self.assertEqual(observed, [("Target#2", "account report Target#2")])
        self.assertEqual(self.plugin.calls, [])
        self.assertEqual(self.event.sent, [])

    async def test_permissions_rechecked_after_binding_lookup(self):
        self.event.admin = True
        self.record["commands"]["fixture.report"]["parents"] = [
            SimpleNamespace(
                enabled=True, event_filters=[PermissionTypeFilter(PermissionType.ADMIN)]
            )
        ]

        async def revoke_during_lookup(user_id):
            self.event.admin = False
            return "Target#2"

        self.plugin._get_account = revoke_during_lookup
        with self.assertRaises(CapabilityError):
            await execute(
                self.context,
                self.event,
                self.cap,
                self.record,
                {"target_user": "小二"},
                self.digest,
            )
        self.assertEqual(self.plugin.calls, [])

    async def test_parent_custom_filter_runs_once_with_final_parameters(self):
        observed = []

        def restrict_target(event, config):
            observed.append(event.get_extra("parsed_params")["account"])
            return False

        group = CommandGroupFilter("account")
        group.add_custom_filter(SimpleNamespace(filter=restrict_target))
        self.record["commands"]["fixture.report"]["parents"] = [
            SimpleNamespace(enabled=True, event_filters=[group])
        ]
        with self.assertRaises(CapabilityError):
            await execute(
                self.context,
                self.event,
                self.cap,
                self.record,
                {"target_user": "小二"},
                self.digest,
            )
        self.assertEqual(observed, ["Target#2"])
        self.assertEqual(self.plugin.calls, [])

    async def test_confirmation_accepts_framework_partial_and_command_parser(self):
        bridge = self.make_bridge()
        await bridge.capability_run(self.event, self.cap["id"], {})
        code = next(iter(bridge.confirmations.items))
        event = FakeEvent()
        event.is_at_or_wake_command = True
        event.message_str = "确认执行 " + code
        command_filter = CommandFilter("确认执行")
        command_filter.init_handler_md(
            SimpleNamespace(handler=CapabilityBridge.confirm_execution)
        )
        self.assertTrue(command_filter.filter(event, {}))
        parsed = event.get_extra("parsed_params")
        event.set_extra(
            "activated_handlers",
            [
                SimpleNamespace(
                    handler_name="confirm_execution",
                    handler=functools.partial(
                        CapabilityBridge.confirm_execution, bridge
                    ),
                    handler_full_name="fixture_confirm_execution",
                )
            ],
        )
        event.set_extra("handlers_parsed_params", {"fixture_confirm_execution": parsed})
        replies = [item async for item in bridge.confirm_execution(event, code)]
        self.assertEqual(replies, [])
        self.assertEqual(self.plugin.calls, [("111111", "Caller#1")])

    async def test_confirmation_does_not_inherit_event_stop_into_original_handler(self):
        bridge = self.make_bridge()
        observed = []

        async def report(event, account=None):
            observed.append(event._force_stopped)

        self.record["commands"]["fixture.report"]["handler"].handler = report
        await bridge.capability_run(self.event, self.cap["id"], {})
        code = next(iter(bridge.confirmations.items))
        event = self.confirmation_event(bridge, code)
        event._force_stopped = True
        replies = [item async for item in bridge.confirm_execution(event, code)]
        self.assertEqual(replies[0].get_plain_text(), "操作已执行")
        self.assertEqual(observed, [False])
        self.assertTrue(event._force_stopped)

    async def test_unreviewed_call_waits_for_actual_human_confirmation(self):
        bridge = self.make_bridge()
        result = json.loads(
            await bridge.capability_run(
                self.event, self.cap["id"], {"target_user": "小二"}
            )
        )
        self.assertEqual(result["status"], "confirmation_required")
        self.assertEqual(self.plugin.calls, [])
        self.assertIn("Target#2", self.event.sent[0])
        code = next(iter(bridge.confirmations.items))
        event = self.confirmation_event(bridge, code)
        replies = [item async for item in bridge.confirm_execution(event, code)]
        self.assertEqual(replies, [])
        self.assertEqual(self.plugin.calls, [("111111", "Target#2")])
        self.assertTrue(event.stopped)
        again = self.confirmation_event(bridge, code)
        replies = [item async for item in bridge.confirm_execution(again, code)]
        self.assertIn("确认码无效", replies[0].get_plain_text())
        self.assertEqual(len(self.plugin.calls), 1)

    async def test_confirmation_can_be_disabled_without_removing_admin_gate(self):
        bridge = self.make_bridge()
        bridge.config["confirmation_enabled"] = False
        result = json.loads(await bridge.capability_run(self.event, self.cap["id"], {}))
        self.assertEqual(result["status"], "completed")
        self.assertEqual(self.plugin.calls, [("111111", "Caller#1")])
        self.assertEqual(bridge.confirmations.items, {})
        bridge = self.make_bridge("admin_confirm")
        bridge.config["confirmation_enabled"] = False
        result = json.loads(
            await bridge.capability_run(FakeEvent(), self.cap["id"], {})
        )
        self.assertIn("仅限", result["error"])
        self.assertEqual(len(self.plugin.calls), 1)
        admin = FakeEvent()
        admin.admin = True
        result = json.loads(await bridge.capability_run(admin, self.cap["id"], {}))
        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(self.plugin.calls), 2)

    async def test_switching_confirmation_cancels_pending_and_old_message_retries(self):
        bridge = self.make_bridge()
        await bridge.capability_run(self.event, self.cap["id"], {})
        code = next(iter(bridge.confirmations.items))
        bridge.config["confirmation_enabled"] = False
        bridge.settings()
        self.assertEqual(bridge.confirmations.items, {})
        result = json.loads(await bridge.capability_run(self.event, self.cap["id"], {}))
        self.assertIn("旧请求不会自动执行", result["error"])
        event = self.confirmation_event(bridge, code)
        replies = [item async for item in bridge.confirm_execution(event, code)]
        self.assertIn("二次确认已关闭", replies[0].get_plain_text())
        self.assertEqual(self.plugin.calls, [])
        bridge.config["confirmation_enabled"] = True
        bridge.settings()
        result = json.loads(await bridge.capability_run(self.event, self.cap["id"], {}))
        self.assertIn("旧请求不会自动执行", result["error"])
        fresh = FakeEvent()
        result = json.loads(await bridge.capability_run(fresh, self.cap["id"], {}))
        self.assertEqual(result["status"], "confirmation_required")

    async def test_scope_blocks_tools_discovery_execution_and_confirmation(self):
        bridge = self.make_bridge("direct")
        bridge.config["group_blacklist"] = ["888888"]
        for result in (
            await bridge.capability_find(self.event, "账号报告"),
            await bridge.capability_run(self.event, self.cap["id"], {}),
        ):
            self.assertIn("允许范围", json.loads(result)["error"])
        original_tools = SimpleNamespace(
            tools=[
                SimpleNamespace(name="capability_find"),
                SimpleNamespace(name="capability_run"),
                SimpleNamespace(name="native_other"),
            ]
        )
        req = SimpleNamespace(func_tool=original_tools, system_prompt="original")
        await bridge.prepare_request(self.event, req)
        self.assertEqual([t.name for t in req.func_tool.tools], ["native_other"])
        self.assertEqual(len(original_tools.tools), 3)
        self.assertEqual(req.system_prompt, "original")
        bridge.config["group_blacklist"] = []
        bridge = self.make_bridge()
        await bridge.capability_run(FakeEvent(), self.cap["id"], {})
        code = next(iter(bridge.confirmations.items))
        bridge.config["group_blacklist"] = ["888888"]
        event = self.confirmation_event(bridge, code)
        replies = [item async for item in bridge.confirm_execution(event, code)]
        self.assertIn("允许范围", replies[0].get_plain_text())
        self.assertEqual(self.plugin.calls, [])

    async def test_scope_is_rechecked_after_async_binding_resolution(self):
        bridge = self.make_bridge("direct")

        async def block_during_lookup(user_id):
            bridge.config["group_blacklist"] = ["888888"]
            return "Caller#1"

        self.plugin._get_account = block_during_lookup
        result = json.loads(await bridge.capability_run(self.event, self.cap["id"], {}))
        self.assertIn("允许范围", result["error"])
        self.assertEqual(self.plugin.calls, [])

    async def test_execution_model_is_selected_before_build_and_only_in_allowed_scope(
        self,
    ):
        bridge = self.make_bridge()
        bridge.config["execution_provider_id"] = "configured-chat"
        self.event.set_extra("selected_model", "old-model")
        request = SimpleNamespace(model="old-model")
        self.event.set_extra("provider_request", request)
        await bridge.select_execution_model(self.event)
        self.assertEqual(self.event.get_extra("selected_provider"), "configured-chat")
        self.assertIsNone(self.event.get_extra("selected_model"))
        self.assertIsNone(request.model)
        blocked = FakeEvent()
        blocked.set_extra("selected_provider", "original")
        bridge.config["group_blacklist"] = ["888888"]
        await bridge.select_execution_model(blocked)
        self.assertEqual(blocked.get_extra("selected_provider"), "original")
        bridge = self.make_bridge()
        original = FakeEvent()
        original.set_extra("selected_provider", "original")
        await bridge.select_execution_model(original)
        self.assertEqual(original.get_extra("selected_provider"), "original")

    def test_legacy_model_migrates_once_and_clearing_it_survives_reload(self):
        class NativeConfig(dict):
            def save_config(self):
                pass

        data = Path(self.temp.name)
        target = data / "astrbot_plugin_capability_bridge" / "state.json"
        target.parent.mkdir()
        target.write_text(
            json.dumps(
                {"settings": {"provider_id": "legacy-analysis"}, "adaptations": {}}
            )
        )
        config = NativeConfig(read_settings({}))
        context = SimpleNamespace(register_web_api=Mock())
        with patch(
            "astrbot_plugin_capability_bridge.main.get_astrbot_plugin_data_path",
            return_value=str(data),
        ):
            first = CapabilityBridge(context, config)
            self.assertEqual(
                first.settings()["analysis_provider_id"], "legacy-analysis"
            )
            config["analysis_provider_id"] = ""
            second = CapabilityBridge(context, config)
            self.assertEqual(second.settings()["analysis_provider_id"], "")

    async def test_settings_api_writes_native_config_and_rejects_stale_or_invalid_updates(
        self,
    ):
        class NativeConfig(dict):
            def save_config(self):
                self.saved = dict(self)

        bridge = self.make_bridge()
        bridge.config = NativeConfig(read_settings({"group_blacklist": ["blocked"]}))
        bridge.context.get_all_providers = lambda: [
            SimpleNamespace(meta=lambda: SimpleNamespace(id="analysis-two"))
        ]
        revision = settings_revision(bridge.settings())
        payload = {"analysis_provider_id": "analysis-two", "revision": revision}
        incoming = SimpleNamespace(
            username="admin", json=AsyncMock(return_value=payload)
        )
        with (
            patch("astrbot_plugin_capability_bridge.main.request", incoming),
            patch(
                "astrbot_plugin_capability_bridge.main.json_response",
                side_effect=lambda value: value,
            ),
            patch(
                "astrbot_plugin_capability_bridge.main.error_response",
                side_effect=lambda message, **kw: {"error": message, **kw},
            ),
        ):
            result = await bridge.api_settings()
            self.assertTrue(result["saved"])
            self.assertEqual(
                bridge.config.saved["analysis_provider_id"], "analysis-two"
            )
            self.assertEqual(bridge.config.saved["group_blacklist"], ["blocked"])
            incoming.json = AsyncMock(
                return_value={"confirmation_enabled": False, "revision": revision}
            )
            result = await bridge.api_settings()
            self.assertEqual(result["status_code"], 409)
            self.assertTrue(bridge.config["confirmation_enabled"])
            incoming.json = AsyncMock(return_value={"confirmation_enabled": "false"})
            result = await bridge.api_settings()
            self.assertIn("必须是开关", result["error"])
            incoming.username = None
            result = await bridge.api_settings()
            self.assertEqual(result["status_code"], 401)

    async def test_model_or_executor_cannot_confirm_a_pending_call(self):
        bridge = self.make_bridge()
        await bridge.capability_run(self.event, self.cap["id"], {})
        code = next(iter(bridge.confirmations.items))
        forged = FakeEvent()
        forged.message_str = "确认执行 " + code
        replies = [item async for item in bridge.confirm_execution(forged, code)]
        self.assertIn("工具不能代为确认", replies[0].get_plain_text())
        self.assertIn(code, bridge.confirmations.items)
        self.assertEqual(self.plugin.calls, [])

    async def test_confirmation_rejects_changed_final_account(self):
        bridge = self.make_bridge()
        await bridge.capability_run(self.event, self.cap["id"], {"target_user": "小二"})
        code = next(iter(bridge.confirmations.items))
        self.plugin.bindings["222222"] = "Changed#3"
        event = self.confirmation_event(bridge, code)
        replies = [item async for item in bridge.confirm_execution(event, code)]
        self.assertIn("已经变化", replies[0].get_plain_text())
        self.assertEqual(self.plugin.calls, [])

    async def test_confirmation_does_not_silently_replace_message_context(self):
        bridge = self.make_bridge()
        self.event.message_obj.message = [At(qq="222222")]
        result = json.loads(await bridge.capability_run(self.event, self.cap["id"], {}))
        self.assertIn("消息对象", result["error"])
        self.assertEqual(bridge.confirmations.items, {})
        self.assertEqual(self.plugin.calls, [])
        self.event = FakeEvent()
        await bridge.capability_run(self.event, self.cap["id"], {})
        code = next(iter(bridge.confirmations.items))
        event = self.confirmation_event(bridge, code)
        event.message_obj.message = [At(qq="222222")]
        replies = [item async for item in bridge.confirm_execution(event, code)]
        self.assertIn("消息对象", replies[0].get_plain_text())
        self.assertEqual(self.plugin.calls, [])

    async def test_confirmation_rechecks_original_permissions(self):
        bridge = self.make_bridge()
        self.event.admin = True
        self.record["commands"]["fixture.report"]["parents"] = [
            SimpleNamespace(
                enabled=True, event_filters=[PermissionTypeFilter(PermissionType.ADMIN)]
            )
        ]
        await bridge.capability_run(self.event, self.cap["id"], {})
        code = next(iter(bridge.confirmations.items))
        event = self.confirmation_event(bridge, code)
        replies = [item async for item in bridge.confirm_execution(event, code)]
        self.assertIn("权限不足", replies[0].get_plain_text())
        self.assertEqual(self.plugin.calls, [])

    async def test_admin_policy_cannot_be_bypassed_by_known_capability_id(self):
        bridge = self.make_bridge("admin_confirm")
        result = json.loads(await bridge.capability_run(self.event, self.cap["id"], {}))
        self.assertIn("仅限", result["error"])
        self.assertEqual(bridge.confirmations.items, {})
        self.assertEqual(self.plugin.calls, [])

    async def test_direct_policy_keeps_original_admin_permission(self):
        bridge = self.make_bridge("direct")
        self.record["commands"]["fixture.report"]["parents"] = [
            SimpleNamespace(
                enabled=True, event_filters=[PermissionTypeFilter(PermissionType.ADMIN)]
            )
        ]
        result = json.loads(await bridge.capability_run(self.event, self.cap["id"], {}))
        self.assertIn("权限不足", result["error"])
        self.assertEqual(self.plugin.calls, [])

    async def test_changed_call_policy_invalidates_pending_confirmation(self):
        bridge = self.make_bridge()
        await bridge.capability_run(self.event, self.cap["id"], {})
        code = next(iter(bridge.confirmations.items))
        bridge.state["policies"][self.cap["id"]] = {
            "mode": "direct",
            "fingerprint": self.digest,
        }
        event = self.confirmation_event(bridge, code)
        replies = [item async for item in bridge.confirm_execution(event, code)]
        self.assertIn("已经变化", replies[0].get_plain_text())
        self.assertEqual(self.plugin.calls, [])

    def test_model_cannot_set_call_policy_in_adaptation(self):
        raw = copy.deepcopy(self.raw)
        raw["call_policy"] = "direct"
        with self.assertRaises(ValueError):
            validate_plan(raw, self.record)

    def test_reject_invented_mapping_and_mutating_helpers(self):
        self.assertNotIn("_unsafe_hint", self.record["helpers"])
        self.assertNotIn("_get_mutating", self.record["helpers"])
        for key, value in [
            ("lookup_method", "_get_mutating"),
            ("target_parameter", "invented"),
        ]:
            raw = copy.deepcopy(self.raw)
            raw["subject_binding"][key] = value
            with self.assertRaises(ValueError):
                validate_plan(raw, self.record)

    async def test_unloaded_instance_cannot_execute(self):
        self.context.get_registered_star = lambda name: None
        with self.assertRaises(CapabilityError):
            await execute(
                self.context, self.event, self.cap, self.record, {}, self.digest
            )
        self.assertEqual(self.plugin.calls, [])

    async def test_analysis_counts_usage_without_executing_handler(self):
        bridge = object.__new__(CapabilityBridge)
        bridge.context = self.context
        bridge.state_path = Path(self.temp.name) / "state.json"
        bridge.state = {"adaptations": {}}
        self.record.update(description="Fixture", native_tools=[])
        self.record["commands"]["fixture.report"].update(
            description="Report",
            source="async def report(event, account=None): ...",
            dependencies=[],
        )
        bridge.catalog = {"fixture": self.record}
        bridge.job = {
            "items": [{"plugin": "fixture"}],
            "tokens": 0,
            "estimated_tokens": 0,
        }
        self.context.llm_generate = AsyncMock(
            return_value=SimpleNamespace(
                completion_text=json.dumps({"capabilities": [self.raw]}),
                usage=SimpleNamespace(total=321),
            )
        )
        settings = {"provider_id": "fixture"}
        await bridge.analyze(["fixture"], settings)
        self.assertEqual(bridge.job["items"][0]["status"], "completed")
        self.assertEqual(bridge.job["tokens"], 321)
        self.assertEqual(bridge.job["estimated_tokens"], 0)
        self.assertEqual(len(bridge.state["adaptations"]["fixture"]["capabilities"]), 1)
        self.assertEqual(self.plugin.calls, [])
        self.context.llm_generate.reset_mock()
        bridge.state["adaptations"]["fixture"]["capabilities"][0]["enabled"] = False
        # Legacy user budgets no longer interrupt necessary analysis.
        settings["token_budget"] = 1
        await bridge.analyze(["fixture"], settings)
        self.context.llm_generate.assert_awaited_once()
        self.assertEqual(len(bridge.state["adaptations"]["fixture"]["capabilities"]), 1)
        self.assertFalse(
            bridge.state["adaptations"]["fixture"]["capabilities"][0]["enabled"]
        )

    def analysis_bridge(self, count=1):
        bridge = object.__new__(CapabilityBridge)
        bridge.context = self.context
        bridge.state_path = Path(self.temp.name) / "state.json"
        bridge.state = {"adaptations": {}}
        self.record.update(description="Fixture", native_tools=[])
        entry = self.record["commands"]["fixture.report"]
        entry.update(
            description="Report", source="async def report(event): ...", dependencies=[]
        )
        plans = []
        for index in range(count):
            key = f"fixture.report{index}" if index else "fixture.report"
            self.record["commands"][key] = {**entry, "id": key}
            plans.append({**copy.deepcopy(self.raw), "command_id": key})
        bridge.catalog = {"fixture": self.record}
        bridge.job = {
            "items": [{"plugin": "fixture"}],
            "tokens": 0,
            "estimated_tokens": 0,
        }
        return bridge, plans

    async def test_timed_out_batch_can_resume_without_paying_for_completed_batches(
        self,
    ):
        bridge, plans = self.analysis_bridge(4)
        self.context.llm_generate = AsyncMock(
            side_effect=[
                SimpleNamespace(
                    completion_text=json.dumps({"capabilities": plans[:3]}),
                    usage=SimpleNamespace(total=321),
                ),
                TimeoutError(),
            ]
        )
        await bridge.analyze(["fixture"], {"provider_id": "fixture"})
        self.assertEqual(bridge.job["items"][0]["status"], "failed")
        self.assertIn("第 2/2 批", bridge.job["items"][0]["error"])
        self.assertEqual(bridge.job["tokens"], 321)
        self.assertEqual(bridge.job["estimated_tokens"], 0)
        self.assertEqual(bridge.job["unknown_requests"], 1)
        self.assertEqual(bridge.job["pending_requests"], 0)
        saved = json.loads(bridge.state_path.read_text())
        self.assertEqual(len(saved["analysis_checkpoints"]["fixture"]["batches"]), 1)
        bridge.state = saved
        bridge.job = {
            "items": [{"plugin": "fixture"}],
            "tokens": 0,
            "estimated_tokens": 0,
        }
        self.context.llm_generate = AsyncMock(
            return_value=SimpleNamespace(
                completion_text=json.dumps({"capabilities": plans[3:]}),
                usage=SimpleNamespace(total=100),
            )
        )
        await bridge.analyze(["fixture"], {"provider_id": "fixture", "timeout": 900})
        self.context.llm_generate.assert_awaited_once()
        self.assertEqual(bridge.job["items"][0]["resumed_batches"], 1)
        self.assertEqual(bridge.job["tokens"], 100)
        self.assertEqual(len(bridge.state["adaptations"]["fixture"]["capabilities"]), 4)
        self.assertNotIn("fixture", bridge.state["analysis_checkpoints"])
        self.assertEqual(self.plugin.calls, [])

    async def test_changed_analysis_model_does_not_reuse_previous_batches(self):
        bridge, plans = self.analysis_bridge(4)
        first = SimpleNamespace(
            completion_text=json.dumps({"capabilities": plans[:3]}), usage=None
        )
        self.context.llm_generate = AsyncMock(side_effect=[first, TimeoutError()])
        await bridge.analyze(["fixture"], {"provider_id": "one"})
        self.context.llm_generate = AsyncMock(side_effect=[first, TimeoutError()])
        await bridge.analyze(["fixture"], {"provider_id": "two"})
        self.assertEqual(self.context.llm_generate.await_count, 2)
        self.assertEqual(bridge.job["items"][0]["resumed_batches"], 0)

    async def test_unknown_usage_is_estimated_only_after_response(self):
        bridge, plans = self.analysis_bridge()

        async def respond(**kwargs):
            self.assertEqual(bridge.job["pending_requests"], 1)
            self.assertEqual(bridge.job["estimated_tokens"], 0)
            return SimpleNamespace(
                completion_text=json.dumps({"capabilities": plans}), usage=None
            )

        self.context.llm_generate = AsyncMock(side_effect=respond)
        await bridge.analyze(["fixture"], {"provider_id": "fixture"})
        self.assertGreater(bridge.job["estimated_tokens"], 0)
        self.assertEqual(bridge.job["pending_requests"], 0)

    async def test_stream_final_response_is_not_concatenated_twice(self):
        item = {"received_chars": 0, "reasoning_chars": 0}
        final = SimpleNamespace(
            completion_text='{"capabilities":[]}',
            usage=SimpleNamespace(total=50),
            reasoning_content=None,
            is_chunk=False,
        )

        class StreamingProvider:
            async def text_chat_stream(self, **kwargs):
                yield SimpleNamespace(
                    completion_text="",
                    reasoning_content="private reasoning",
                    is_chunk=True,
                    usage=None,
                )
                yield SimpleNamespace(
                    completion_text='{"capabilities":',
                    reasoning_content=None,
                    is_chunk=True,
                    usage=None,
                )
                yield final

        self.context.get_provider_by_id = lambda _: StreamingProvider()
        response = await generate(self.context, "fixture", "document", item)
        self.assertIs(response, final)
        self.assertEqual(response.usage.total, 50)
        self.assertEqual(item["phase"], "receiving")
        self.assertNotIn("private reasoning", json.dumps(item))

    async def test_stream_timeout_closes_generator_and_does_not_retry(self):
        bridge, _ = self.analysis_bridge()
        closed = []

        class StreamingProvider:
            async def text_chat_stream(self, **kwargs):
                try:
                    yield SimpleNamespace(
                        completion_text="",
                        reasoning_content="thinking",
                        is_chunk=True,
                        usage=None,
                    )
                    await asyncio.sleep(1000)
                finally:
                    closed.append(True)

        self.context.get_provider_by_id = lambda _: StreamingProvider()
        self.context.llm_generate = AsyncMock()
        await bridge.analyze(["fixture"], {"provider_id": "fixture", "timeout": 0.01})
        self.assertEqual(closed, [True])
        self.assertEqual(bridge.job["items"][0]["status"], "failed")
        self.context.llm_generate.assert_not_called()
        self.assertEqual(bridge.job["estimated_tokens"], 0)

    def test_analysis_batching_keeps_all_commands_and_bounds_large_requests(self):
        _, _ = self.analysis_bridge(7)
        for command in self.record["commands"].values():
            command["source"] = "x" * 2600
            command["dependencies"] = ["x" * 1000, "x" * 1000]
        batches = analysis_batches(self.record)
        ids = [key for record, _ in batches for key in record["commands"]]
        self.assertEqual(ids, list(self.record["commands"]))
        self.assertTrue(all(len(record["commands"]) <= 3 for record, _ in batches))
        self.assertTrue(all(len(document) <= 12000 for _, document in batches))

    async def test_message_tail_is_scanned_and_passed_as_event_text(self):
        metadata = self.record["metadata"]
        metadata.reserved = False
        metadata.module = SimpleNamespace(__file__=str(self.record["root"] / "main.py"))
        metadata.module_path = type(self.plugin).__module__
        metadata.display_name = "Fixture"
        metadata.desc = "Text query"
        handler = SimpleNamespace(
            handler=self.plugin.text_query,
            handler_name="text_query",
            enabled=True,
            event_filters=[CommandFilter("find")],
        )
        self.context.get_all_stars = lambda: [metadata]
        self.context.get_llm_tool_manager = lambda: SimpleNamespace(func_list=[])
        descriptor = {
            "handler_full_name": "fixture.text_query",
            "module_path": metadata.module_path,
            "effective_command": "catalog find",
            "description": "Search",
            "enabled": True,
            "plugin_activated": True,
            "has_conflict": False,
        }
        with patch(
            "astrbot_plugin_capability_bridge.scanner.star_handlers_registry.star_handlers_map",
            {"fixture.text_query": handler},
        ):
            record = _collect(self.context, [descriptor])["fixture"]
        entry = record["commands"]["fixture.text_query"]
        self.assertEqual(entry["parameters"], {})
        self.assertTrue(entry["accepts_message_tail"])
        raw = {
            "name": "目录查询",
            "description": "查询目录内容",
            "examples": ["找一下条目"],
            "command_id": "fixture.text_query",
            "input_schema": {
                "type": "object",
                "properties": {"keyword": {"type": "string"}},
                "required": ["keyword"],
            },
            "argument_map": {},
            "message_tail": {"input": "keyword"},
        }
        cap = validate_plan(raw, record)
        result = await execute(
            self.context,
            self.event,
            cap,
            record,
            {"keyword": "Alpha Beta 礼装"},
            self.digest,
        )
        self.assertTrue(result["delivered"])
        self.assertEqual(
            self.plugin.calls,
            [("111111", "Alpha Beta 礼装", "catalog find Alpha Beta 礼装")],
        )
        self.assertEqual(self.event.message_str, "帮我查群友的报告")
        # Local variables must still never be passed as invented Python kwargs.
        raw["argument_map"] = {"keyword": {"input": "keyword"}}
        with self.assertRaises(ValueError):
            validate_plan(raw, record)

    async def test_validation_failure_has_one_reason_per_command(self):
        bridge = object.__new__(CapabilityBridge)
        bridge.context = self.context
        bridge.state_path = Path(self.temp.name) / "state.json"
        bridge.state = {"adaptations": {}}
        self.record.update(description="Fixture", native_tools=[])
        self.record["commands"]["fixture.report"].update(
            description="Report",
            source="async def report(event, account=None): ...",
            dependencies=[],
        )
        bridge.catalog = {"fixture": self.record}
        bridge.job = {
            "items": [{"plugin": "fixture"}],
            "tokens": 0,
            "estimated_tokens": 0,
        }
        raw = copy.deepcopy(self.raw)
        raw["argument_map"] = {"invented": {"value": "bad"}}
        self.context.llm_generate = AsyncMock(
            return_value=SimpleNamespace(
                completion_text=json.dumps({"capabilities": [raw]}),
                usage=SimpleNamespace(total=123),
            )
        )
        await bridge.analyze(["fixture"], {"provider_id": "fixture"})
        reasons = bridge.state["adaptations"]["fixture"]["unsupported"]
        self.assertEqual(len(reasons), 1)
        self.assertIn("不存在的参数", reasons[0]["reason"])

    def test_message_tail_requires_source_evidence(self):
        raw = copy.deepcopy(self.raw)
        raw["message_tail"] = {"input": "target_user"}
        with self.assertRaises(ValueError):
            validate_plan(raw, self.record)

    async def test_native_discovery_does_not_invoke_an_adapter(self):
        bridge = object.__new__(CapabilityBridge)
        bridge.config = {"enabled": True}
        bridge.context = self.context
        bridge.dirty = False
        bridge.catalog = {}
        bridge.state = {"adaptations": {}}
        req = SimpleNamespace(
            func_tool=SimpleNamespace(
                tools=[
                    SimpleNamespace(name="capability_find", active=True),
                    SimpleNamespace(
                        name="generate_image",
                        active=True,
                        description="按文字需求生成图片",
                        parameters={"type": "object"},
                    ),
                ]
            ),
            system_prompt="persona",
        )
        await bridge.prepare_request(self.event, req)
        result = json.loads(
            await bridge.capability_find(self.event, "帮我生成一张图片")
        )
        self.assertEqual(result["matches"][0]["execute_with"], "generate_image")
        self.assertIn("CAPABILITY_BRIDGE", req.system_prompt)


if __name__ == "__main__":
    unittest.main()
