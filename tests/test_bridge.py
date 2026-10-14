"""Behavioral checks using real AstrBot filters and in-memory plugin fixtures."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from astrbot.core.message.message_event_result import MessageChain, MessageEventResult
from astrbot.core.star.filter.command import CommandFilter
from astrbot.core.star.filter.permission import PermissionType, PermissionTypeFilter

from ..main import CapabilityBridge
from ..plans import validate_plan
from ..runtime import CapabilityError, execute, resolve_member
from ..scanner import _collect, fingerprint, inspect_helpers, parameter_schema


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
            "adaptations": {
                "fixture": {"fingerprint": self.digest, "capabilities": [self.cap]}
            }
        }
        a = await bridge.capability_run(self.event, self.cap["id"], {})
        b = await bridge.capability_run(self.event, self.cap["id"], {})
        self.assertEqual(a, b)
        self.assertEqual(len(self.plugin.calls), 1)

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
