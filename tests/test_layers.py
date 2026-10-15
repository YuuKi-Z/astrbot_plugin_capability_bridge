"""No paid models or platform sends: exercise the real two-layer boundaries."""

import copy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from astrbot.api.message_components import Image
from astrbot.core.agent.tool import FunctionTool, ToolSet
from astrbot.core.message.message_event_result import MessageChain, MessageEventResult
from astrbot.core.platform.astr_message_event import AstrMessageEvent
from astrbot.core.provider.entities import LLMResponse
from astrbot.core.provider.provider import Provider
from astrbot.core.star.context import Context

from ..layers import EXECUTION_INSTRUCTION, ExecutionLayer, OutputBuffer, user_contexts
from . import test_bridge as fixtures

FakeEvent = fixtures.FakeEvent


class LayerTests(unittest.IsolatedAsyncioTestCase):
    """Use the existing permission/binding fixture with real tool dispatch."""

    setUp = fixtures.BridgeTests.setUp
    make_bridge = fixtures.BridgeTests.make_bridge
    confirmation_event = fixtures.BridgeTests.confirmation_event

    def request(self, *extra):
        async def find(event, query):
            return await self.bridge.capability_find(event, query)

        async def run(event, capability_id, arguments):
            return await self.bridge.capability_run(event, capability_id, arguments)

        return SimpleNamespace(
            prompt="大肥鱼，帮我查一下报告",
            system_prompt="你是大肥鱼，爱抬杠，拒绝做事。",
            contexts=[
                {"role": "system", "content": "人设历史"},
                {"role": "user", "content": "查小二的报告"},
                {"role": "assistant", "content": "我才不查，笨蛋！"},
                {"role": "tool", "content": "旧工具结果"},
            ],
            image_urls=[],
            audio_urls=[],
            extra_user_content_parts=[],
            func_tool=ToolSet(
                [
                    FunctionTool(
                        name="capability_find",
                        description="查找能力",
                        parameters={
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                            "required": ["query"],
                        },
                        handler=find,
                    ),
                    FunctionTool(
                        name="capability_run",
                        description="执行能力",
                        parameters={
                            "type": "object",
                            "properties": {
                                "capability_id": {"type": "string"},
                                "arguments": {"type": "object"},
                            },
                            "required": ["capability_id", "arguments"],
                        },
                        handler=run,
                    ),
                    *extra,
                ]
            ),
        )

    async def asyncSetUp(self):
        for name in ("AstrAgentContext", "AgentContextWrapper"):
            mocked = patch(
                "astrbot_plugin_capability_bridge.layers." + name, SimpleNamespace
            )
            mocked.start()
            self.addCleanup(mocked.stop)
        self.bridge = self.make_bridge("direct")
        self.context.get_current_chat_provider_id = AsyncMock(
            return_value="persona-model"
        )
        self.context.tool_loop_agent = AsyncMock(return_value=self.response("chat"))
        self.context.llm_generate = AsyncMock(
            return_value=SimpleNamespace(completion_text="大肥鱼：办好了。")
        )

    @staticmethod
    def response(status, **fields):
        return SimpleNamespace(
            completion_text=json.dumps(
                {
                    "status": status,
                    "summary": "内部资料",
                    "question": "",
                    **fields,
                },
                ensure_ascii=False,
            )
        )

    async def test_isolation_and_tool_free_persona_preserve_original_user_history(self):
        request = self.request()
        original_history = copy.deepcopy(request.contexts)
        self.bridge.config["execution_provider_id"] = "execution-model"
        await self.bridge.prepare_request(self.event, request)
        kwargs = self.context.tool_loop_agent.call_args.kwargs
        self.assertEqual(kwargs["chat_provider_id"], "execution-model")
        self.assertEqual(kwargs["system_prompt"], EXECUTION_INSTRUCTION)
        self.assertNotIn("我才不查", str(kwargs["contexts"]))
        self.assertNotIn("人设历史", str(kwargs["contexts"]))
        self.assertEqual(request.contexts, original_history)
        self.assertEqual(request.prompt, "大肥鱼，帮我查一下报告")
        self.assertIn("你是大肥鱼", request.system_prompt)
        self.assertIsNone(request.func_tool)
        self.assertFalse(kwargs["stream"])
        self.assertEqual(self.event.sent, [])
        self.assertEqual(
            self.event.get_extra("capability_bridge_outcome")["status"], "chat"
        )

    async def test_adapted_execution_is_buffered_and_not_repeated_by_persona(self):
        async def model(**kwargs):
            run = kwargs["tools"].get_tool("capability_run")
            a = await run.handler(
                kwargs["event"], capability_id=self.cap["id"], arguments={}
            )
            b = await run.handler(
                kwargs["event"], capability_id=self.cap["id"], arguments={}
            )
            self.assertEqual(a, b)
            self.assertEqual(self.event.sent, [])
            return self.response("completed")

        self.context.tool_loop_agent.side_effect = model
        request = self.request()
        await self.bridge.prepare_request(self.event, request)
        self.assertEqual(len(self.plugin.calls), 1)
        self.assertIsNone(request.func_tool)
        outcome = self.event.get_extra("capability_bridge_outcome")
        self.assertIn("report: Caller#1", outcome["plugin_output"])
        self.assertEqual(outcome["status"], "completed")
        self.assertEqual(self.event.sent, [])

    async def test_native_yield_and_direct_send_preserve_media_until_presentation(self):
        image = Image.fromURL("https://example.invalid/result.png")

        async def native(event):
            yield MessageEventResult().message("查询值：42")
            await event.send(MessageChain(chain=[image]))

        tool = FunctionTool(
            name="native_query",
            description="查询",
            parameters={"type": "object"},
            handler=native,
        )

        async def model(**kwargs):
            result = (
                await kwargs["tools"].get_tool("native_query").handler(kwargs["event"])
            )
            self.assertTrue(json.loads(result)["returned"])
            return self.response("completed")

        self.context.tool_loop_agent.side_effect = model
        await self.bridge.prepare_request(self.event, self.request(tool))
        output = self.event.get_extra("capability_bridge_output")
        self.assertEqual(output.texts, ["查询值：42"])
        self.assertIs(output.media[0].chain[0], image)
        self.assertEqual(self.event.sent, [])
        await self.bridge.deliver_attachments(self.event)
        await self.bridge.deliver_attachments(self.event)
        self.assertEqual(len(self.event.sent), 1)

    async def test_approval_snapshot_survives_persona_and_only_human_can_execute(self):
        self.bridge = self.make_bridge()

        async def model(**kwargs):
            await (
                kwargs["tools"]
                .get_tool("capability_run")
                .handler(kwargs["event"], capability_id=self.cap["id"], arguments={})
            )
            return self.response("completed")  # Deliberately incorrect model status.

        self.context.tool_loop_agent.side_effect = model
        await self.bridge.prepare_request(self.event, self.request())
        self.assertEqual(self.plugin.calls, [])
        outcome = self.event.get_extra("capability_bridge_outcome")
        self.assertEqual(outcome["status"], "pending_confirmation")
        output = self.event.get_extra("capability_bridge_output")
        preview = output.protected[0].get_plain_text()
        code = next(iter(self.bridge.confirmations.items))
        self.assertIn("/确认执行 " + code, preview)
        self.assertIn("Caller#1", preview)
        self.assertEqual(self.event.sent, [])
        await self.bridge.deliver_attachments(self.event)
        self.assertEqual(self.event.sent, [preview])
        confirmation = self.confirmation_event(self.bridge, code)
        self.assertEqual(
            [r async for r in self.bridge.confirm_execution(confirmation, code)], []
        )
        self.assertEqual(len(self.plugin.calls), 1)
        self.assertEqual(confirmation.sent, ["大肥鱼：办好了。"])
        self.assertIsNone(self.context.llm_generate.call_args.kwargs["tools"])

    async def test_failure_is_cached_and_stops_further_side_effects(self):
        calls = []

        async def failure(event):
            calls.append("ran")
            raise RuntimeError("test failure")

        tool = FunctionTool(
            name="broken",
            description="失败",
            parameters={"type": "object"},
            handler=failure,
        )
        output = OutputBuffer()
        layer = ExecutionLayer(self.bridge, self.event, self.request(tool), output)
        a = await layer.call(tool, {})
        b = await layer.call(tool, {})
        self.assertEqual(a, b)
        self.assertEqual(calls, ["ran"])
        self.assertTrue(layer.terminal)
        self.assertIn("do_not_retry_automatically", a)

    async def test_fake_completion_without_tools_and_invalid_handoff_do_not_succeed(
        self,
    ):
        for response in (
            self.response("completed"),
            SimpleNamespace(completion_text="我才不办！"),
        ):
            self.context.tool_loop_agent.return_value = response
            event = FakeEvent()
            await self.bridge.prepare_request(event, self.request())
            self.assertEqual(
                event.get_extra("capability_bridge_outcome")["status"], "error"
            )
            self.assertEqual(event.sent, [])

    async def test_partial_success_preserves_question_and_followup_task(self):
        async def model(**kwargs):
            await (
                kwargs["tools"]
                .get_tool("capability_run")
                .handler(kwargs["event"], capability_id=self.cap["id"], arguments={})
            )
            return self.response(
                "needs_input",
                summary="报告已查到，第二项还缺城市",
                question="还要查询哪个城市？",
            )

        self.context.tool_loop_agent.side_effect = model
        await self.bridge.prepare_request(self.event, self.request())
        outcome = self.event.get_extra("capability_bridge_outcome")
        self.assertEqual(outcome["status"], "needs_input")
        self.assertEqual(outcome["question"], "还要查询哪个城市？")
        self.assertEqual(len(self.plugin.calls), 1)
        self.assertTrue(self.bridge.execution_turns)
        await self.bridge.guard_presentation(
            self.event, LLMResponse(role="err", completion_text="timeout")
        )
        self.assertIn("还需要补充信息：还要查询哪个城市？", self.event.sent)
        self.assertEqual(len(self.plugin.calls), 1)

    async def test_one_completed_step_does_not_finish_submitted_or_unavailable_work(
        self,
    ):
        for status in ("submitted", "unavailable"):

            async def model(**kwargs):
                await (
                    kwargs["tools"]
                    .get_tool("capability_run")
                    .handler(
                        kwargs["event"], capability_id=self.cap["id"], arguments={}
                    )
                )
                return self.response(status)

            self.context.tool_loop_agent.side_effect = model
            event = FakeEvent()
            await self.bridge.prepare_request(event, self.request())
            self.assertEqual(
                event.get_extra("capability_bridge_outcome")["status"], status
            )

    async def test_persona_failure_returns_original_outputs_once_without_reexecution(
        self,
    ):
        async def model(**kwargs):
            await (
                kwargs["tools"]
                .get_tool("capability_run")
                .handler(kwargs["event"], capability_id=self.cap["id"], arguments={})
            )
            return self.response("completed")

        self.context.tool_loop_agent.side_effect = model
        await self.bridge.prepare_request(self.event, self.request())
        failure = LLMResponse(role="err", completion_text="provider timeout")
        await self.bridge.guard_presentation(self.event, failure)
        await self.bridge.guard_presentation(self.event, failure)
        await self.bridge.deliver_attachments(self.event)
        self.assertEqual(len(self.plugin.calls), 1)
        self.assertEqual(
            self.event.sent[1:], ["report: Caller#1", "native image placeholder"]
        )
        self.assertIn("不会自动重做", self.event.sent[0])

    async def test_successful_persona_does_not_also_send_raw_text(self):
        await self.bridge.prepare_request(self.event, self.request())
        output = self.event.get_extra("capability_bridge_output")
        output.texts.append("不应重复发送的原始结果")
        await self.bridge.guard_presentation(
            self.event,
            LLMResponse(role="assistant", completion_text="人设已经说明结果"),
        )
        await self.bridge.deliver_attachments(self.event)
        self.assertEqual(self.event.sent, [])
        self.assertTrue(output.presentation_succeeded)

    async def test_discarded_reply_does_not_dump_results_or_repeat_tools(self):
        async def model(**kwargs):
            await (
                kwargs["tools"]
                .get_tool("capability_run")
                .handler(kwargs["event"], capability_id=self.cap["id"], arguments={})
            )
            return self.response("completed")

        self.context.tool_loop_agent.side_effect = model
        await self.bridge.prepare_request(self.event, self.request())
        output = self.event.get_extra("capability_bridge_output")
        output.outcome["calls"].append(
            {
                "tool": "memorize_long_term_memory",
                "results": [{"memorized": True, "id": 17}],
            }
        )
        response = LLMResponse(role="assistant", completion_text="Preference saved")
        await self.bridge.observe_presentation(self.event, response)
        response.completion_text = ""
        await self.bridge.guard_presentation(self.event, response)
        await self.bridge.guard_presentation(self.event, response)
        await self.bridge.deliver_attachments(self.event)
        self.assertEqual(self.event.sent, [])
        self.assertEqual(len(self.plugin.calls), 1)
        self.assertTrue(output.released)

    async def test_discarded_stream_does_not_trigger_failure_fallback(self):
        async def streaming(generator, *args, **kwargs):
            response = LLMResponse(role="assistant", completion_text="Preference saved")
            await self.bridge.observe_presentation(self.event, response)
            response.completion_text = ""
            await self.bridge.guard_presentation(self.event, response)

        self.event.send_streaming = streaming
        await self.bridge.prepare_request(self.event, self.request())
        output = self.event.get_extra("capability_bridge_output")
        output.texts.append("Internal tool output")
        await self.event.send_streaming(None)
        await self.bridge.deliver_attachments(self.event)
        self.assertEqual(self.event.sent, [])

    async def test_discarded_reply_preserves_protected_outputs_and_media_once(self):
        await self.bridge.prepare_request(self.event, self.request())
        output = self.event.get_extra("capability_bridge_output")
        output.texts.append("Internal tool output")
        output.protect("Pending confirmation")
        image = Image.fromURL("https://example.invalid/completed-result.png")
        output.media.append(MessageChain(chain=[image]))
        self.event.send = AsyncMock()
        response = LLMResponse(role="assistant", completion_text="Preference saved")
        await self.bridge.observe_presentation(self.event, response)
        response.completion_text = ""
        await self.bridge.guard_presentation(self.event, response)
        await output.release(self.event, fallback=True)
        await self.bridge.deliver_attachments(self.event)
        sent = [call.args[0] for call in self.event.send.await_args_list]
        self.assertEqual(len(sent), 2)
        self.assertEqual(sent[0].get_plain_text(), "Pending confirmation")
        self.assertIs(sent[1].chain[0], image)

    async def test_genuinely_empty_generation_still_preserves_tool_results(self):
        await self.bridge.prepare_request(self.event, self.request())
        output = self.event.get_extra("capability_bridge_output")
        output.texts.append("Completed tool result")
        response = LLMResponse(role="assistant", completion_text="")
        await self.bridge.observe_presentation(self.event, response)
        await self.bridge.guard_presentation(self.event, response)
        self.assertIn("Completed tool result", self.event.sent)
        self.assertFalse(output.presentation_succeeded)

    async def test_error_response_still_uses_fallback_after_successful_snapshot(self):
        await self.bridge.prepare_request(self.event, self.request())
        output = self.event.get_extra("capability_bridge_output")
        output.texts.append("Completed tool result")
        response = LLMResponse(role="assistant", completion_text="Generated reply")
        await self.bridge.observe_presentation(self.event, response)
        response.role = "err"
        response.completion_text = "Provider failed"
        await self.bridge.guard_presentation(self.event, response)
        self.assertIn("Completed tool result", self.event.sent)
        self.assertFalse(output.presentation_succeeded)

    async def test_generation_snapshot_is_reset_for_each_response(self):
        await self.bridge.prepare_request(self.event, self.request())
        output = self.event.get_extra("capability_bridge_output")
        output.texts.append("Completed tool result")
        await self.bridge.observe_presentation(
            self.event, LLMResponse(role="assistant", completion_text="Generated reply")
        )
        response = LLMResponse(role="assistant", completion_text="")
        await self.bridge.observe_presentation(self.event, response)
        await self.bridge.guard_presentation(self.event, response)
        self.assertIn("Completed tool result", self.event.sent)

    async def test_stream_interruption_before_final_guard_still_preserves_results(self):
        async def streaming(generator, *args, **kwargs):
            await self.bridge.observe_presentation(
                self.event,
                LLMResponse(role="assistant", completion_text="Generated reply"),
            )
            raise TimeoutError("Stream interrupted during response processing")

        self.event.send_streaming = streaming
        await self.bridge.prepare_request(self.event, self.request())
        output = self.event.get_extra("capability_bridge_output")
        output.texts.append("Completed tool result")
        with self.assertRaises(TimeoutError):
            await self.event.send_streaming(None)
        self.assertIn("Completed tool result", self.event.sent)

    async def test_empty_persona_reply_preserves_native_returned_data(self):
        await self.bridge.prepare_request(self.event, self.request())
        output = self.event.get_extra("capability_bridge_output")
        output.outcome = {
            "calls": [
                {"tool": "weather", "results": [{"temperature": 23, "unit": "C"}]}
            ]
        }
        await self.bridge.guard_presentation(
            self.event, LLMResponse(role="assistant", completion_text="")
        )
        self.assertEqual(
            json.loads(self.event.sent[1]), {"temperature": 23, "unit": "C"}
        )

    async def test_interrupted_stream_preserves_text_and_media_once(self):
        async def streaming(generator, *args, **kwargs):
            raise TimeoutError("persona stream interrupted")

        self.event.send_streaming = streaming
        await self.bridge.prepare_request(self.event, self.request())
        output = self.event.get_extra("capability_bridge_output")
        output.texts.append("真实查询结果：42")
        output.media.append(
            MessageChain(chain=[Image.fromURL("https://example.invalid/result.png")])
        )
        with self.assertRaises(TimeoutError):
            await self.event.send_streaming(None)
        self.assertEqual(self.event.sent[1], "真实查询结果：42")
        self.assertEqual(len(self.event.sent), 3)
        await self.bridge.guard_presentation(
            self.event, LLMResponse(role="err", completion_text="timeout")
        )
        self.assertEqual(len(self.event.sent), 3)

    async def test_failed_lookup_cannot_fall_back_to_persona_banter(self):
        async def model(**kwargs):
            await (
                kwargs["tools"]
                .get_tool("capability_find")
                .handler(kwargs["event"], query="不存在的功能")
            )
            return self.response("chat")

        self.context.tool_loop_agent.side_effect = model
        await self.bridge.prepare_request(self.event, self.request())
        self.assertEqual(
            self.event.get_extra("capability_bridge_outcome")["status"], "unavailable"
        )
        self.assertEqual(self.plugin.calls, [])

    async def test_followup_task_is_caller_bound_and_never_contains_persona(self):
        self.context.tool_loop_agent.return_value = self.response(
            "needs_input", question="查哪个账号？"
        )
        await self.bridge.prepare_request(self.event, self.request())
        await self.bridge.prepare_request(FakeEvent(), self.request())
        contexts = self.context.tool_loop_agent.call_args.kwargs["contexts"]
        self.assertIn("上一轮未完成", str(contexts))
        other = FakeEvent()
        other.get_sender_id = lambda: "222222"
        await self.bridge.prepare_request(other, self.request())
        self.assertNotIn(
            "上一轮未完成",
            str(self.context.tool_loop_agent.call_args.kwargs["contexts"]),
        )

    async def test_late_plugin_output_uses_persona_without_tools(self):
        await self.bridge.prepare_request(self.event, self.request())
        output = self.event.get_extra("capability_bridge_output")
        await self.bridge.deliver_attachments(self.event)
        await output.capture(MessageChain().message("后台任务已完成"))
        self.assertEqual(self.event.sent, ["大肥鱼：办好了。"])
        self.assertIsNone(self.context.llm_generate.call_args.kwargs["tools"])

    async def test_outputs_during_persona_generation_are_not_lost(self):
        await self.bridge.prepare_request(self.event, self.request())
        output = self.event.get_extra("capability_bridge_output")
        await output.capture(MessageChain().message("较晚到达的结果"))
        self.assertEqual(self.event.sent, [])
        await self.bridge.deliver_attachments(self.event)
        self.assertEqual(self.event.sent, ["大肥鱼：办好了。"])

    async def test_streaming_persona_releases_original_media_after_text(self):
        messages = []

        async def streaming(generator, *args, **kwargs):
            async for chain in generator:
                messages.append(chain.get_plain_text())

        self.event.send_streaming = streaming
        await self.bridge.prepare_request(self.event, self.request())
        output = self.event.get_extra("capability_bridge_output")
        output.protect("固定的确认数据")

        async def chunks():
            yield MessageChain().message("人设回复")

        await self.event.send_streaming(chunks())
        self.assertEqual(messages, ["人设回复"])
        self.assertEqual(self.event.sent, ["固定的确认数据"])
        self.assertTrue(output.released)
        await self.bridge.deliver_attachments(self.event)
        self.assertEqual(len(self.event.sent), 1)

    def test_history_discards_system_assistant_and_tool_roles(self):
        contexts = self.request().contexts
        self.assertEqual(
            user_contexts(contexts), [{"role": "user", "content": "查小二的报告"}]
        )


class IntegrationEvent(FakeEvent, AstrMessageEvent):
    """An AstrBot event type whose sends stay in memory."""


class ScriptedProvider(Provider):
    """Drive the real AstrBot runner without contacting a model service."""

    def __init__(self, responses):
        super().__init__({"id": "scripted", "type": "test"}, {})
        self.responses = iter(responses)
        self.requests = []

    def get_current_key(self):
        return ""

    def set_key(self, key):
        pass

    async def get_models(self):
        return ["test"]

    async def text_chat(self, **kwargs):
        self.requests.append(kwargs)
        return next(self.responses)


class RunnerIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_runner_and_native_executor_have_no_platform_output(self):
        from astrbot.core.tools.message_tools import SendMessageToUserTool

        from ..main import CapabilityBridge

        event = IntegrationEvent()
        context = object.__new__(Context)
        provider = ScriptedProvider(
            [
                LLMResponse(
                    role="assistant",
                    tools_call_name=["native_query", "send_message_to_user"],
                    tools_call_args=[
                        {},
                        {"messages": [{"type": "plain", "text": "通过会话发送的结果"}]},
                    ],
                    tools_call_ids=["query-1", "send-1"],
                ),
                LLMResponse(
                    role="assistant",
                    completion_text='{"status":"completed","summary":"查询完成","question":""}',
                ),
            ]
        )
        context.provider_manager = SimpleNamespace(
            get_provider_by_id=AsyncMock(return_value=provider)
        )
        context.send_message = AsyncMock(return_value=True)
        bridge = object.__new__(CapabilityBridge)
        bridge.config = {"enabled": True}
        bridge.context = context

        async def native(incoming):
            await incoming.send(MessageChain().message("真实执行值：42"))
            return "查询成功，值为42"

        request = SimpleNamespace(
            prompt="查一下数值",
            contexts=[],
            func_tool=ToolSet(
                [
                    FunctionTool(
                        name="native_query",
                        description="查数值",
                        parameters={"type": "object"},
                        handler=native,
                    ),
                    SendMessageToUserTool(),
                ]
            ),
        )
        output = OutputBuffer()
        layer = ExecutionLayer(bridge, event, request, output)
        outcome = await layer.run("scripted")
        self.assertEqual(outcome["status"], "completed")
        self.assertEqual(
            outcome["plugin_output"], ["真实执行值：42", "通过会话发送的结果"]
        )
        self.assertEqual(event.sent, [])
        context.send_message.assert_not_awaited()
        self.assertEqual(len(provider.requests), 2)
        self.assertEqual(len(layer.calls), 2)


if __name__ == "__main__":
    unittest.main()
