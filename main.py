"""Discover plugin interfaces, generate cached mappings, and expose two tools."""

import asyncio
import copy
import functools
import json
import re
import time
from pathlib import Path

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import At, Plain
from astrbot.api.star import Context, Star
from astrbot.api.web import error_response, json_response, request
from astrbot.core.message.message_event_result import MessageChain
from astrbot.core.utils.astrbot_path import get_astrbot_plugin_data_path
from jsonschema.exceptions import SchemaError

from .analysis import checkpoint_key, estimated_tokens, generate
from .layers import (
    PRESENTATION_INSTRUCTION,
    ExecutionLayer,
    OutputBuffer,
    attach_presentation,
)
from .plans import parse_response, validate_plan
from .policy import POLICY_MODES, PendingCalls, call_policy
from .prompts import ANALYZE_PROMPT
from .runtime import CapabilityError, check_access, execute
from .scanner import PLUGIN_NAME, analysis_batches, scan
from .settings import access_error, read_settings, settings_revision, validate_update


class CapabilityBridge(Star):
    """A plugin-neutral capability catalog with explicit analysis jobs."""

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self.state_path = (
            Path(get_astrbot_plugin_data_path()) / PLUGIN_NAME / "state.json"
        )
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state = {
            "settings": {"provider_id": ""},
            "adaptations": {},
            "policies": {},
            "analysis_checkpoints": {},
        }
        if self.state_path.exists():
            try:
                loaded = json.loads(self.state_path.read_text(encoding="utf-8"))
                self.state["settings"]["provider_id"] = loaded.get("settings", {}).get(
                    "provider_id", ""
                )
                self.state["adaptations"] = loaded.get("adaptations", {})
                if isinstance(loaded.get("policies"), dict):
                    self.state["policies"] = loaded["policies"]
                if isinstance(loaded.get("analysis_checkpoints"), dict):
                    self.state["analysis_checkpoints"] = loaded["analysis_checkpoints"]
                self.state["config_settings_migrated"] = bool(
                    loaded.get("config_settings_migrated", False)
                )
            except (OSError, ValueError):
                logger.warning("Capability bridge state could not be loaded")
        self.catalog = {}
        self.dirty = True
        self.scan_lock = asyncio.Lock()
        self.job_task = None
        self.confirmations = PendingCalls()
        self.execution_turns = {}
        self.closed = False
        self._controls_generation = 0
        self._controls_revision = settings_revision(
            read_settings(self.config), controls_only=True
        )
        if not self.state.get("config_settings_migrated"):
            legacy_provider = self.state["settings"].get("provider_id", "")
            if legacy_provider and not self.config.get("analysis_provider_id"):
                self.config["analysis_provider_id"] = legacy_provider
                self.config.save_config()
            self.state["config_settings_migrated"] = True
            self.save()
        self.job = {"status": "idle", "items": [], "tokens": 0, "estimated_tokens": 0}
        for path, handler, methods in [
            ("status", self.api_status, ["GET"]),
            ("scan", self.api_scan, ["POST"]),
            ("settings", self.api_settings, ["POST"]),
            ("adapt", self.api_adapt, ["POST"]),
            ("cancel", self.api_cancel, ["POST"]),
            ("toggle", self.api_toggle, ["POST"]),
            ("policy", self.api_policy, ["POST"]),
        ]:
            context.register_web_api(
                f"/{PLUGIN_NAME}/{path}", handler, methods, "Capability bridge " + path
            )

    def save(self):
        """Persist settings and validated mappings atomically."""
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(self.state, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(self.state_path)

    def settings(self) -> dict:
        """Read the native plugin configuration, shared by both settings UIs."""
        values = read_settings(self.config)
        revision = settings_revision(values, controls_only=True)
        if getattr(self, "_controls_revision", revision) != revision:
            self.confirmations.items.clear()
            self._controls_generation = getattr(self, "_controls_generation", 0) + 1
        self._controls_revision = revision
        return values

    def ensure_access(self, event) -> dict:
        """Check current scope again at execution time, including confirmations."""
        if getattr(self, "closed", False):
            raise CapabilityError("插件已重载，请重新发起需求")
        try:
            values = self.settings()
        except ValueError as exc:
            raise CapabilityError("插件设置无效，请管理员检查配置") from exc
        error = access_error(values, event, PLUGIN_NAME)
        if error:
            raise CapabilityError(error)
        token = self.controls_token(values)
        previous = event.get_extra("capability_bridge_controls")
        if previous is not None and previous != token:
            raise CapabilityError("运行设置已变化，请重新发起需求；旧请求不会自动执行")
        event.set_extra("capability_bridge_controls", token)
        return values

    def controls_token(self, values) -> str:
        """Include a generation so toggling off and back on cannot revive old calls."""
        return (
            settings_revision(values, controls_only=True)
            + ":"
            + str(getattr(self, "_controls_generation", 0))
        )

    async def refresh(self):
        """Refresh free interface metadata, serializing concurrent requests."""
        async with self.scan_lock:
            self.catalog = await scan(self.context)
            self.dirty = False

    @filter.on_astrbot_loaded()
    async def loaded(self):
        """Scan only after all plugins have registered their interfaces."""
        await self.refresh()

    @filter.on_plugin_loaded()
    async def plugin_loaded(self, metadata):
        """Mark the catalog dirty without starting a paid model call."""
        self.dirty = True

    @filter.on_plugin_unloaded()
    async def plugin_unloaded(self, metadata):
        """Invalidate live instance references immediately after an unload."""
        self.dirty = True

    def status(self) -> dict:
        """Return a serializable dashboard snapshot without plugin secrets."""
        plugins = []
        for name, record in self.catalog.items():
            adaptation = self.state["adaptations"].get(name, {})
            caps = adaptation.get("capabilities", [])
            stale = bool(
                adaptation and adaptation.get("fingerprint") != record["fingerprint"]
            )
            state = (
                "disabled"
                if not record["active"]
                else "stale"
                if stale
                else "ready"
                if caps
                else "native"
                if record["native_tools"] and not record["commands"]
                else "unsupported"
                if adaptation
                else "passive"
                if not record["commands"]
                else "new"
            )
            plugins.append(
                {
                    "name": name,
                    "display_name": record["display_name"],
                    "version": record["version"],
                    "fingerprint": record["fingerprint"],
                    "active": record["active"],
                    "state": state,
                    "command_count": len(record["commands"]),
                    "native_tools": record["native_tools"],
                    "capabilities": [
                        {
                            **cap,
                            "call_policy": call_policy(
                                self.state, cap["id"], record["fingerprint"]
                            )[0],
                            "policy_reviewed": call_policy(
                                self.state, cap["id"], record["fingerprint"]
                            )[1],
                        }
                        for cap in caps
                    ],
                    "unsupported": list(
                        {
                            u["command_id"]: u
                            for u in adaptation.get("unsupported", [])
                        }.values()
                    ),
                    "error": adaptation.get("error", ""),
                    "adapted_at": adaptation.get("adapted_at"),
                    "tokens": adaptation.get("tokens", 0),
                }
            )
        providers = []
        for provider in self.context.get_all_providers():
            meta = provider.meta()
            providers.append({"id": meta.id, "model": provider.get_model()})
        return {
            "enabled": bool(self.config.get("enabled", True)),
            "settings": self.settings(),
            "settings_revision": settings_revision(self.settings()),
            "providers": providers,
            "plugins": plugins,
            "job": {
                **self.job,
                "items": [
                    {
                        **item,
                        "elapsed_seconds": max(
                            0,
                            int(time.time()) - item["request_started_at"],
                        )
                        if item.get("request_started_at")
                        and item.get("status") == "analyzing"
                        else item.get("elapsed_seconds", 0),
                    }
                    for item in self.job["items"]
                ],
            },
        }

    async def api_status(self):
        """Get the authenticated dashboard status."""
        if not request.username:
            return error_response("需要登录管理后台", status_code=401)
        if self.dirty:
            await self.refresh()
        return json_response(self.status())

    async def api_scan(self):
        """Rescan registrations and fingerprints without model usage."""
        if not request.username:
            return error_response("需要登录管理后台", status_code=401)
        await self.refresh()
        return json_response(self.status())

    async def api_settings(self):
        """Save validated partial updates to the native plugin configuration."""
        if not request.username:
            return error_response("需要登录管理后台", status_code=401)
        data = await request.json(default={})
        if not isinstance(data, dict):
            return error_response("设置必须为对象")
        data = dict(data)
        revision = data.pop("revision", None)
        current = self.settings()
        if revision is not None and revision != settings_revision(current):
            return error_response(
                "设置已在其他页面更改，请重新载入后再保存", status_code=409
            )
        # Accept the previous analysis-page field during rolling upgrades.
        if "provider_id" in data:
            if "analysis_provider_id" in data:
                return error_response("分析模型字段重复")
            data["analysis_provider_id"] = data.pop("provider_id")
        try:
            values = validate_update(
                current, data, {p.meta().id for p in self.context.get_all_providers()}
            )
        except ValueError as exc:
            return error_response(str(exc))
        original = dict(self.config)
        self.config.update(values)
        try:
            self.config.save_config()
        except Exception:
            self.config.clear()
            self.config.update(original)
            raise
        self.settings()  # Invalidate pending approvals when runtime controls change.
        return json_response({"saved": True, "revision": settings_revision(values)})

    async def api_adapt(self):
        """Start one background job for selected or changed plugins."""
        if not request.username:
            return error_response("需要登录管理后台", status_code=401)
        if self.job_task and not self.job_task.done():
            return error_response("已有适配任务运行中", status_code=409)
        await self.refresh()
        data = await request.json(default={})
        selected = data.get("plugins") if isinstance(data, dict) else None
        if selected is not None and (
            not isinstance(selected, list)
            or any(n not in self.catalog for n in selected)
        ):
            return error_response("插件选择无效")
        values = self.settings()
        settings = {
            "provider_id": values["analysis_provider_id"],
            "timeout": values["analysis_timeout"],
        }
        if settings["provider_id"] not in {
            p.meta().id for p in self.context.get_all_providers()
        }:
            return error_response("请先选择并保存分析模型")
        names = (
            selected
            if selected is not None
            else [
                name
                for name, r in self.catalog.items()
                if self.state["adaptations"].get(name, {}).get("fingerprint")
                != r["fingerprint"]
            ]
        )
        names = list(
            dict.fromkeys(
                n
                for n in names
                if self.catalog[n]["active"] and self.catalog[n]["commands"]
            )
        )
        self.job = {
            "status": "running",
            "started_at": int(time.time()),
            "items": [{"plugin": n, "status": "queued"} for n in names],
            "tokens": 0,
            "estimated_tokens": 0,
            "pending_requests": 0,
            "unknown_requests": 0,
        }
        self.job_task = asyncio.create_task(self.analyze(names, settings))
        return json_response({"started": True, "plugins": names})

    async def analyze(self, names: list, settings: dict):
        """Analyze source evidence and activate only validated declarations.

        Args:
            names: Plugin IDs selected by the dashboard user.
            settings: Immutable settings for this job.
        """
        try:
            for index, name in enumerate(names):
                item = self.job["items"][index]
                record = self.catalog.get(name)
                if not record or not record["active"]:
                    item.update(status="cancelled", error="插件已卸载或停用")
                    continue
                item["status"] = "analyzing"
                tokens = 0
                try:
                    parsed = {"capabilities": [], "unsupported": []}
                    batches = analysis_batches(record)
                    checkpoints = self.state.setdefault("analysis_checkpoints", {})
                    checkpoint = checkpoints.get(name, {})
                    if (
                        checkpoint.get("fingerprint") != record["fingerprint"]
                        or checkpoint.get("provider_id") != settings["provider_id"]
                    ):
                        checkpoint = {
                            "fingerprint": record["fingerprint"],
                            "provider_id": settings["provider_id"],
                            "batches": {},
                        }
                        checkpoints[name] = checkpoint
                    item.update(completed_batches=0, resumed_batches=0)
                    for batch_index, (batch_record, document) in enumerate(batches):
                        item["batch"] = f"{batch_index + 1}/{len(batches)}"
                        key = checkpoint_key(document)
                        cached = checkpoint["batches"].get(key)
                        if cached is not None:
                            part = cached
                            item["resumed_batches"] += 1
                        else:
                            timeout = settings.get("timeout", 600)
                            item.update(
                                phase="waiting",
                                request_started_at=int(time.time()),
                                timeout_seconds=timeout,
                                received_chars=0,
                                reasoning_chars=0,
                                last_response_at=None,
                                command_count=len(batch_record["commands"]),
                            )
                            self.job["pending_requests"] = 1
                            try:
                                response = await asyncio.wait_for(
                                    generate(
                                        self.context,
                                        settings["provider_id"],
                                        document,
                                        item,
                                    ),
                                    timeout=timeout,
                                )
                            except BaseException:
                                self.job["unknown_requests"] = (
                                    self.job.get("unknown_requests", 0) + 1
                                )
                                raise
                            finally:
                                self.job["pending_requests"] = 0
                                item["elapsed_seconds"] = max(
                                    0, int(time.time()) - item["request_started_at"]
                                )
                            usage = getattr(response, "usage", None)
                            batch_tokens = int(usage.total) if usage else 0
                            tokens += batch_tokens
                            if batch_tokens:
                                self.job["tokens"] += batch_tokens
                            else:
                                self.job["estimated_tokens"] += estimated_tokens(
                                    document + ANALYZE_PROMPT + response.completion_text
                                )
                            item["phase"] = "validating"
                            part = parse_response(response.completion_text)
                            # Cache only declarations that pass the same source checks.
                            # Invalid responses remain retryable on a later user request.
                            try:
                                for raw in part["capabilities"]:
                                    validate_plan(raw, batch_record)
                            except (ValueError, TypeError, KeyError, SchemaError):
                                pass
                            else:
                                checkpoint["batches"][key] = part
                                self.save()
                        parsed["capabilities"].extend(part["capabilities"])
                        parsed["unsupported"].extend(part.get("unsupported") or [])
                        item["completed_batches"] += 1
                    capabilities, rejected, seen = [], [], set()
                    for raw in parsed["capabilities"][:500]:
                        try:
                            cap = validate_plan(raw, record)
                            if cap["command_id"] in seen:
                                raise ValueError("同一指令重复定义")
                            seen.add(cap["command_id"])
                            capabilities.append(cap)
                        except (ValueError, TypeError, KeyError, SchemaError) as exc:
                            rejected.append(
                                {
                                    "command_id": str(raw.get("command_id", ""))[:300]
                                    if isinstance(raw, dict)
                                    else "",
                                    "reason": str(exc)[:300],
                                }
                            )
                    unsupported = {
                        e["id"]: {
                            "command_id": e["id"],
                            "reason": "模型未生成可校验的适配；可能由原生工具覆盖或证据不足",
                        }
                        for e in record["commands"].values()
                        if e["enabled"] and e["id"] not in seen
                    }
                    for reported in (parsed.get("unsupported") or [])[:500]:
                        if (
                            isinstance(reported, dict)
                            and reported.get("command_id") in unsupported
                        ):
                            unsupported[reported["command_id"]]["reason"] = str(
                                reported.get("reason", "")
                            )[:500]
                    for rejected_plan in rejected:
                        if rejected_plan["command_id"] in unsupported:
                            unsupported[rejected_plan["command_id"]] = rejected_plan
                    unsupported = list(unsupported.values())
                    previous = self.state["adaptations"].get(name, {})
                    previous_switches = {
                        c["id"]: c.get("enabled", True)
                        for c in previous.get("capabilities", [])
                    }
                    for cap in capabilities:
                        cap["enabled"] = previous_switches.get(cap["id"], True)
                    if (
                        not capabilities
                        and previous.get("capabilities")
                        and previous.get("fingerprint") == record["fingerprint"]
                    ):
                        checkpoints.pop(name, None)
                        raise ValueError("本次未产生有效适配，保留已有规则")
                    self.state["adaptations"][name] = {
                        "fingerprint": record["fingerprint"],
                        "capabilities": capabilities,
                        "unsupported": unsupported,
                        "adapted_at": int(time.time()),
                        "tokens": tokens,
                        "provider_id": settings["provider_id"],
                    }
                    checkpoints.pop(name, None)
                    self.save()
                    item.update(
                        status="completed",
                        capabilities=len(capabilities),
                        rejected=len(rejected),
                        tokens=tokens,
                    )
                except asyncio.CancelledError:
                    item["status"] = "cancelled"
                    raise
                except (TimeoutError, asyncio.TimeoutError):
                    item.update(
                        status="failed",
                        error=(
                            f"第 {item.get('batch', '?')} 批模型请求超时"
                            f"（已等待 {item.get('elapsed_seconds', 0)} 秒）。"
                            "已完成批次已保存，再次适配会接着处理。"
                            "可检查模型服务、调大单批超时或切换分析模型。"
                            "本次请求未返回实际用量，无法确认计费。"
                        ),
                        tokens=tokens,
                    )
                except Exception as exc:  # noqa: BLE001 - Isolate provider and plugin failures.
                    # Provider exceptions can contain URLs and credentials; log only type.
                    logger.warning(
                        "Capability adaptation failed for %s (%s)",
                        name,
                        type(exc).__name__,
                    )
                    item.update(
                        status="failed",
                        error=str(exc)[:300]
                        if isinstance(
                            exc, (ValueError, TypeError, json.JSONDecodeError)
                        )
                        else f"分析失败：{type(exc).__name__}，请检查模型连接或稍后重试",
                        tokens=tokens,
                    )
            self.job["status"] = "completed"
        except asyncio.CancelledError:
            self.job["status"] = "cancelled"
        finally:
            for item in self.job["items"]:
                if item["status"] == "queued":
                    item["status"] = "cancelled"
            self.job["finished_at"] = int(time.time())

    async def api_cancel(self):
        """Cancel remaining analysis work without discarding saved mappings."""
        if not request.username:
            return error_response("需要登录管理后台", status_code=401)
        if self.job_task and not self.job_task.done():
            self.job_task.cancel()
        return json_response({"cancel_requested": True})

    async def api_toggle(self):
        """Enable or disable one validated capability."""
        if not request.username:
            return error_response("需要登录管理后台", status_code=401)
        data = await request.json(default={})
        if not isinstance(data, dict) or type(data.get("enabled")) is not bool:
            return error_response("开关参数无效")
        for adaptation in self.state["adaptations"].values():
            for cap in adaptation.get("capabilities", []):
                if cap["id"] == data.get("id"):
                    cap["enabled"] = data["enabled"]
                    self.save()
                    return json_response({"saved": True})
        return error_response("能力不存在", status_code=404)

    async def api_policy(self):
        """Save an administrator's call policy for the exact scanned source."""
        if not request.username:
            return error_response("需要登录管理后台", status_code=401)
        data = await request.json(default={})
        if (
            not isinstance(data, dict)
            or not isinstance(data.get("mode"), str)
            or data["mode"] not in POLICY_MODES
        ):
            return error_response("调用方式无效")
        await self.refresh()
        for name, adaptation in self.state["adaptations"].items():
            record = self.catalog.get(name)
            if not record:
                continue
            for cap in adaptation.get("capabilities", []):
                if cap["id"] != data.get("id"):
                    continue
                if (
                    data.get("fingerprint") != record["fingerprint"]
                    or adaptation.get("fingerprint") != record["fingerprint"]
                ):
                    return error_response(
                        "插件已变化，请刷新并重新适配后再设置", status_code=409
                    )
                self.state.setdefault("policies", {})[cap["id"]] = {
                    "mode": data["mode"],
                    "fingerprint": record["fingerprint"],
                }
                self.save()
                return json_response({"saved": True})
        return error_response("能力不存在", status_code=404)

    @filter.on_waiting_llm_request(priority=-20000)
    async def select_execution_model(self, event: AstrMessageEvent):
        """Select only the private execution model; preserve the persona model."""
        try:
            settings = self.ensure_access(event)
        except CapabilityError:
            return
        if provider_id := settings["execution_provider_id"]:
            event.set_extra("capability_bridge_execution_provider", provider_id)

    @filter.on_llm_request(priority=-20000)
    async def prepare_request(self, event: AstrMessageEvent, req):
        """Complete private execution before the normal persona request starts."""
        available = getattr(req.func_tool, "tools", []) if req.func_tool else []
        try:
            self.ensure_access(event)
        except CapabilityError:
            event.set_extra("capability_bridge_native", [])
            if req.func_tool:
                req.func_tool = copy.copy(req.func_tool)
                req.func_tool.tools = [
                    t
                    for t in available
                    if t.name not in {"capability_find", "capability_run"}
                ]
            return
        event.set_extra(
            "capability_bridge_native",
            [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.parameters,
                }
                for t in available
                if t.active and t.name not in {"capability_find", "capability_run"}
            ],
        )
        if not any(t.name == "capability_find" and t.active for t in available):
            return
        previous_outcome = event.get_extra("capability_bridge_outcome")
        if previous_outcome is not None:
            attach_presentation(req, previous_outcome)
            return
        output = OutputBuffer()
        event.set_extra("capability_bridge_output", output)
        layer = ExecutionLayer(self, event, req, output)
        presentation = {
            "system_prompt": req.system_prompt or "",
            "prompt": req.prompt,
            "provider_id": event.get_extra("selected_provider") or "",
        }
        # Snapshot only presentation context, never provider credentials.
        event.set_extra("capability_bridge_presentation", presentation)
        layer.event.set_extra("capability_bridge_presentation", presentation)
        try:
            if not presentation["provider_id"]:
                presentation[
                    "provider_id"
                ] = await self.context.get_current_chat_provider_id(
                    event.unified_msg_origin
                )
            provider_id = (
                self.settings()["execution_provider_id"] or presentation["provider_id"]
            )
            turns = getattr(self, "execution_turns", {})
            now = time.monotonic()
            turns = {k: v for k, v in turns.items() if now - v["time"] < 600}
            owner = (event.unified_msg_origin, str(event.get_sender_id()))
            previous = turns.get(owner)
            if previous and previous["controls"] != self.controls_token(
                self.settings()
            ):
                previous = None
            outcome = await layer.run(provider_id, previous and previous["task"])
            self.ensure_access(layer.event)
            if outcome["status"] == "needs_input":
                turns[owner] = {
                    "time": now,
                    "controls": self.controls_token(self.settings()),
                    "task": {"request": req.prompt, "outcome": outcome},
                }
            else:
                turns.pop(owner, None)
            while len(turns) > 128:
                turns.pop(next(iter(turns)))
            self.execution_turns = turns
        except Exception as exc:  # noqa: BLE001 - Hand off failures without rerunning tools.
            logger.warning("Private execution failed (%s)", type(exc).__name__)
            outcome = {
                "status": "error",
                "summary": "指令判断或执行未正常完成，不能确认任务成功；不会自动重试",
                "calls": layer.calls,
                "plugin_output": output.texts,
                "protected_output": bool(output.protected or output.media),
            }
        event.set_extra("capability_bridge_outcome", outcome)
        output.outcome = outcome
        output.late_present = functools.partial(
            self.present_output, event, presentation
        )
        output.seal(event)
        attach_presentation(req, outcome)

    @filter.after_message_sent(priority=-20000)
    async def deliver_attachments(self, event: AstrMessageEvent):
        """The presentation boundary releases immutable data after the reply."""
        output = event.get_extra("capability_bridge_output")
        if output and not output.released:
            await output.release(event)

    @filter.on_llm_response(priority=100000)
    async def observe_presentation(self, event: AstrMessageEvent, response):
        """Remember generation success before response filters can suppress text.

        Args:
            event: The current message event and its execution output buffer.
            response: The model response before other response hooks run.
        """
        output = event.get_extra("capability_bridge_output")
        if output:
            output.presentation_generated = bool(
                response is not None
                and response.role == "assistant"
                and response.completion_text
                and response.completion_text.strip()
            )

    @filter.on_llm_response(priority=-20000)
    async def guard_presentation(self, event: AstrMessageEvent, response):
        """Preserve completed tool output if the normal persona request fails."""
        output = event.get_extra("capability_bridge_output")
        if not output:
            return
        if (
            response is not None
            and response.role == "assistant"
            and response.completion_text
            and response.completion_text.strip()
        ):
            output.presentation_succeeded = True
            return
        if (
            response is not None
            and response.role == "assistant"
            and output.presentation_generated
        ):
            # Other plugins suppressed generated text; preserve only immutable output.
            output.presentation_succeeded = True
            await output.release(event)
            return
        await output.release(event, fallback=True)

    async def present_output(self, event, presentation, message):
        """Render confirmations and late plugin messages with no execution tools."""
        text = message.get_plain_text()
        media = [part for part in message.chain if not isinstance(part, Plain)]
        if text:
            try:
                response = await self.context.llm_generate(
                    chat_provider_id=presentation["provider_id"],
                    prompt=json.dumps(
                        {
                            "request": presentation.get("prompt", ""),
                            "plugin_output": text,
                        },
                        ensure_ascii=False,
                    ),
                    system_prompt=presentation["system_prompt"]
                    + PRESENTATION_INSTRUCTION,
                    tools=None,
                )
                text = response.completion_text or text
            except Exception as exc:  # noqa: BLE001 - Preserve output if the model is down.
                logger.warning("Persona presentation failed (%s)", type(exc).__name__)
            await event.send(MessageChain().message(text))
        if media:
            await event.send(MessageChain(chain=media))

    @filter.llm_tool(name="capability_find")
    async def capability_find(self, event: AstrMessageEvent, query: str) -> str:
        """查找已适配的插件功能或当前可用原生工具。用户要求查询、生成或执行功能时使用。

        Args:
            query(string): 用户的功能需求，保留插件名、游戏名、动作、查询对象等关键信息。
        """
        try:
            self.ensure_access(event)
        except CapabilityError as exc:
            return json.dumps({"error": str(exc)}, ensure_ascii=False)
        if self.dirty:
            await self.refresh()
        query = str(query)[:1000].casefold()
        terms = set(re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", query))
        terms.update(
            query[i : i + 2]
            for i in range(len(query) - 1)
            if all("\u4e00" <= c <= "\u9fff" for c in query[i : i + 2])
        )
        candidates = []
        for name, adaptation in self.state["adaptations"].items():
            record = self.catalog.get(name)
            if not record or adaptation.get("fingerprint") != record["fingerprint"]:
                continue
            for cap in adaptation.get("capabilities", []):
                if not cap.get("enabled"):
                    continue
                entry = record["commands"].get(cap["command_id"])
                if not entry:
                    continue
                try:
                    await check_access(self.context, event, record, entry)
                except CapabilityError:
                    continue
                mode, reviewed = call_policy(
                    self.state, cap["id"], record["fingerprint"]
                )
                if mode == "admin_confirm" and not event.is_admin():
                    continue
                text = (
                    cap["name"] + cap["description"] + name + " ".join(cap["examples"])
                ).casefold()
                score = sum((2 if len(t) > 1 else 0.25) for t in terms if t in text)
                candidates.append(
                    (
                        score,
                        {
                            k: cap[k]
                            for k in [
                                "id",
                                "name",
                                "description",
                                "input_schema",
                                "permission",
                            ]
                        }
                        | {
                            "kind": "adapted",
                            "execute_with": "capability_run",
                            "call_policy": mode,
                            "policy_reviewed": reviewed,
                        },
                    )
                )
        for tool in event.get_extra("capability_bridge_native", []):
            text = (tool["name"] + tool["description"]).casefold()
            score = sum((2 if len(t) > 1 else 0.25) for t in terms if t in text)
            candidates.append(
                (score, {**tool, "kind": "native", "execute_with": tool["name"]})
            )
        candidates.sort(key=lambda pair: pair[0], reverse=True)
        return json.dumps(
            {
                "matches": [v for score, v in candidates[:5] if score > 0],
                "instruction": "匹配到原生工具时直接调用该工具；适配功能用 capability_run。未找到时如实说明，可请管理员在能力中心适配。",
            },
            ensure_ascii=False,
        )

    @filter.llm_tool(name="capability_run")
    async def capability_run(
        self, event: AstrMessageEvent, capability_id: str, arguments: dict
    ) -> str:
        """执行 capability_find 返回的已校验能力，保留原插件权限、账号绑定与结果发送。

        Args:
            capability_id(string): capability_find 返回的适配能力 id。
            arguments(object): 严格符合该能力 input_schema 的参数对象；对象必须来自用户的明确请求。
        """
        try:
            self.ensure_access(event)
        except CapabilityError as exc:
            return json.dumps({"error": str(exc)}, ensure_ascii=False)
        if not isinstance(arguments, dict):
            return '{"error":"参数必须是对象"}'
        if self.dirty:
            await self.refresh()
        cache = event.get_extra("capability_bridge_runs", None)
        if cache is None:
            cache = {}
            event.set_extra("capability_bridge_runs", cache)
        key = (
            self.controls_token(self.settings())
            + capability_id
            + json.dumps(arguments, sort_keys=True, ensure_ascii=False)
        )
        if key in cache:
            return cache[key]
        # Cache a pending result before awaiting to prevent concurrent duplicates.
        cache[key] = '{"status":"already_started","do_not_repeat":true}'
        result = await self._run_capability(event, capability_id, arguments)
        cache[key] = json.dumps(result, ensure_ascii=False)
        return cache[key]

    async def _run_capability(
        self, event, capability_id, arguments, approved_snapshot=None
    ):
        """Route tool calls and human approvals through the same permission gate."""
        try:
            self.ensure_access(event)
        except CapabilityError as exc:
            return {"error": str(exc)}
        for name, adaptation in self.state["adaptations"].items():
            for cap in adaptation.get("capabilities", []):
                if cap["id"] != capability_id or not cap.get("enabled"):
                    continue
                try:
                    if name not in self.catalog:
                        raise CapabilityError("原插件已卸载")
                    mode, _ = call_policy(
                        self.state, cap["id"], self.catalog[name]["fingerprint"]
                    )
                    if mode == "admin_confirm" and not event.is_admin():
                        raise CapabilityError("该功能仅限当前会话的 AstrBot 管理员使用")

                    async def authorize(snapshot):
                        return await self._authorize_call(
                            event, cap, arguments, snapshot, approved_snapshot
                        )

                    result = await execute(
                        self.context,
                        event,
                        cap,
                        self.catalog[name],
                        arguments,
                        adaptation["fingerprint"],
                        before_execute=authorize,
                    )
                except CapabilityError as exc:
                    result = {"error": str(exc)}
                except Exception as exc:  # noqa: BLE001 - Isolate provider and plugin failures.
                    logger.warning(
                        "Capability execution failed (%s)", type(exc).__name__
                    )
                    result = {
                        "error": "原插件执行失败或超时，请稍后重试",
                        "do_not_retry_automatically": True,
                    }
                return result
        return {"error": "能力不存在、已停用或尚未适配，请先查找可用能力"}

    async def _authorize_call(self, event, cap, arguments, snapshot, approved_snapshot):
        """Authorize final arguments without trusting model-side confirmations."""
        settings = self.ensure_access(event)
        mode, _ = call_policy(self.state, cap["id"], snapshot["fingerprint"])
        if mode == "admin_confirm" and not event.is_admin():
            raise CapabilityError("该功能仅限当前会话的 AstrBot 管理员使用")
        snapshot = {
            **snapshot,
            "capability_id": cap["id"],
            "policy": mode,
            "controls_revision": self.controls_token(settings),
        }
        if approved_snapshot is None and (
            mode == "direct" or not settings["confirmation_enabled"]
        ):
            return None
        # A later confirmation message does not carry the original attachment or
        # reply. Fail closed rather than silently authorizing a different input.
        self_id = str(getattr(event, "get_self_id", lambda: "")())
        if any(
            not isinstance(component, Plain)
            and not (isinstance(component, At) and str(component.qq) == self_id)
            for component in event.get_messages()
        ):
            raise CapabilityError(
                "此操作依赖图片、引用或其他消息对象，暂不支持确认后执行；"
                "请使用原插件指令，或由管理员审阅后调整调用方式"
            )
        if approved_snapshot is not None:
            if snapshot != approved_snapshot:
                raise CapabilityError(
                    "操作参数或调用方式已经变化，请重新发起请求并确认"
                )
            return None
        preview = {**snapshot["parameters"]}
        if snapshot["message_tail"] is not None:
            preview["消息正文"] = snapshot["message_tail"]
        text = json.dumps(preview, ensure_ascii=False, indent=2, allow_nan=False)
        if len(text) > 3000:
            raise CapabilityError(
                "操作参数过长，无法完整展示确认内容，请使用原插件指令"
            )
        code = self.confirmations.request(event, cap["id"], arguments, snapshot)
        message = (
            f"待确认：{cap['name']}\n原指令：{snapshot['command']}\n"
            f"最终参数：\n{text}\n\n尚未执行。确认无误后，请本人在本会话发送：\n"
            f"/确认执行 {code}\n确认码两分钟内有效，仅可使用一次。"
        )
        output = event.get_extra("capability_bridge_output")
        if output:
            output.protect(message)
            self.confirmations.items[code]["presentation"] = copy.copy(
                event.get_extra("capability_bridge_presentation", {})
            )
        else:
            await event.send(MessageChain().message(message))
        return {
            "status": "confirmation_required",
            "delivered": True,
            "do_not_repeat": True,
            "message": "尚未执行，已向请求者发送最终参数及确认方式，等待本人确认。",
        }

    @filter.command("确认执行")
    async def confirm_execution(self, event: AstrMessageEvent, confirmation_code: str):
        """Execute one incoming, caller-bound confirmation through normal checks."""
        event.stop_event()
        # Require the framework's original command dispatch proof. A model or a
        # generic command executor calling this method cannot manufacture consent.
        dispatched = any(
            h.handler_name == "confirm_execution"
            and (
                (
                    getattr(h.handler, "__self__", None) is self
                    and getattr(h.handler, "__func__", None)
                    is type(self).confirm_execution
                )
                or (
                    isinstance(h.handler, functools.partial)
                    and h.handler.args
                    and h.handler.args[0] is self
                    and h.handler.func is type(self).confirm_execution
                )
            )
            and event.get_extra("handlers_parsed_params", {})
            .get(h.handler_full_name, {})
            .get("confirmation_code")
            == confirmation_code
            for h in event.get_extra("activated_handlers", [])
        )
        if not dispatched:
            yield event.plain_result("请本人在原会话直接发送确认指令，工具不能代为确认")
            return
        try:
            settings = self.ensure_access(event)
            if not settings["confirmation_enabled"]:
                yield event.plain_result(
                    "二次确认已关闭，旧确认码已失效；请重新发起需求"
                )
                return
            pending = self.confirmations.consume(event, confirmation_code)
            await self.refresh()
            presentation = pending.get("presentation")
            output = OutputBuffer() if presentation else None
            execution_event = output.proxy(event) if output else event
            result = await self._run_capability(
                execution_event,
                pending["capability_id"],
                pending["arguments"],
                approved_snapshot=pending["snapshot"],
            )
        except ValueError as exc:
            yield event.plain_result(str(exc))
            return
        if output:
            summary = (
                result.get("error")
                or result.get("text")
                or "原插件执行流程已结束，请根据实际输出说明结果"
            )
            if output.texts:
                summary += "\n" + "\n".join(output.texts)
            await self.present_output(
                event, presentation, MessageChain().message(summary)
            )
            output.late_present = functools.partial(
                self.present_output, event, presentation
            )
            await output.release(event)
            return
        if not result.get("delivered"):
            yield event.plain_result(result.get("error") or "操作已执行")

    async def terminate(self):
        """Stop a running analysis job during plugin unload."""
        self.closed = True
        self.confirmations.items.clear()
        getattr(self, "execution_turns", {}).clear()
        if self.job_task and not self.job_task.done():
            self.job_task.cancel()
            await self.job_task
