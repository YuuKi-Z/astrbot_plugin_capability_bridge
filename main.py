"""Discover plugin interfaces, generate cached mappings, and expose two tools."""

import asyncio
import json
import math
import re
import time
from pathlib import Path

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star
from astrbot.api.web import error_response, json_response, request
from astrbot.core.utils.astrbot_path import get_astrbot_plugin_data_path
from jsonschema.exceptions import SchemaError

from .plans import parse_response, validate_plan
from .prompts import ANALYZE_PROMPT
from .runtime import CapabilityError, check_access, execute
from .scanner import PLUGIN_NAME, analysis_document, scan

BRIDGE_INSTRUCTION = """\n[CAPABILITY_BRIDGE]
用户明确请求执行某个功能时，主动判断并调用可用工具。需要查找插件功能时先调用 capability_find，按返回的参数说明调用 capability_run；原生工具直接用其原名调用。不把‘能不能帮我查/生成’仅回答成能力介绍。缺少必要信息才追问。能力不存在或未适配要如实说明，不能编造已执行。target_user 必须来自用户的明确指代，查别人不能替换成自己。工具返回 delivered/do_not_repeat 表示原插件已发送结果，不重复转述整份结果。\n"""


ANALYSIS_SOURCE_LIMIT = 48000
ANALYSIS_OUTPUT_TOKENS = 6000


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
        }
        if self.state_path.exists():
            try:
                loaded = json.loads(self.state_path.read_text(encoding="utf-8"))
                self.state["settings"]["provider_id"] = loaded.get("settings", {}).get(
                    "provider_id", ""
                )
                self.state["adaptations"] = loaded.get("adaptations", {})
            except (OSError, ValueError):
                logger.warning("Capability bridge state could not be loaded")
        self.catalog = {}
        self.dirty = True
        self.scan_lock = asyncio.Lock()
        self.job_task = None
        self.job = {"status": "idle", "items": [], "tokens": 0, "estimated_tokens": 0}
        for path, handler, methods in [
            ("status", self.api_status, ["GET"]),
            ("scan", self.api_scan, ["POST"]),
            ("settings", self.api_settings, ["POST"]),
            ("adapt", self.api_adapt, ["POST"]),
            ("cancel", self.api_cancel, ["POST"]),
            ("toggle", self.api_toggle, ["POST"]),
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
                    "active": record["active"],
                    "state": state,
                    "command_count": len(record["commands"]),
                    "native_tools": record["native_tools"],
                    "capabilities": caps,
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
            "settings": self.state["settings"],
            "providers": providers,
            "plugins": plugins,
            "job": self.job,
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
        """Save the existing provider used for source analysis."""
        if not request.username:
            return error_response("需要登录管理后台", status_code=401)
        data = await request.json(default={})
        if not isinstance(data, dict):
            return error_response("设置必须为对象")
        settings = dict(self.state["settings"])
        if "provider_id" in data:
            if data["provider_id"] not in {
                p.meta().id for p in self.context.get_all_providers()
            }:
                return error_response("请选择已配置的聊天模型")
            settings["provider_id"] = data["provider_id"]
        self.state["settings"] = settings
        self.save()
        return json_response({"saved": True})

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
        settings = dict(self.state["settings"])
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
                    commands = [e for e in record["commands"].values() if e["enabled"]]
                    batches = [commands[i : i + 8] for i in range(0, len(commands), 8)]
                    for batch_index, batch in enumerate(batches):
                        item["batch"] = f"{batch_index + 1}/{len(batches)}"
                        batch_record = {
                            **record,
                            "commands": {e["id"]: e for e in batch},
                        }
                        document = analysis_document(
                            batch_record, ANALYSIS_SOURCE_LIMIT
                        )
                        # A conservative character estimate reserves output before each call.
                        estimate = (
                            len(document) + len(ANALYZE_PROMPT) + ANALYSIS_OUTPUT_TOKENS
                        )
                        self.job["estimated_tokens"] += estimate
                        response = await asyncio.wait_for(
                            self.context.llm_generate(
                                chat_provider_id=settings["provider_id"],
                                prompt=document,
                                system_prompt=ANALYZE_PROMPT,
                                max_tokens=ANALYSIS_OUTPUT_TOKENS,
                                temperature=0.1,
                            ),
                            timeout=180,
                        )
                        self.job["estimated_tokens"] -= estimate
                        usage = getattr(response, "usage", None)
                        batch_tokens = int(usage.total) if usage else 0
                        tokens += batch_tokens
                        if batch_tokens:
                            self.job["tokens"] += batch_tokens
                        else:
                            self.job["estimated_tokens"] += math.ceil(
                                (
                                    len(document)
                                    + len(ANALYZE_PROMPT)
                                    + len(response.completion_text)
                                )
                                / 1.5
                            )
                        part = parse_response(response.completion_text)
                        parsed["capabilities"].extend(part["capabilities"])
                        parsed["unsupported"].extend(part.get("unsupported") or [])
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
                        raise ValueError("本次未产生有效适配，保留已有规则")
                    self.state["adaptations"][name] = {
                        "fingerprint": record["fingerprint"],
                        "capabilities": capabilities,
                        "unsupported": unsupported,
                        "adapted_at": int(time.time()),
                        "tokens": tokens,
                        "provider_id": settings["provider_id"],
                    }
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

    @filter.on_llm_request(priority=-20000)
    async def prepare_request(self, event: AstrMessageEvent, req):
        """Attach a compact instruction after other plugins assemble tools."""
        if not self.config.get("enabled", True):
            return
        available = getattr(req.func_tool, "tools", []) if req.func_tool else []
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
        if any(t.name == "capability_find" for t in available):
            req.system_prompt = (req.system_prompt or "") + BRIDGE_INSTRUCTION

    @filter.llm_tool(name="capability_find")
    async def capability_find(self, event: AstrMessageEvent, query: str) -> str:
        """查找已适配的插件功能或当前可用原生工具。用户要求查询、生成或执行功能时使用。

        Args:
            query(string): 用户的功能需求，保留插件名、游戏名、动作、查询对象等关键信息。
        """
        if not self.config.get("enabled", True):
            return '{"error":"能力桥已停用"}'
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
                        | {"kind": "adapted", "execute_with": "capability_run"},
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
        if not self.config.get("enabled", True):
            return '{"error":"能力桥已停用"}'
        if self.dirty:
            await self.refresh()
        cache = event.get_extra("capability_bridge_runs", None)
        if cache is None:
            cache = {}
            event.set_extra("capability_bridge_runs", cache)
        key = capability_id + json.dumps(arguments, sort_keys=True, ensure_ascii=False)
        if key in cache:
            return cache[key]
        for name, adaptation in self.state["adaptations"].items():
            for cap in adaptation.get("capabilities", []):
                if cap["id"] != capability_id or not cap.get("enabled"):
                    continue
                try:
                    if name not in self.catalog:
                        raise CapabilityError("原插件已卸载")
                    # Cache a pending result before awaiting to prevent concurrent duplicates.
                    cache[key] = '{"status":"already_started","do_not_repeat":true}'
                    result = await execute(
                        self.context,
                        event,
                        cap,
                        self.catalog[name],
                        arguments,
                        adaptation["fingerprint"],
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
                cache[key] = json.dumps(result, ensure_ascii=False)
                return cache[key]
        return '{"error":"能力不存在、已停用或尚未适配，请先查找可用能力"}'

    async def terminate(self):
        """Stop a running analysis job during plugin unload."""
        if self.job_task and not self.job_task.done():
            self.job_task.cancel()
            await self.job_task
