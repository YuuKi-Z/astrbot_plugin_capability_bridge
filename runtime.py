"""Execute validated mappings using the caller's original AstrBot identity."""

import asyncio
import copy
import inspect
import json
import re

from astrbot.api.message_components import Plain
from astrbot.core.message.message_event_result import MessageChain, MessageEventResult
from astrbot.core.star.filter.command import CommandFilter
from astrbot.core.star.filter.command_group import CommandGroupFilter
from astrbot.core.star.filter.permission import PermissionTypeFilter
from astrbot.core.star.session_plugin_manager import SessionPluginManager
from jsonschema import Draft202012Validator

from .scanner import fingerprint


class CapabilityError(ValueError):
    """An actionable execution error safe to return to the chat model."""


async def resolve_member(event, query: str) -> dict:
    """Resolve self, an ID, a mention, or an unambiguous current-group name.

    Args:
        event: Real caller event.
        query: User-specified target, never an invented model-side identity.

    Returns:
        The verified target ID and display name.

    Raises:
        CapabilityError: The target cannot be resolved within the current chat.
    """
    query = str(query or "").strip()
    sender_id = str(event.get_sender_id())
    if query.casefold() in {"", "我", "自己", "本人", "me", "self", sender_id}:
        return {"id": sender_id, "name": event.get_sender_name()}
    if not event.get_group_id():
        raise CapabilityError(
            "私聊中查询其他人时，请提供明确的账号；群昵称只能在当前群内识别"
        )
    matched = re.fullmatch(
        r"(?:\[CQ:at,qq=(\d+)[^\]]*\]|<@!?(\d+)>|@?(\d{5,20}))", query
    )
    requested_id = next((g for g in matched.groups() if g), None) if matched else None
    try:
        if event.get_platform_name() == "aiocqhttp":
            members = await asyncio.wait_for(
                event.bot.call_action(
                    "get_group_member_list", group_id=int(event.get_group_id())
                ),
                timeout=8,
            )
            if isinstance(members, dict):
                members = members.get("data", [])
        else:
            group = await asyncio.wait_for(event.get_group(), timeout=8)
            members = [
                {
                    "user_id": m.user_id,
                    "nickname": getattr(m, "nickname", "") or getattr(m, "name", ""),
                }
                for m in group.members
            ]
    except Exception as exc:
        raise CapabilityError(
            "暂时无法获取当前群成员，请用明确的外部账号查询或稍后重试"
        ) from exc
    if not isinstance(members, list):
        raise CapabilityError("当前平台未返回可用的群成员列表")
    normalized = []
    for member in members:
        if not isinstance(member, dict) or member.get("user_id") is None:
            continue
        member_id = str(member["user_id"])
        names = {str(member.get("card") or ""), str(member.get("nickname") or "")}
        normalized.append(
            {
                "id": member_id,
                "names": names,
                "name": str(member.get("card") or member.get("nickname") or member_id),
            }
        )
    needle = query.lstrip("@").casefold()
    matches = (
        [m for m in normalized if m["id"] == requested_id]
        if requested_id
        else [
            m for m in normalized if needle in {n.casefold() for n in m["names"] if n}
        ]
    )
    if not matches and not requested_id and len(needle) >= 2:
        matches = [
            m
            for m in normalized
            if any(needle in n.casefold() for n in m["names"] if n)
        ]
    if len(matches) > 1:
        raise CapabilityError(
            "当前群有多个匹配成员，请明确 @其中一位："
            + "、".join(m["name"] for m in matches[:5])
        )
    if not matches:
        raise CapabilityError("在当前群中没有找到该成员，请明确 @对方或提供其账号")
    return {"id": matches[0]["id"], "name": matches[0]["name"]}


async def check_access(context, event, record: dict, entry: dict) -> None:
    """Enforce current activation, session scope, and command permissions.

    Args:
        context: AstrBot context.
        event: Original caller event.
        record: Current scanned plugin record.
        entry: Registered command record.

    Raises:
        CapabilityError: Current caller cannot use this command.
    """
    metadata = record["metadata"]
    if context.get_registered_star(metadata.name) is not metadata:
        raise CapabilityError("插件已重载或卸载，请刷新能力目录")
    if (
        not metadata.activated
        or metadata.star_cls is None
        or not entry["enabled"]
        or not entry["handler"].enabled
    ):
        raise CapabilityError("该插件或指令已被停用，请刷新能力目录")
    if event.plugins_name is not None and metadata.name not in event.plugins_name:
        raise CapabilityError("当前会话未启用该插件")
    handler = entry["handler"]
    if not await SessionPluginManager.filter_handlers_by_session(event, [handler]):
        raise CapabilityError("当前会话已禁用该插件")
    config = context.get_config(umo=event.unified_msg_origin)
    for registered in [*entry.get("parents", []), handler]:
        if not registered.enabled:
            raise CapabilityError("该指令或所属指令组已停用")
        for rule in registered.event_filters:
            if isinstance(rule, PermissionTypeFilter) and not rule.filter(
                event, config
            ):
                raise CapabilityError("你当前的权限不足以执行这个功能")


async def execute(
    context,
    event,
    capability: dict,
    record: dict,
    arguments: dict,
    expected_fingerprint: str,
) -> dict:
    """Run a validated mapping, preserving binding targets and native output.

    Args:
        context: AstrBot plugin context.
        event: Real incoming event.
        capability: Validated adaptation record.
        record: Current scanned plugin interfaces.
        arguments: User arguments checked against the generated schema.
        expected_fingerprint: Source version used to generate this adaptation.

    Returns:
        A bounded result for the chat model; native messages are delivered once.

    Raises:
        CapabilityError: The mapping is stale, not allowed, or lacks valid input.
    """
    if (
        fingerprint(record["root"], record["metadata"].version or "")
        != expected_fingerprint
    ):
        raise CapabilityError("插件源码已变化，请在能力中心重新适配后使用")
    entry = record["commands"].get(capability["command_id"])
    if not entry:
        raise CapabilityError("原指令已不存在，请重新适配")
    await check_access(context, event, record, entry)
    if (
        not isinstance(arguments, dict)
        or len(json.dumps(arguments, ensure_ascii=False)) > 20000
    ):
        raise CapabilityError("参数必须是大小受限的对象")
    errors = sorted(
        Draft202012Validator(capability["input_schema"]).iter_errors(arguments),
        key=lambda e: str(e.path),
    )
    if errors:
        raise CapabilityError("参数不符合能力要求：" + errors[0].message[:250])
    values = dict(arguments)
    for key, info in capability["input_schema"].get("properties", {}).items():
        if key not in values and "default" in info:
            values[key] = info["default"]
    params = {}
    for target, expression in capability["argument_map"].items():
        if "input" in expression:
            if expression["input"] in values:
                params[target] = values[expression["input"]]
        else:
            params[target] = expression["value"]
    message_tail = capability.get("message_tail")
    tail_text = None
    if message_tail is not None:
        if not entry.get("accepts_message_tail"):
            raise CapabilityError("原指令的消息正文参数需要重新校验")
        tail_text = values.get(message_tail["input"], "")
        if not isinstance(tail_text, str) or len(tail_text) > 8000:
            raise CapabilityError("消息尾部参数必须是不超过 8000 字符的文字")
    proxy = copy.copy(event)
    proxy._extras = dict(event.get_extra() or {})
    proxy._result = None
    proxy._has_send_oper = False
    proxy.message_obj = copy.copy(event.message_obj)
    proxy.message_str = (
        entry["command"]
        + " "
        + (
            tail_text
            if tail_text is not None
            else " ".join(
                str(params[k])
                for k in entry["parameters"]
                if k in params and params[k] is not None
            )
        )
    ).rstrip()
    proxy.message_obj.message_str = proxy.message_str
    proxy.message_obj.message = [Plain(text=proxy.message_str)] + [
        c for c in event.get_messages() if not isinstance(c, Plain)
    ]
    proxy.is_at_or_wake_command = True
    proxy.set_extra("parsed_params", params)
    config = context.get_config(umo=event.unified_msg_origin)
    for registered in [*entry.get("parents", []), entry["handler"]]:
        for rule in registered.event_filters:
            if isinstance(rule, (CommandFilter, CommandGroupFilter)):
                if not rule.custom_filter_ok(proxy, config):
                    raise CapabilityError("原指令的使用条件未满足")
            elif not isinstance(rule, PermissionTypeFilter) and not rule.filter(
                proxy, config
            ):
                raise CapabilityError("当前平台或消息未满足原插件的使用条件")
    binding = capability.get("subject_binding")
    subject = None
    if binding and not params.get(binding["target_parameter"]):
        subject = await resolve_member(event, values.get(binding["input"], "我"))
        helper = record["helpers"].get(binding["lookup_method"], {})
        hint = record["helpers"].get(binding["missing_method"], {})
        if helper.get("kind") != "subject_lookup" or hint.get("kind") != "literal_hint":
            raise CapabilityError("绑定解析方法需要重新校验")
        getter = getattr(record["metadata"].star_cls, binding["lookup_method"])
        account = getter(subject["id"])
        if inspect.isawaitable(account):
            account = await asyncio.wait_for(account, timeout=8)
        if not account:
            message = hint["value"]
            prefix = (
                ""
                if subject["id"] == str(event.get_sender_id())
                else f"查询对象：{subject['name']}\n"
            )
            text = prefix + str(message)[:2000]
            await event.send(MessageChain().message(text))
            return {
                "status": "missing_binding",
                "delivered": True,
                "text": text,
                "do_not_repeat": True,
            }
        if not isinstance(account, (str, int)) or len(str(account)) > 512:
            raise CapabilityError("绑定接口返回了不支持的账号格式")
        params[binding["target_parameter"]] = str(account)
    for name, value in params.items():
        kind = entry["parameters"][name]["type"]
        if value is None and "None" in entry["parameters"][name]["annotation"]:
            continue
        valid = (
            (kind == "string" and isinstance(value, str))
            or (
                kind == "integer"
                and isinstance(value, int)
                and not isinstance(value, bool)
            )
            or (
                kind == "number"
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
            )
            or (kind == "boolean" and isinstance(value, bool))
            or (kind == "array" and isinstance(value, list))
            or (kind == "object" and isinstance(value, dict))
        )
        if not valid:
            raise CapabilityError(f"原指令参数 {name} 的类型应为 {kind}")
    for name, info in entry["parameters"].items():
        if info["required"] and name not in params:
            raise CapabilityError(f"缺少原指令所需参数：{name}")
    proxy.set_extra("parsed_params", params)
    proxy.message_str = (
        entry["command"]
        + " "
        + (
            tail_text
            if tail_text is not None
            else " ".join(
                str(params[k])
                for k in entry["parameters"]
                if k in params and params[k] is not None
            )
        )
    ).rstrip()
    proxy.message_obj.message_str = proxy.message_str
    proxy.message_obj.message = [Plain(text=proxy.message_str)] + [
        c for c in event.get_messages() if not isinstance(c, Plain)
    ]
    result = entry["handler"].handler(proxy, **params)
    texts = []
    sent = 0
    if inspect.isasyncgen(result):
        try:
            while True:
                try:
                    item = await asyncio.wait_for(anext(result), timeout=120)
                except StopAsyncIteration:
                    break
                if isinstance(item, MessageEventResult) and item.chain:
                    await proxy.send(
                        MessageChain(chain=item.chain, type="tool_direct_result")
                    )
                    texts.append(item.get_plain_text()[:2000])
                    sent += 1
                elif isinstance(item, str) and item:
                    await proxy.send(MessageChain().message(item[:4000]))
                    texts.append(item[:2000])
                    sent += 1
                proxy.clear_result()
        finally:
            await result.aclose()
    elif inspect.isawaitable(result):
        item = await asyncio.wait_for(result, timeout=120)
        if isinstance(item, MessageEventResult) and item.chain:
            await proxy.send(MessageChain(chain=item.chain, type="tool_direct_result"))
            texts.append(item.get_plain_text()[:2000])
            sent += 1
        elif isinstance(item, str) and item:
            await proxy.send(MessageChain().message(item[:4000]))
            texts.append(item[:2000])
            sent += 1
    if proxy._has_send_oper or sent:
        event._has_send_oper = True
    return {
        "status": "completed",
        "delivered": bool(proxy._has_send_oper or sent),
        "text": "\n".join(texts)[-4000:],
        "do_not_repeat": bool(proxy._has_send_oper or sent),
    }
