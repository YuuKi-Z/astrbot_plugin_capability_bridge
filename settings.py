"""Shared configuration validation and access decisions, independent of AstrBot."""

import copy
import hashlib
import json

DEFAULT_SETTINGS = {
    "enabled": True,
    "confirmation_enabled": True,
    "analysis_provider_id": "",
    "execution_provider_id": "",
    "allow_private_chat": True,
    "group_whitelist_enabled": False,
    "group_whitelist": [],
    "group_blacklist": [],
    "user_blacklist": [],
}


def read_settings(config) -> dict:
    """Return independent defaults and reject malformed security settings."""
    result = copy.deepcopy(DEFAULT_SETTINGS)
    for key, default in DEFAULT_SETTINGS.items():
        if key not in config:
            continue
        value = config[key]
        if isinstance(default, bool):
            if type(value) is not bool:
                raise ValueError(f"设置 {key} 必须是开关")
        elif isinstance(default, list):
            if not isinstance(value, list) or len(value) > 500:
                raise ValueError(f"设置 {key} 必须是不超过 500 项的名单")
            clean = []
            for item in value:
                if not isinstance(item, str) or not item.strip():
                    raise ValueError("名单中的每一项都需要填写完整的群号或用户 ID")
                item = item.strip()
                if len(item) > 256 or any(char.isspace() for char in item):
                    raise ValueError("名单中的 ID 不能包含空白字符或超过 256 个字符")
                if item not in clean:
                    clean.append(item)
            value = clean
        elif not isinstance(value, str) or len(value) > 256:
            raise ValueError("模型 ID 必须是有效的字符串")
        result[key] = value
    return result


def validate_update(current: dict, update: dict, provider_ids: set[str]) -> dict:
    """Validate the complete change before writing either UI's configuration."""
    if not isinstance(update, dict) or set(update) - set(DEFAULT_SETTINGS):
        raise ValueError("设置包含不支持的字段")
    result = read_settings({**current, **update})
    for key in ("analysis_provider_id", "execution_provider_id"):
        if key in update and result[key] and result[key] not in provider_ids:
            raise ValueError("请选择 AstrBot 中已配置的聊天模型，或留空")
    return result


def settings_revision(settings: dict, *, controls_only=False) -> str:
    """Detect stale forms and invalidate approvals when control settings change."""
    values = dict(settings)
    if controls_only:
        values.pop("analysis_provider_id", None)
    return hashlib.sha256(
        json.dumps(values, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def access_error(settings: dict, event, plugin_name: str) -> str | None:
    """Apply explicit scope to every entry point; administrators get no bypass."""
    if not settings["enabled"]:
        return "自然语言指令已停用"
    plugins = getattr(event, "plugins_name", None)
    if plugins is not None and "*" not in plugins and plugin_name not in plugins:
        return "当前会话未启用自然语言指令"
    platform = str(event.get_platform_name())

    def matches(value, entries):
        value = str(value)
        return value in entries or f"{platform}:{value}" in entries

    if matches(event.get_sender_id(), settings["user_blacklist"]):
        return "当前用户不在自然语言指令的允许范围内"
    group = event.get_group_id()
    if not group:
        return (
            None if settings["allow_private_chat"] else "自然语言指令当前不允许私聊使用"
        )
    if matches(group, settings["group_blacklist"]):
        return "当前群不在自然语言指令的允许范围内"
    if settings["group_whitelist_enabled"] and not matches(
        group, settings["group_whitelist"]
    ):
        return "当前群未加入自然语言指令的白名单"
    return None
