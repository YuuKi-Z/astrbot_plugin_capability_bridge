"""Validate declarative model output against registered plugin interfaces."""

import hashlib
import json
import re

from jsonschema import Draft202012Validator


def parse_response(text: str) -> dict:
    """Decode a model JSON response, accepting a single fenced JSON block.

    Args:
        text: Model completion text.

    Returns:
        Parsed adaptation object.

    Raises:
        ValueError: The response is not a bounded JSON adaptation.
    """
    if len(text) > 150000:
        raise ValueError("模型响应超过上限")
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    result = json.loads(text)
    if not isinstance(result, dict) or not isinstance(result.get("capabilities"), list):
        raise TypeError("模型必须返回包含 capabilities 数组的 JSON")
    return result


def validate_plan(raw: dict, record: dict) -> dict:
    """Validate one generated mapping without executing its business handler.

    Args:
        raw: Model-generated capability mapping.
        record: Current scanned plugin record.

    Returns:
        A normalized capability mapping with a stable identifier.

    Raises:
        ValueError: The mapping references an unsupported interface or shape.
    """
    if not isinstance(raw, dict):
        raise TypeError("能力定义必须是对象")
    permitted = {
        "name",
        "description",
        "examples",
        "command_id",
        "input_schema",
        "argument_map",
        "subject_binding",
        "message_tail",
    }
    if set(raw) - permitted:
        raise ValueError("能力包含未支持的字段")
    command_id = raw.get("command_id")
    entry = record["commands"].get(command_id)
    if not entry or not entry["enabled"]:
        raise ValueError("目标不是当前已启用且无冲突的指令")
    if any(p.get("positional_only") for p in entry["parameters"].values()):
        raise ValueError("该指令含有当前不支持的位置专用参数")
    if not isinstance(raw.get("name"), str) or not 1 <= len(raw["name"]) <= 100:
        raise ValueError("能力名称无效")
    if (
        not isinstance(raw.get("description"), str)
        or not 1 <= len(raw["description"]) <= 1000
    ):
        raise ValueError("能力说明无效")
    schema = raw.get("input_schema")
    if not isinstance(schema, dict) or schema.get("type") != "object":
        raise ValueError("input_schema 必须是 object")
    if "$ref" in json.dumps(schema) or len(json.dumps(schema)) > 16000:
        raise ValueError("参数 schema 不能包含引用或超过大小限制")
    Draft202012Validator.check_schema(schema)
    properties = schema.get("properties", {})
    if not isinstance(properties, dict) or len(properties) > 24:
        raise ValueError("参数数量超过上限")
    if any(not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", key) for key in properties):
        raise ValueError("参数名无效")
    if set(schema.get("required", [])) - set(properties):
        raise ValueError("必填输入必须在 properties 中声明")
    schema = {**schema, "additionalProperties": False}
    argument_map = raw.get("argument_map", {})
    if not isinstance(argument_map, dict) or set(argument_map) - set(
        entry["parameters"]
    ):
        raise ValueError("映射包含原指令不存在的参数")
    for target, expression in argument_map.items():
        if not isinstance(expression, dict) or len(expression) != 1:
            raise ValueError(f"参数 {target} 必须使用 input 或 value 表达式")
        if "input" in expression:
            if expression["input"] not in properties:
                raise ValueError("引用了不存在的输入参数")
        elif "value" in expression:
            if not isinstance(expression["value"], (str, int, float, bool, type(None))):
                raise ValueError("常量参数只能是简单 JSON 值")
        else:
            raise ValueError("不支持的参数表达式")
    message_tail = raw.get("message_tail")
    if message_tail is not None:
        if not entry.get("accepts_message_tail"):
            raise ValueError("原指令缺少从消息正文读取参数的源码证据")
        if not isinstance(message_tail, dict) or set(message_tail) != {"input"}:
            raise ValueError("消息尾部映射必须且只能包含 input")
        source = properties.get(message_tail["input"], {})
        if source.get("type") != "string":
            raise ValueError("消息尾部必须引用已声明的字符串输入")
    binding = raw.get("subject_binding")
    if binding:
        expected = {"input", "lookup_method", "target_parameter", "missing_method"}
        if not isinstance(binding, dict) or set(binding) != expected:
            raise ValueError("主体绑定定义不完整")
        if (
            binding["input"] not in properties
            or properties[binding["input"]].get("type") != "string"
        ):
            raise ValueError("主体参数必须是已声明的字符串")
        if binding["target_parameter"] not in entry["parameters"]:
            raise ValueError("绑定目标参数不存在")
        lookup = record["helpers"].get(binding["lookup_method"], {})
        hint = record["helpers"].get(binding["missing_method"], {})
        if lookup.get("kind") != "subject_lookup" or hint.get("kind") != "literal_hint":
            raise ValueError("绑定查询或未绑定提示缺少已校验的源码证据")
    required = {name for name, info in entry["parameters"].items() if info["required"]}
    supplied = set(argument_map)
    if binding:
        supplied.add(binding["target_parameter"])
    if required - supplied:
        raise ValueError("未覆盖原指令的必填参数")
    examples = raw.get("examples", [])
    if (
        not isinstance(examples, list)
        or len(examples) > 6
        or any(not isinstance(e, str) or len(e) > 200 for e in examples)
    ):
        raise ValueError("自然语言示例无效")
    capability_id = (
        "cap_"
        + hashlib.sha256((record["name"] + ":" + command_id).encode()).hexdigest()[:16]
    )
    return {
        "id": capability_id,
        "plugin": record["name"],
        "name": raw["name"],
        "description": raw["description"],
        "examples": examples,
        "command_id": command_id,
        "command": entry["command"],
        "permission": entry["permission"],
        "input_schema": schema,
        "argument_map": argument_map,
        "message_tail": message_tail,
        "subject_binding": binding or None,
        "enabled": True,
        "validation": "schema_and_source_checked",
    }
