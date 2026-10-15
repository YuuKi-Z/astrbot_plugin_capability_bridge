"""Inspect registered interfaces and their source without executing plugins."""

import ast
import asyncio
import hashlib
import inspect
import json
import textwrap
from pathlib import Path

from astrbot.core.star.command_management import list_commands
from astrbot.core.star.filter.command import CommandFilter
from astrbot.core.star.star_handler import star_handlers_registry

PLUGIN_NAME = "astrbot_plugin_capability_bridge"
_FILE_CACHE = {}
_HELPER_CACHE = {}
_READ_CALLS = {
    "str",
    "int",
    "float",
    "bool",
    "dict",
    "list",
    "tuple",
    "set",
    "len",
    "isinstance",
    "get",
    "items",
    "keys",
    "values",
    "strip",
    "lower",
    "casefold",
    "exists",
    "is_file",
    "read_text",
    "loads",
    "warning",
    "debug",
    "info",
}
_DENIED_CALLS = {
    "eval",
    "exec",
    "compile",
    "__import__",
    "getattr",
    "setattr",
    "delattr",
    "open",
    "write",
    "write_text",
    "write_bytes",
    "unlink",
    "remove",
    "rmdir",
    "rmtree",
    "rename",
    "replace",
    "mkdir",
    "makedirs",
    "execute",
    "executemany",
    "commit",
    "put",
    "post",
    "patch",
    "delete",
    "send",
    "send_message",
    "call_action",
    "system",
    "popen",
    "run",
    "create_task",
    "create_subprocess_exec",
    "dump",
}


def fingerprint(root: Path, version: str) -> str:
    """Hash plugin Python sources, with a stat-keyed content cache.

    Args:
        root: Resolved plugin source directory.
        version: Plugin metadata version.

    Returns:
        A digest that changes when relevant source or metadata changes.
    """
    digest = hashlib.sha256(str(version).encode())
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root)
        if any(
            p
            in {
                ".git",
                ".venv",
                "venv",
                "tests",
                "test",
                "__pycache__",
                "site-packages",
            }
            for p in relative.parts
        ):
            continue
        if not path.resolve().is_relative_to(root):
            continue
        stat = path.stat()
        key = (str(path), stat.st_mtime_ns, stat.st_size)
        cached = _FILE_CACHE.get(str(path))
        if cached is None or cached[0] != key:
            file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            _FILE_CACHE[str(path)] = (key, file_hash)
        digest.update(str(relative).encode())
        digest.update(_FILE_CACHE[str(path)][1].encode())
    return digest.hexdigest()


def inspect_helpers(instance) -> tuple[dict, dict]:
    """Find bounded getters and literal hints, including their source evidence.

    Args:
        instance: Live plugin instance; no method will be executed.

    Returns:
        Helper metadata and indexed class method source.
    """
    methods = {}
    for cls in type(instance).__mro__:
        if not cls.__module__.startswith(type(instance).__module__.rsplit(".", 1)[0]):
            continue
        try:
            source = inspect.getsource(cls)
            tree = ast.parse(source)
            lines = source.splitlines(keepends=True)
        except (OSError, TypeError, SyntaxError, IndentationError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                methods.setdefault(
                    node.name, (node, "".join(lines[node.lineno - 1 : node.end_lineno]))
                )
    helpers = {}
    for name, (node, source) in methods.items():
        arguments = [a.arg for a in node.args.args if a.arg != "self"]
        returns = [
            n.value
            for n in ast.walk(node)
            if isinstance(n, ast.Return) and n.value is not None
        ]
        body = [
            n
            for n in node.body
            if not (
                isinstance(n, ast.Expr)
                and isinstance(n.value, ast.Constant)
                and isinstance(n.value.value, str)
            )
        ]
        if (
            not arguments
            and len(body) == 1
            and isinstance(body[0], ast.Return)
            and len(returns) == 1
            and isinstance(returns[0], ast.Constant)
            and isinstance(returns[0].value, str)
        ):
            helpers[name] = {
                "kind": "literal_hint",
                "parameters": [],
                "source": source,
                "value": returns[0].value,
            }
            continue
        if (
            not name.lstrip("_").startswith("get_")
            or len(arguments) != 1
            or arguments[0] in {"event", "context", "config"}
        ):
            continue
        queue = [name]
        seen = set()
        safe = True
        evidence = []
        while queue and safe:
            current = queue.pop()
            if current in seen:
                continue
            seen.add(current)
            if len(seen) > 8 or current not in methods:
                safe = False
                break
            child, code = methods[current]
            evidence.append(code)
            for item in ast.walk(child):
                if isinstance(
                    item,
                    (ast.Import, ast.ImportFrom, ast.Delete, ast.Global, ast.Nonlocal),
                ):
                    safe = False
                if isinstance(item, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                    targets = getattr(item, "targets", [getattr(item, "target", None)])
                    if any(
                        isinstance(t, ast.Attribute)
                        and isinstance(t.value, ast.Name)
                        and t.value.id == "self"
                        for t in targets
                    ):
                        safe = False
                if not isinstance(item, ast.Call):
                    continue
                call_name = (
                    item.func.id
                    if isinstance(item.func, ast.Name)
                    else getattr(item.func, "attr", "")
                )
                if call_name in _DENIED_CALLS or not call_name:
                    safe = False
                if (
                    isinstance(item.func, ast.Attribute)
                    and isinstance(item.func.value, ast.Name)
                    and item.func.value.id == "self"
                ):
                    if call_name not in methods:
                        safe = False
                    else:
                        queue.append(call_name)
                elif call_name not in _READ_CALLS:
                    safe = False
        if safe:
            helpers[name] = {
                "kind": "subject_lookup",
                "parameters": arguments,
                "source": "\n\n".join(evidence),
            }
    return helpers, {name: code for name, (_, code) in methods.items()}


def parameter_schema(handler) -> dict:
    """Describe handler arguments without evaluating model-supplied types.

    Args:
        handler: Registered command handler.

    Returns:
        Argument metadata for analysis and execution validation.
    """
    result = {}
    signature = inspect.signature(handler)
    params = list(signature.parameters.items())
    if params and params[0][0] == "self":
        params = params[1:]
    # The first remaining argument is the injected event, regardless of its name.
    for name, param in params[1:]:
        if param.kind in {param.VAR_POSITIONAL, param.VAR_KEYWORD}:
            continue
        annotation = str(param.annotation)
        kind = "string"
        if "list" in annotation or "List" in annotation:
            kind = "array"
        elif "dict" in annotation or "Dict" in annotation:
            kind = "object"
        elif "bool" in annotation:
            kind = "boolean"
        elif "int" in annotation:
            kind = "integer"
        elif "float" in annotation:
            kind = "number"
        required = param.default is inspect.Parameter.empty
        result[name] = {
            "type": kind,
            "annotation": annotation,
            "required": required,
            "positional_only": param.kind == param.POSITIONAL_ONLY,
        }
        if not required and isinstance(
            param.default, (str, int, float, bool, type(None))
        ):
            result[name]["default"] = param.default
    return result


async def scan(context) -> dict:
    """Collect enabled plugin commands, tool names, and source evidence.

    Args:
        context: AstrBot plugin context.

    Returns:
        Plugin records containing JSON metadata and live interface references.
    """
    descriptors = await list_commands()
    return await asyncio.to_thread(_collect, context, descriptors)


def _collect(context, descriptors: list) -> dict:
    """Inspect source off the event loop and reuse unchanged class evidence.

    Args:
        context: Context with registered plugin and tool metadata.
        descriptors: Snapshot of registered command descriptors.

    Returns:
        The current plugin capability catalog.
    """
    flat = []
    all_descriptors = {}
    pending = list(descriptors)
    while pending:
        item = pending.pop(0)
        all_descriptors[item["handler_full_name"]] = item
        pending.extend(item.get("sub_commands", []))
        if not item.get("is_group"):
            flat.append(item)
    handlers = star_handlers_registry.star_handlers_map
    tools = context.get_llm_tool_manager().func_list
    catalog = {}
    for metadata in context.get_all_stars():
        if not metadata.name or metadata.name == PLUGIN_NAME or metadata.reserved:
            continue
        module_file = getattr(metadata.module, "__file__", None)
        if not module_file:
            continue
        root = Path(module_file).resolve().parent
        digest = fingerprint(root, metadata.version or "")
        cache_key = (type(metadata.star_cls), digest)
        if cache_key not in _HELPER_CACHE:
            _HELPER_CACHE[cache_key] = (
                inspect_helpers(metadata.star_cls) if metadata.star_cls else ({}, {})
            )
        helpers, method_sources = _HELPER_CACHE[cache_key]
        entries = {}
        for desc in flat:
            if desc.get("module_path") != metadata.module_path:
                continue
            handler = handlers.get(desc["handler_full_name"])
            if not handler:
                continue
            command_filters = [
                f for f in handler.event_filters if isinstance(f, CommandFilter)
            ]
            if len(command_filters) != 1:
                continue
            try:
                source = inspect.getsource(inspect.unwrap(handler.handler))
            except (OSError, TypeError):
                source = method_sources.get(handler.handler_name, "")
            dependencies = []
            for name in method_sources:
                if f"self.{name}(" in source and name != handler.handler_name:
                    dependencies.append(method_sources[name][:1600])
            parameters = parameter_schema(handler.handler)
            accepts_message_tail = False
            if not parameters:
                signature = list(inspect.signature(handler.handler).parameters)
                event_name = next((p for p in signature if p != "self"), "event")
                try:
                    tree = ast.parse(textwrap.dedent(source))
                    accepts_message_tail = any(
                        isinstance(node, ast.Attribute)
                        and isinstance(node.value, ast.Name)
                        and node.value.id == event_name
                        and node.attr in {"message_str", "get_message_str"}
                        for node in ast.walk(tree)
                    )
                except SyntaxError:
                    pass
            entries[desc["handler_full_name"]] = {
                "id": desc["handler_full_name"],
                "command": desc.get("effective_command", ""),
                "description": desc.get("description", ""),
                "permission": desc.get("permission", "everyone"),
                "enabled": bool(
                    desc.get("enabled")
                    and desc.get("plugin_activated")
                    and not desc.get("has_conflict")
                ),
                "parameters": parameters,
                "accepts_message_tail": accepts_message_tail,
                "source": source,
                "dependencies": dependencies[:5],
                "handler": handler,
                "filter": command_filters[0],
            }
            parents = []
            parent_id = desc.get("parent_group_handler")
            seen_parents = set()
            while parent_id and parent_id not in seen_parents:
                seen_parents.add(parent_id)
                parent = handlers.get(parent_id)
                if parent:
                    parents.append(parent)
                parent_desc = all_descriptors.get(parent_id, {})
                parent_id = parent_desc.get("parent_group_handler")
            entries[desc["handler_full_name"]]["parents"] = parents
        native = []
        for tool in tools:
            origins = [
                tool.handler_module_path or "",
                type(tool).__module__,
                getattr(tool.handler, "__module__", ""),
            ]
            if tool.active and any(
                origin.startswith(metadata.module_path.rsplit(".", 1)[0] + ".")
                for origin in origins
            ):
                native.append(tool.name)
        catalog[metadata.name] = {
            "name": metadata.name,
            "display_name": metadata.display_name or metadata.name,
            "description": metadata.desc or "",
            "version": metadata.version or "",
            "active": bool(metadata.activated and metadata.star_cls),
            "root": root,
            "fingerprint": digest,
            "commands": entries,
            "helpers": helpers,
            "native_tools": sorted(native),
            "metadata": metadata,
        }
    return catalog


def analysis_batches(record: dict) -> list[tuple[dict, str]]:
    """Keep each model request small without dropping registered commands."""
    batches = []
    pending = {}
    for entry in record["commands"].values():
        if not entry["enabled"]:
            continue
        candidate = {**pending, entry["id"]: entry}
        batch_record = {**record, "commands": candidate}
        document = analysis_document(batch_record, 48000)
        if pending and (len(candidate) > 3 or len(document) > 12000):
            previous = {**record, "commands": pending}
            batches.append((previous, analysis_document(previous, 48000)))
            candidate = {entry["id"]: entry}
        pending = candidate
    if pending:
        batch_record = {**record, "commands": pending}
        batches.append((batch_record, analysis_document(batch_record, 48000)))
    return batches


def analysis_document(record: dict, limit: int) -> str:
    """Build a bounded model input from registered commands and source slices.

    Args:
        record: Scanned plugin record.
        limit: Maximum source document characters.

    Returns:
        A JSON document with executable-interface evidence.
    """
    commands = []
    entries = [e for e in record["commands"].values() if e["enabled"]]
    source_budget = max(400, (limit - 10000) // max(1, len(entries)))
    for entry in entries:
        commands.append(
            {
                k: entry[k]
                for k in ["id", "command", "description", "permission", "parameters"]
            }
            | {
                "accepts_message_tail": entry.get("accepts_message_tail", False),
                "source": entry["source"][: min(source_budget, 2600)],
                "dependencies": [s[:1000] for s in entry["dependencies"][:2]],
            }
        )
    helpers = {
        name: {**value, "source": value["source"][:4000]}
        for name, value in record["helpers"].items()
    }
    document = {
        "plugin": record["name"],
        "description": record["description"],
        "commands": commands,
        "helpers": helpers,
        "native_tools": record["native_tools"],
    }
    text = json.dumps(document, ensure_ascii=False)
    if len(text) > limit:
        for entry in commands:
            entry["dependencies"] = []
            entry["source"] = entry["source"][:700]
        text = json.dumps(document, ensure_ascii=False)
    if len(text) > limit:
        raise ValueError("插件接口证据超过单次分析上限，请提高源码字符上限后重试")
    return text
