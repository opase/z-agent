"""MCP JSON Schema 裁剪/规范化 — 将 MCP 工具 Schema 转为 LangChain 兼容格式"""

import copy
import logging

logger = logging.getLogger(__name__)

# JSON Schema 关键字中在 LangChain tool binding 中不支持或无需的
_UNSUPPORTED_KEYWORDS = {"default", "examples", "const", "enum", "pattern",
                         "minLength", "maxLength", "minimum", "maximum",
                         "exclusiveMinimum", "exclusiveMaximum", "multipleOf"}


def normalize_tool_schema(raw_schema: dict) -> dict:
    """将 MCP 原始 JSON Schema 裁剪为 LangChain bind_tools 兼容格式

    处理步骤:
    1. 展开 $ref 引用
    2. 移除 $defs 定义区
    3. 展平 oneOf/anyOf（取第一个选项）
    4. 移除不支持的 JSON Schema 关键字
    5. 确保顶级 type 为 "object"

    Args:
        raw_schema: MCP 工具返回的原始 inputSchema

    Returns:
        规范化后的简化 Schema
    """
    schema = copy.deepcopy(raw_schema)

    # 步骤 1+2: 解析 $defs 和 $ref
    _resolve_refs(schema, schema)

    # 步骤 3: 展平 oneOf / anyOf
    schema = _flatten_alternatives(schema)

    # 步骤 4: 移除不支持的关键字
    schema = _strip_unsupported(schema)

    # 步骤 5: 确保顶级 type 是 object
    if schema.get("type") != "object":
        logger.warning("MCP 工具 Schema 顶级 type 不是 object，已强制修正")
        schema["type"] = "object"

    # 保证有 properties 字段
    if "properties" not in schema:
        schema["properties"] = {}

    return schema


def _resolve_refs(node: dict, root: dict, _visited: set = None) -> dict:
    """递归展开 JSON Schema 中的 $ref 引用"""
    if _visited is None:
        _visited = set()

    if not isinstance(node, dict):
        return node

    ref = node.get("$ref", "")
    if ref and ref.startswith("#/"):
        if ref in _visited:
            logger.warning("检测到循环 $ref: %s，跳过", ref)
            return node
        _visited.add(ref)
        path_parts = ref[2:].split("/")
        resolved = root
        for part in path_parts:
            if isinstance(resolved, dict):
                resolved = resolved.get(part)
            else:
                resolved = None
                break
        if resolved and isinstance(resolved, dict):
            # 用 resolved 内容替换当前节点（保留非 $ref 字段）
            resolved_copy = copy.deepcopy(resolved)
            for k, v in node.items():
                if k != "$ref":
                    resolved_copy[k] = v
            return _resolve_refs(resolved_copy, root, _visited)

    # 递归处理子节点
    result = {}
    for k, v in node.items():
        if k == "$ref":
            continue
        if isinstance(v, dict):
            result[k] = _resolve_refs(v, root, _visited)
        elif isinstance(v, list):
            result[k] = [_resolve_refs(item, root, _visited) if isinstance(item, dict) else item for item in v]
        else:
            result[k] = v
    return result


def _flatten_alternatives(node: dict) -> dict:
    """展平 oneOf/anyOf——取第一个可作为 object schema 的选项"""
    if not isinstance(node, dict):
        return node

    result = {}
    for key, value in node.items():
        if key in ("oneOf", "anyOf") and isinstance(value, list) and len(value) > 0:
            # 取第一个选项
            chosen = value[0]
            if isinstance(chosen, dict):
                # 将选定项的属性合并到父级
                for ck, cv in chosen.items():
                    if ck == "type":
                        result.setdefault("type", cv)
                    elif ck == "properties":
                        result.setdefault("properties", cv)
                    elif ck == "required":
                        result.setdefault("required", cv)
                    elif ck == "description":
                        result.setdefault("description", cv)
                logger.info("展平 %s: 取第一个选项", key)
            continue
        elif isinstance(value, dict):
            result[key] = _flatten_alternatives(value)
        elif isinstance(value, list):
            result[key] = [_flatten_alternatives(item) if isinstance(item, dict) else item for item in value]
        else:
            result[key] = value

    return result


def _strip_unsupported(node: dict) -> dict:
    """递归移除 LangChain 不支持的 JSON Schema 关键字"""
    if not isinstance(node, dict):
        return node

    result = {}
    for key, value in node.items():
        if key in _UNSUPPORTED_KEYWORDS:
            continue
        if isinstance(value, dict):
            result[key] = _strip_unsupported(value)
        elif isinstance(value, list):
            result[key] = [
                _strip_unsupported(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            result[key] = value
    return result
