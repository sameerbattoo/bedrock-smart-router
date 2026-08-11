# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bidirectional format translation between Bedrock Converse and OpenAI Chat Completions.

Handles the conversion of messages, system prompts, tool configurations,
inference parameters, and responses between the two API formats.

This module enables the router to:
1. Accept Chat Completions requests and route to Converse-only models
2. Accept Converse requests and route to Chat Completions-only (Mantle) models

Both directions are stateless message-in/message-out, making translation lossless
for the common case (text messages + tool calls).
"""

from __future__ import annotations

import base64
import json
import uuid
from typing import Any


# ═══════════════════════════════════════════════════════════════
# Converse → Chat Completions (for calling Mantle from Converse input)
# ═══════════════════════════════════════════════════════════════

def converse_to_chat_completions(
    messages: list[dict[str, Any]],
    system: list[dict[str, Any]] | None = None,
    tool_config: dict[str, Any] | None = None,
    inference_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert Bedrock Converse request params to Chat Completions request body.

    Args:
        messages: Converse messages (role + content blocks)
        system: Converse system prompt blocks
        tool_config: Converse tool configuration
        inference_config: Converse inference config (maxTokens, temperature, etc.)

    Returns:
        Dict ready to be sent as JSON body to /v1/chat/completions
    """
    cc_messages: list[dict[str, Any]] = []

    # System prompt → first message with role=system
    if system:
        system_text = " ".join(
            block.get("text", "") for block in system if isinstance(block, dict) and "text" in block
        )
        if system_text.strip():
            cc_messages.append({"role": "system", "content": system_text.strip()})

    # Convert each Converse message
    for msg in messages:
        role = msg.get("role", "user")
        content_blocks = msg.get("content", [])

        if role == "assistant":
            cc_msg = _converse_assistant_to_cc(content_blocks)
        elif role == "user":
            cc_msg = _converse_user_to_cc(content_blocks)
        else:
            # Pass through unknown roles
            cc_msg = {"role": role, "content": _extract_text_from_blocks(content_blocks)}

        if cc_msg:
            # _converse_user_to_cc may return a list (multiple tool results)
            if isinstance(cc_msg, list):
                cc_messages.extend(cc_msg)
            else:
                cc_messages.append(cc_msg)

    # Build request body
    body: dict[str, Any] = {"messages": cc_messages}

    # Inference config
    if inference_config:
        if "maxTokens" in inference_config:
            body["max_tokens"] = inference_config["maxTokens"]
        if "temperature" in inference_config:
            body["temperature"] = inference_config["temperature"]
        if "topP" in inference_config:
            body["top_p"] = inference_config["topP"]
        if "stopSequences" in inference_config:
            body["stop"] = inference_config["stopSequences"]

    # Tool config → tools
    if tool_config:
        tools = tool_config.get("tools", [])
        if tools:
            body["tools"] = [_converse_tool_to_cc(t) for t in tools]

    return body


def _converse_assistant_to_cc(content_blocks: list[dict]) -> dict[str, Any]:
    """Convert assistant content blocks to Chat Completions format."""
    text_parts: list[str] = []
    tool_calls: list[dict] = []

    for block in content_blocks:
        if "text" in block:
            text_parts.append(block["text"])
        elif "toolUse" in block:
            tool_calls.append(_tool_use_to_function_call(block["toolUse"]))
        # Skip reasoningContent — not translatable to Chat Completions

    msg: dict[str, Any] = {"role": "assistant"}
    if text_parts:
        msg["content"] = "\n".join(text_parts)
    else:
        msg["content"] = None  # Required when tool_calls present
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


def _converse_user_to_cc(content_blocks: list[dict]) -> dict[str, Any] | list[dict[str, Any]]:
    """Convert user content blocks to Chat Completions format.

    Handles text, images, and toolResult blocks.
    Returns a single message dict, or a list of messages when multiple
    toolResult blocks are present (CC requires separate role=tool messages).
    """
    # Check if this is a tool result message
    tool_results = [b for b in content_blocks if "toolResult" in b]
    if tool_results:
        # Chat Completions uses separate messages with role=tool for each result
        messages = []
        for b in tool_results:
            tr = b["toolResult"]
            content_text = ""
            for c in tr.get("content", []):
                if "text" in c:
                    content_text += c["text"]
                elif "json" in c:
                    content_text += _to_json_string(c["json"])
            messages.append({
                "role": "tool",
                "tool_call_id": tr.get("toolUseId", ""),
                "content": content_text,
            })
        return messages if len(messages) > 1 else messages[0]

    # Regular user message
    text_parts = [b["text"] for b in content_blocks if "text" in b]
    image_blocks = [b for b in content_blocks if "image" in b]

    if image_blocks and text_parts:
        # Multimodal: use content array format
        content: list[dict] = []
        for t in text_parts:
            content.append({"type": "text", "text": t})
        for img_block in image_blocks:
            img = img_block["image"]
            source = img.get("source", {})
            if "bytes" in source:
                b64 = base64.b64encode(source["bytes"]).decode("utf-8")
                media_type = img.get("format", "png")
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/{media_type};base64,{b64}"},
                })
        return {"role": "user", "content": content}

    # Text only
    return {"role": "user", "content": "\n".join(text_parts) if text_parts else ""}


def _converse_tool_to_cc(tool: dict) -> dict[str, Any]:
    """Convert a Converse toolSpec to Chat Completions function tool."""
    spec = tool.get("toolSpec", {})
    return {
        "type": "function",
        "function": {
            "name": spec.get("name", ""),
            "description": spec.get("description", ""),
            "parameters": spec.get("inputSchema", {}).get("json", {}),
        },
    }


# ═══════════════════════════════════════════════════════════════
# Chat Completions → Converse (for calling bedrock-runtime from CC input)
# ═══════════════════════════════════════════════════════════════

def chat_completions_to_converse(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    stop: list[str] | str | None = None,
) -> dict[str, Any]:
    """Convert Chat Completions request params to Converse API params.

    Args:
        messages: Chat Completions messages (role + content)
        tools: Chat Completions tool definitions
        max_tokens: Maximum output tokens
        temperature: Sampling temperature
        top_p: Top-p sampling
        stop: Stop sequences

    Returns:
        Dict with keys: messages, system, tool_config, inference_config
        ready to be passed to bedrock.converse()
    """
    converse_messages: list[dict[str, Any]] = []
    system_blocks: list[dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content")

        if role == "system":
            # System messages → system blocks
            text = content if isinstance(content, str) else _cc_content_to_text(content)
            if text:
                system_blocks.append({"text": text})

        elif role == "assistant":
            converse_msg = _cc_assistant_to_converse(msg)
            if converse_msg:
                converse_messages.append(converse_msg)

        elif role == "tool":
            # Tool result → user message with toolResult block
            converse_messages.append({
                "role": "user",
                "content": [{
                    "toolResult": {
                        "toolUseId": msg.get("tool_call_id", ""),
                        "content": [{"text": str(content) if content else ""}],
                        "status": "success",
                    }
                }],
            })

        elif role == "user":
            converse_messages.append(_cc_user_to_converse(msg))

    # Build result
    result: dict[str, Any] = {"messages": converse_messages}

    if system_blocks:
        result["system"] = system_blocks

    # Tool config
    if tools:
        converse_tools = []
        for tool in tools:
            if tool.get("type") == "function":
                fn = tool.get("function", {})
                converse_tools.append({
                    "toolSpec": {
                        "name": fn.get("name", ""),
                        "description": fn.get("description", ""),
                        "inputSchema": {"json": fn.get("parameters", {})},
                    }
                })
        if converse_tools:
            result["tool_config"] = {"tools": converse_tools}

    # Inference config
    inference_config: dict[str, Any] = {}
    if max_tokens is not None:
        inference_config["maxTokens"] = max_tokens
    if temperature is not None:
        inference_config["temperature"] = temperature
    if top_p is not None:
        inference_config["topP"] = top_p
    if stop:
        inference_config["stopSequences"] = [stop] if isinstance(stop, str) else stop
    if inference_config:
        result["inference_config"] = inference_config

    return result


def _cc_assistant_to_converse(msg: dict) -> dict[str, Any] | None:
    """Convert CC assistant message to Converse format."""
    content_blocks: list[dict] = []
    content = msg.get("content")

    if content:
        text = content if isinstance(content, str) else _cc_content_to_text(content)
        if text:
            content_blocks.append({"text": text})

    # Tool calls
    for tc in msg.get("tool_calls", []):
        if tc.get("type") == "function":
            content_blocks.append(_function_call_to_tool_use(tc))

    if not content_blocks:
        return None

    return {"role": "assistant", "content": content_blocks}


def _cc_user_to_converse(msg: dict) -> dict[str, Any]:
    """Convert CC user message to Converse format."""
    content = msg.get("content")

    if isinstance(content, str):
        return {"role": "user", "content": [{"text": content}]}

    if isinstance(content, list):
        blocks: list[dict] = []
        for part in content:
            if part.get("type") == "text":
                blocks.append({"text": part.get("text", "")})
            elif part.get("type") == "image_url":
                url = part.get("image_url", {}).get("url", "")
                if url.startswith("data:"):
                    # Parse data URI safely
                    try:
                        # data:image/png;base64,xxxxx
                        header, b64data = url.split(",", 1)
                        media_type = header.split(":")[1].split(";")[0]  # image/png
                        fmt = media_type.split("/")[1]  # png
                        blocks.append({
                            "image": {
                                "format": fmt,
                                "source": {"bytes": base64.b64decode(b64data)},
                            }
                        })
                    except (ValueError, IndexError, KeyError):
                        # Malformed data URI — skip this image
                        pass
        return {"role": "user", "content": blocks if blocks else [{"text": ""}]}

    return {"role": "user", "content": [{"text": str(content) if content else ""}]}


# ═══════════════════════════════════════════════════════════════
# Response translation
# ═══════════════════════════════════════════════════════════════

def chat_completions_response_to_converse(cc_response: dict[str, Any]) -> dict[str, Any]:
    """Convert a Chat Completions response to Converse response format.

    Maps the CC response structure to Bedrock Converse output format so the
    router can return a consistent response regardless of backend.
    """
    choices = cc_response.get("choices", [])
    if not choices:
        return {"output": {"message": {"role": "assistant", "content": [{"text": ""}]}}}

    choice = choices[0]
    message = choice.get("message", {})
    content_blocks: list[dict] = []

    # Text content
    text = message.get("content")
    if text:
        content_blocks.append({"text": text})
    elif not text and message.get("reasoning"):
        # Some models (e.g., gpt-oss-safeguard) put their response in 'reasoning'
        # when content is null/empty. Use reasoning as fallback text.
        content_blocks.append({"text": message["reasoning"]})

    # Tool calls
    for tc in message.get("tool_calls", []):
        if tc.get("type") == "function":
            content_blocks.append(_function_call_to_tool_use(tc))

    # Map stop reason
    finish_reason = choice.get("finish_reason", "stop")
    stop_reason_map = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "content_filter": "content_filtered",
    }
    stop_reason = stop_reason_map.get(finish_reason, "end_turn")

    # Usage
    cc_usage = cc_response.get("usage", {})
    usage = {
        "inputTokens": cc_usage.get("prompt_tokens", 0),
        "outputTokens": cc_usage.get("completion_tokens", 0),
        "totalTokens": cc_usage.get("total_tokens", 0),
    }

    return {
        "output": {"message": {"role": "assistant", "content": content_blocks}},
        "stopReason": stop_reason,
        "usage": usage,
    }


def converse_response_to_chat_completions(
    converse_response: dict[str, Any],
    model: str = "",
) -> dict[str, Any]:
    """Convert a Converse response to Chat Completions response format.

    Maps Bedrock Converse output to standard OpenAI Chat Completions response.
    """
    output = converse_response.get("output", {})
    message = output.get("message", {})
    content_blocks = message.get("content", [])

    # Extract text and tool calls
    text_parts: list[str] = []
    tool_calls: list[dict] = []

    for block in content_blocks:
        if "text" in block:
            text_parts.append(block["text"])
        elif "toolUse" in block:
            tool_calls.append(_tool_use_to_function_call(block["toolUse"]))

    # Build CC message
    cc_message: dict[str, Any] = {"role": "assistant"}
    if text_parts:
        cc_message["content"] = "\n".join(text_parts)
    else:
        cc_message["content"] = None
    if tool_calls:
        cc_message["tool_calls"] = tool_calls

    # Map stop reason
    stop_reason = converse_response.get("stopReason", "end_turn")
    finish_reason_map = {
        "end_turn": "stop",
        "max_tokens": "length",
        "tool_use": "tool_calls",
        "content_filtered": "content_filter",
        "stop_sequence": "stop",
    }
    finish_reason = finish_reason_map.get(stop_reason, "stop")

    # Usage
    converse_usage = converse_response.get("usage", {})
    usage = {
        "prompt_tokens": converse_usage.get("inputTokens", 0),
        "completion_tokens": converse_usage.get("outputTokens", 0),
        "total_tokens": converse_usage.get("inputTokens", 0) + converse_usage.get("outputTokens", 0),
    }

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "model": model,
        "choices": [{
            "index": 0,
            "message": cc_message,
            "finish_reason": finish_reason,
        }],
        "usage": usage,
    }


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def _tool_use_to_function_call(tu: dict) -> dict:
    """Convert a Converse toolUse block to Chat Completions tool_call."""
    return {
        "id": tu.get("toolUseId", f"call_{uuid.uuid4().hex[:8]}"),
        "type": "function",
        "function": {
            "name": tu["name"],
            "arguments": _to_json_string(tu.get("input", {})),
        },
    }


def _function_call_to_tool_use(tc: dict) -> dict:
    """Convert a Chat Completions tool_call to Converse toolUse block."""
    fn = tc.get("function", {})
    return {
        "toolUse": {
            "toolUseId": tc.get("id", f"call_{uuid.uuid4().hex[:8]}"),
            "name": fn.get("name", ""),
            "input": _parse_json_string(fn.get("arguments", "{}")),
        }
    }


def _extract_text_from_blocks(blocks: list[dict]) -> str:
    """Extract text from Converse content blocks."""
    parts = [b["text"] for b in blocks if isinstance(b, dict) and "text" in b]
    return "\n".join(parts)


def _cc_content_to_text(content: Any) -> str:
    """Extract text from Chat Completions content (string or array)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [p.get("text", "") for p in content if p.get("type") == "text"]
        return "\n".join(parts)
    return str(content) if content else ""


def _to_json_string(obj: Any) -> str:
    """Convert a dict/object to JSON string."""
    if isinstance(obj, str):
        return obj
    return json.dumps(obj)


def _parse_json_string(s: str) -> Any:
    """Parse a JSON string to dict, returning empty dict on failure."""
    if not s:
        return {}
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return {"raw": s}
