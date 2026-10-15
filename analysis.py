"""Analysis transport and progress without publishing model reasoning or source."""

import hashlib
import math
import time
from types import SimpleNamespace

from astrbot.core.provider.provider import Provider

from .prompts import ANALYZE_PROMPT

OUTPUT_TOKENS = 6000


def estimated_tokens(text: str) -> int:
    """Approximate text tokens; never treat this as provider billing data."""
    non_ascii = sum(ord(char) > 127 for char in text)
    return math.ceil((len(text) - non_ascii) / 4 + non_ascii)


def checkpoint_key(document: str) -> str:
    """Invalidate completed batches when either evidence or instructions change."""
    return hashlib.sha256((ANALYZE_PROMPT + document).encode()).hexdigest()


async def generate(context, provider_id: str, document: str, item: dict):
    """Use the provider stream where implemented; never retry a failed request."""
    getter = getattr(context, "get_provider_by_id", None)
    provider = getter(provider_id) if getter else None
    stream_method = getattr(type(provider), "text_chat_stream", None)
    options = {
        "prompt": document,
        "system_prompt": ANALYZE_PROMPT,
        "max_tokens": OUTPUT_TOKENS,
        "temperature": 0.1,
        # Let the dashboard user decide whether to retry failed analysis calls.
        "request_max_retries": 1,
    }
    if stream_method and stream_method is not Provider.text_chat_stream:
        stream = provider.text_chat_stream(**options)
        chunks, usage = [], None
        try:
            async for response in stream:
                item["last_response_at"] = int(time.time())
                text = response.completion_text or ""
                reasoning = response.reasoning_content or ""
                item["phase"] = "receiving" if text else "thinking"
                if response.is_chunk:
                    item["received_chars"] += len(text)
                    item["reasoning_chars"] += len(reasoning)
                elif text:
                    item["received_chars"] = len(text)
                if getattr(response, "usage", None):
                    usage = response.usage
                if not response.is_chunk:
                    return response
                chunks.append(text)
                if item["received_chars"] > 150000:
                    raise ValueError("模型响应超过上限")
        finally:
            await stream.aclose()
        if not chunks:
            raise ValueError("模型未返回适配内容")
        return SimpleNamespace(completion_text="".join(chunks), usage=usage)
    return await context.llm_generate(chat_provider_id=provider_id, **options)
