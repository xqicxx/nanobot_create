"""LiteLLM provider implementation for multi-provider support."""

import json
import os
from typing import Any, Callable

import litellm
from litellm import acompletion
from loguru import logger

from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest
from nanobot.providers.registry import find_by_model, find_gateway


class LiteLLMProvider(LLMProvider):
    """
    LLM provider using LiteLLM for multi-provider support.
    
    Supports OpenRouter, Anthropic, OpenAI, Gemini, MiniMax, and many other providers through
    a unified interface.  Provider-specific logic is driven by the registry
    (see providers/registry.py) — no if-elif chains needed here.
    """
    
    def __init__(
        self, 
        api_key: str | None = None, 
        api_base: str | None = None,
        default_model: str = "anthropic/claude-opus-4-5",
        extra_headers: dict[str, str] | None = None,
        provider_name: str | None = None,
    ):
        super().__init__(api_key, api_base)
        self.default_model = default_model
        self.extra_headers = extra_headers or {}
        
        # Detect gateway / local deployment.
        # provider_name (from config key) is the primary signal;
        # api_key / api_base are fallback for auto-detection.
        self._gateway = find_gateway(provider_name, api_key, api_base)
        
        # Configure environment variables
        if api_key:
            self._setup_env(api_key, api_base, default_model)
        
        if api_base:
            litellm.api_base = api_base
        
        # Disable LiteLLM logging noise
        litellm.suppress_debug_info = True
        # Drop unsupported parameters for providers (e.g., gpt-5 rejects some params)
        litellm.drop_params = True
    
    def _setup_env(self, api_key: str, api_base: str | None, model: str) -> None:
        """Set environment variables based on detected provider."""
        spec = self._gateway or find_by_model(model)
        if not spec:
            return

        # Gateway/local overrides existing env; standard provider doesn't
        if self._gateway:
            os.environ[spec.env_key] = api_key
        else:
            os.environ.setdefault(spec.env_key, api_key)

        # Resolve env_extras placeholders:
        #   {api_key}  → user's API key
        #   {api_base} → user's api_base, falling back to spec.default_api_base
        effective_base = api_base or spec.default_api_base
        for env_name, env_val in spec.env_extras:
            resolved = env_val.replace("{api_key}", api_key)
            resolved = resolved.replace("{api_base}", effective_base)
            os.environ.setdefault(env_name, resolved)
    
    def _resolve_model(self, model: str) -> str:
        """Resolve model name by applying provider/gateway prefixes."""
        if self._gateway:
            # Gateway mode: apply gateway prefix, skip provider-specific prefixes
            prefix = self._gateway.litellm_prefix
            if self._gateway.strip_model_prefix:
                model = model.split("/")[-1]
            if prefix and not model.startswith(f"{prefix}/"):
                model = f"{prefix}/{model}"
            return model
        
        # Standard mode: auto-prefix for known providers
        spec = find_by_model(model)
        if spec and spec.litellm_prefix:
            if not any(model.startswith(s) for s in spec.skip_prefixes):
                model = f"{spec.litellm_prefix}/{model}"
        
        return model
    
    def _apply_model_overrides(self, model: str, kwargs: dict[str, Any]) -> None:
        """Apply model-specific parameter overrides from the registry."""
        model_lower = model.lower()
        spec = find_by_model(model)
        if spec:
            for pattern, overrides in spec.model_overrides:
                if pattern in model_lower:
                    kwargs.update(overrides)
                    return

    def _infer_provider_spec(self, model: str):
        spec = find_by_model(model)
        if spec:
            return spec
        if "/" in model:
            suffix = model.split("/", 1)[1]
            return find_by_model(suffix)
        return None

    @staticmethod
    def _strip_native_prefix(model: str) -> str:
        for prefix in ("openai/", "stepfun/", "minimax/"):
            if model.lower().startswith(prefix):
                return model[len(prefix):]
        return model

    async def _chat_native_stream(
        self,
        *,
        requested_model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        max_tokens: int,
        temperature: float,
        on_token: Callable[[str], None] | None,
    ) -> LLMResponse | None:
        """
        Stream with provider-native OpenAI-compatible SDK for StepFun/MiniMax.
        Falls back to LiteLLM when unavailable.
        """
        if self._gateway:
            return None

        spec = self._infer_provider_spec(requested_model)
        if not spec or spec.name not in {"stepfun", "minimax"}:
            return None

        api_key = self.api_key or os.getenv(spec.env_key)
        base_url = self.api_base or spec.default_api_base
        if not api_key or not base_url:
            return None

        try:
            from openai import AsyncOpenAI
        except Exception:
            return None

        model_name = self._strip_native_prefix(requested_model)
        kwargs: dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers=self.extra_headers or None,
        )
        try:
            stream = await client.chat.completions.create(**kwargs)
            content_parts: list[str] = []
            tool_call_map: dict[int, dict[str, Any]] = {}
            finish_reason = "stop"

            async for chunk in stream:
                choice = chunk.choices[0] if chunk.choices else None
                if choice and getattr(choice, "finish_reason", None):
                    finish_reason = choice.finish_reason or finish_reason
                delta = getattr(choice, "delta", None) if choice else None
                if delta is None:
                    continue

                delta_content = getattr(delta, "content", None)
                if delta_content:
                    content_parts.append(delta_content)
                    if on_token:
                        on_token(delta_content)

                delta_tool_calls = getattr(delta, "tool_calls", None)
                if delta_tool_calls:
                    for tc in delta_tool_calls:
                        idx = getattr(tc, "index", 0) or 0
                        entry = tool_call_map.setdefault(
                            idx,
                            {"id": None, "name": None, "arguments": ""},
                        )
                        if getattr(tc, "id", None):
                            entry["id"] = tc.id
                        fn = getattr(tc, "function", None)
                        if fn is not None:
                            if getattr(fn, "name", None):
                                entry["name"] = fn.name
                            if getattr(fn, "arguments", None):
                                entry["arguments"] += fn.arguments

            tool_calls: list[ToolCallRequest] = []
            if tool_call_map:
                for idx in sorted(tool_call_map.keys()):
                    entry = tool_call_map[idx]
                    args = entry.get("arguments", "")
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {"raw": args}
                    tool_calls.append(
                        ToolCallRequest(
                            id=entry.get("id") or f"tool_{idx}",
                            name=entry.get("name") or "tool",
                            arguments=args,
                        )
                    )

            return LLMResponse(
                content="".join(content_parts) if content_parts else None,
                tool_calls=tool_calls,
                finish_reason=finish_reason,
            )
        except Exception as exc:
            logger.warning(
                "Native stream failed for provider={} model={}: {}. Fallback to LiteLLM stream.",
                spec.name,
                requested_model,
                exc,
            )
            return None
        finally:
            close_fn = getattr(client, "close", None)
            if close_fn:
                try:
                    maybe_awaitable = close_fn()
                    if hasattr(maybe_awaitable, "__await__"):
                        await maybe_awaitable
                except Exception:
                    pass
    
    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        stream: bool = False,
        on_token: Callable[[str], None] | None = None,
    ) -> LLMResponse:
        """
        Send a chat completion request via LiteLLM.
        
        Args:
            messages: List of message dicts with 'role' and 'content'.
            tools: Optional list of tool definitions in OpenAI format.
            model: Model identifier (e.g., 'anthropic/claude-sonnet-4-5').
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature.
        
        Returns:
            LLMResponse with content and/or tool calls.
        """
        requested_model = model or self.default_model
        model = self._resolve_model(requested_model)
        
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        
        # Apply model-specific overrides (e.g. kimi-k2.5 temperature)
        self._apply_model_overrides(model, kwargs)
        
        # Pass api_key directly — more reliable than env vars alone
        if self.api_key:
            kwargs["api_key"] = self.api_key
        
        # Pass api_base for custom endpoints
        if self.api_base:
            kwargs["api_base"] = self.api_base
        
        # Pass extra headers (e.g. APP-Code for AiHubMix)
        if self.extra_headers:
            kwargs["extra_headers"] = self.extra_headers
        
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        
        try:
            if stream:
                native = await self._chat_native_stream(
                    requested_model=requested_model,
                    messages=messages,
                    tools=tools,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    on_token=on_token,
                )
                if native is not None:
                    return native
                kwargs["stream"] = True
                response = await acompletion(**kwargs)
                content_parts: list[str] = []
                tool_call_map: dict[int, dict[str, Any]] = {}

                async for chunk in response:
                    choice = chunk.choices[0] if chunk.choices else None
                    delta = getattr(choice, "delta", None) if choice else None
                    if delta is None:
                        continue

                    delta_content = getattr(delta, "content", None)
                    if delta_content:
                        content_parts.append(delta_content)
                        if on_token:
                            on_token(delta_content)

                    delta_tool_calls = getattr(delta, "tool_calls", None)
                    if delta_tool_calls:
                        for tc in delta_tool_calls:
                            idx = getattr(tc, "index", 0) or 0
                            entry = tool_call_map.setdefault(
                                idx,
                                {"id": None, "name": None, "arguments": ""},
                            )
                            if getattr(tc, "id", None):
                                entry["id"] = tc.id
                            fn = getattr(tc, "function", None)
                            if fn is not None:
                                if getattr(fn, "name", None):
                                    entry["name"] = fn.name
                                if getattr(fn, "arguments", None):
                                    entry["arguments"] += fn.arguments

                tool_calls: list[ToolCallRequest] = []
                if tool_call_map:
                    for idx in sorted(tool_call_map.keys()):
                        entry = tool_call_map[idx]
                        args = entry.get("arguments", "")
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except json.JSONDecodeError:
                                args = {"raw": args}
                        tool_calls.append(
                            ToolCallRequest(
                                id=entry.get("id") or f"tool_{idx}",
                                name=entry.get("name") or "tool",
                                arguments=args,
                            )
                        )

                return LLMResponse(
                    content="".join(content_parts) if content_parts else None,
                    tool_calls=tool_calls,
                    finish_reason="stop",
                )

            response = await acompletion(**kwargs)
            return self._parse_response(response)
        except Exception as e:
            # Return error as content for graceful handling
            return LLMResponse(
                content=f"Error calling LLM: {str(e)}",
                finish_reason="error",
            )
    
    def _parse_response(self, response: Any) -> LLMResponse:
        """Parse LiteLLM response into our standard format."""
        choice = response.choices[0]
        message = choice.message
        
        tool_calls = []
        if hasattr(message, "tool_calls") and message.tool_calls:
            for tc in message.tool_calls:
                # Parse arguments from JSON string if needed
                args = tc.function.arguments
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"raw": args}
                
                tool_calls.append(ToolCallRequest(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=args,
                ))
        
        usage = {}
        if hasattr(response, "usage") and response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        
        reasoning_content = getattr(message, "reasoning_content", None)
        
        return LLMResponse(
            content=message.content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "stop",
            usage=usage,
            reasoning_content=reasoning_content,
        )
    
    def get_default_model(self) -> str:
        """Get the default model."""
        return self.default_model
