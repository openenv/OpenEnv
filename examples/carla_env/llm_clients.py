"""
Unified LLM client interface for all providers.

Supports:
- Anthropic (Claude)
- OpenAI (GPT)
- Qwen (via OpenAI-compatible API)
- Hugging Face (Open models via Inference API)
- Local (Hugging Face models loaded with transformers)
"""

import os
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from anthropic import Anthropic
from openai import OpenAI

def build_vision_message(prompt: str, image_base64: str, provider: str) -> Dict[str, Any]:
    """Build a user message with image + text, formatted for the provider.

    Args:
        prompt: Text prompt to include with the image.
        image_base64: Base64-encoded JPEG image data.
        provider: LLM provider ("anthropic", "openai", "qwen", "huggingface").

    Returns:
        Message dict with multimodal content array.
    """
    if provider == "anthropic":
        return {
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_base64}},
                {"type": "text", "text": prompt},
            ],
        }
    else:  # openai, qwen, huggingface — all OpenAI-compatible
        return {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                {"type": "text", "text": prompt},
            ],
        }


class LLMClient(ABC):
    """Base class for LLM clients."""

    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        max_tokens: int = 2048
    ) -> Dict[str, Any]:
        """
        Send chat request with tool calling support.

        Returns:
            {
                "tool_calls": [{"name": str, "arguments": dict}],
                "text": str  # Any text content
            }
        """
        pass

class ClaudeClient(LLMClient):
    """Anthropic Claude client."""

    def __init__(self, model_id: str):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not set. "
                "Set it with: export ANTHROPIC_API_KEY=your-key"
            )
        self.client = Anthropic(api_key=api_key)
        self.model_id = model_id

    def chat(self, messages, tools, max_tokens=2048):
        # Convert OpenAI-style tools to Anthropic format
        anthropic_tools = [
            {
                "name": tool["function"]["name"],
                "description": tool["function"]["description"],
                "input_schema": tool["function"]["parameters"]
            }
            for tool in tools
        ]

        response = self.client.messages.create(
            model=self.model_id,
            max_tokens=max_tokens,
            tools=anthropic_tools,
            messages=messages
        )

        # Normalize response format
        tool_calls = []
        text_content = []

        for block in response.content:
            if block.type == "tool_use":
                tool_calls.append({
                    "name": block.name,
                    "arguments": block.input
                })
            elif block.type == "text":
                text_content.append(block.text)

        return {
            "tool_calls": tool_calls,
            "text": " ".join(text_content)
        }

class OpenAIClient(LLMClient):
    """OpenAI client."""

    def __init__(self, model_id: str):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not set. "
                "Set it with: export OPENAI_API_KEY=your-key"
            )
        self.client = OpenAI(api_key=api_key)
        self.model_id = model_id

    def chat(self, messages, tools, max_tokens=2048):
        response = self.client.chat.completions.create(
            model=self.model_id,
            max_tokens=max_tokens,
            tools=tools,
            messages=messages
        )

        message = response.choices[0].message

        tool_calls = []
        if message.tool_calls:
            for tc in message.tool_calls:
                try:
                    # Parse JSON arguments
                    import json
                    arguments = json.loads(tc.function.arguments)
                except:
                    arguments = {}

                tool_calls.append({
                    "name": tc.function.name,
                    "arguments": arguments
                })

        return {
            "tool_calls": tool_calls,
            "text": message.content or ""
        }

class QwenClient(LLMClient):
    """Qwen client (via OpenAI-compatible API)."""

    def __init__(self, model_id: str):
        api_key = os.getenv("QWEN_API_KEY")
        if not api_key:
            raise ValueError(
                "QWEN_API_KEY not set. "
                "Set it with: export QWEN_API_KEY=your-key"
            )
        # Qwen uses OpenAI-compatible API
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.model_id = model_id

    def chat(self, messages, tools, max_tokens=2048):
        # Same as OpenAI
        response = self.client.chat.completions.create(
            model=self.model_id,
            max_tokens=max_tokens,
            tools=tools,
            messages=messages
        )

        message = response.choices[0].message

        tool_calls = []
        if message.tool_calls:
            for tc in message.tool_calls:
                try:
                    import json
                    arguments = json.loads(tc.function.arguments)
                except:
                    arguments = {}

                tool_calls.append({
                    "name": tc.function.name,
                    "arguments": arguments
                })

        return {
            "tool_calls": tool_calls,
            "text": message.content or ""
        }

class HuggingFaceClient(LLMClient):
    """Hugging Face Inference API client (OpenAI-compatible)."""

    def __init__(self, model_id: str):
        api_key = os.getenv("HF_TOKEN")
        if not api_key:
            raise ValueError(
                "HF_TOKEN not set. "
                "Set it with: export HF_TOKEN=your-token\n"
                "Get token from: https://huggingface.co/settings/tokens"
            )
        # Hugging Face Inference Providers (OpenAI-compatible)
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://router.huggingface.co/v1/"
        )
        self.model_id = model_id

    def chat(self, messages, tools, max_tokens=2048):
        import json
        import re

        # Qwen3 defaults to thinking mode which burns all tokens on hidden
        # reasoning. Disable it by prepending /no_think to the last message
        # and increasing max_tokens as a safety margin.
        if "qwen3" in self.model_id.lower():
            messages = [*messages]  # shallow copy
            last = messages[-1]
            if isinstance(last.get("content"), str):
                messages[-1] = {**last, "content": "/no_think\n\n" + last["content"]}
            max_tokens = max(max_tokens, 512)

        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                max_tokens=max_tokens,
                tools=tools,
                messages=messages,
            )
        except Exception as e:
            # Some models generate malformed tool calls like
            # <function=name={"arg": val}</function> — parse them manually
            failed = self._parse_failed_generation(e)
            if failed:
                return failed
            raise

        message = response.choices[0].message

        tool_calls = []
        if message.tool_calls:
            for tc in message.tool_calls:
                try:
                    arguments = json.loads(tc.function.arguments)
                except Exception:
                    arguments = {}

                tool_calls.append({
                    "name": tc.function.name,
                    "arguments": arguments
                })

        return {
            "tool_calls": tool_calls,
            "text": message.content or ""
        }

    @staticmethod
    def _parse_failed_generation(error: Exception) -> dict | None:
        """Extract tool call from HF's failed_generation error response.

        Some open models emit malformed syntax like:
            <function=control_vehicle={"throttle": 0.5, "steer": -0.5}</function>
        HF returns this in the error body. We parse it as a fallback.
        """
        import json
        import re

        error_str = str(error)
        if "tool_use_failed" not in error_str and "failed_generation" not in error_str:
            return None

        # Match <function=name={"key": val}</function> or <function=name>{"key": val}</function>
        pattern = r'<function=(\w+)[=>](\{.*?\})\s*</function>'
        match = re.search(pattern, error_str)
        if not match:
            return None

        name = match.group(1)
        try:
            arguments = json.loads(match.group(2))
        except json.JSONDecodeError:
            return None

        print(f"  [HF fallback] Recovered tool call: {name}({arguments})")
        return {
            "tool_calls": [{"name": name, "arguments": arguments}],
            "text": ""
        }

class LocalClient(LLMClient):
    """Local model client using transformers (AutoModelForCausalLM + AutoTokenizer).

    Loads a Hugging Face model locally on GPU and generates responses with tool calling
    support. Uses the tokenizer's built-in chat template (e.g., Qwen3 has native tool
    support via <tool_call> tags).
    """

    def __init__(self, model_id: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"  [LocalClient] Loading model: {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model_id = model_id
        print(f"  [LocalClient] Model loaded on {self.model.device}")

    def chat(self, messages, tools, max_tokens=2048):
        import json
        import re

        # Convert OpenAI-style tools to the format expected by the chat template
        tool_defs = []
        for tool in tools:
            func = tool["function"]
            tool_defs.append({
                "type": "function",
                "function": {
                    "name": func["name"],
                    "description": func["description"],
                    "parameters": func["parameters"],
                },
            })

        # Apply chat template with tools
        text = self.tokenizer.apply_chat_template(
            messages,
            tools=tool_defs,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
        )

        # Decode only the generated tokens (skip the prompt).
        # NOTE: skip_special_tokens=False is required because Qwen3 uses <tool_call>
        # and </tool_call> as special tokens. With True, they get stripped and we
        # can't parse tool calls. We manually strip EOS/padding tokens instead.
        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        response_text = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
        # Remove EOS and padding tokens that may appear in the raw decoded text
        for special_tok in [self.tokenizer.eos_token, self.tokenizer.pad_token, "<|endoftext|>", "<|im_end|>"]:
            if special_tok:
                response_text = response_text.replace(special_tok, "")

        # Parse tool calls from <tool_call>...</tool_call> tags (Qwen3 format)
        tool_calls = []
        text_parts = []

        tool_call_pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
        matches = re.findall(tool_call_pattern, response_text, re.DOTALL)

        if matches:
            for match in matches:
                try:
                    parsed = json.loads(match)
                    tool_calls.append({
                        "name": parsed.get("name", ""),
                        "arguments": parsed.get("arguments", {}),
                    })
                except json.JSONDecodeError:
                    continue

            # Extract non-tool-call text
            remaining = re.sub(tool_call_pattern, "", response_text, flags=re.DOTALL).strip()
            if remaining:
                text_parts.append(remaining)
        else:
            # No tool_call tags — try to parse as raw JSON (some models output JSON directly)
            try:
                parsed = json.loads(response_text.strip())
                if "name" in parsed:
                    tool_calls.append({
                        "name": parsed["name"],
                        "arguments": parsed.get("arguments", {}),
                    })
                else:
                    text_parts.append(response_text)
            except json.JSONDecodeError:
                text_parts.append(response_text)

        return {
            "tool_calls": tool_calls,
            "text": " ".join(text_parts),
        }


def create_client(provider: str, model_id: str) -> LLMClient:
    """
    Factory function to create LLM client.

    Args:
        provider: "anthropic", "openai", "qwen", "huggingface", or "local"
        model_id: Model identifier for the provider

    Returns:
        LLMClient instance
    """
    if provider == "anthropic":
        return ClaudeClient(model_id)
    elif provider == "openai":
        return OpenAIClient(model_id)
    elif provider == "qwen":
        return QwenClient(model_id)
    elif provider == "huggingface":
        return HuggingFaceClient(model_id)
    elif provider == "local":
        return LocalClient(model_id)
    else:
        raise ValueError(f"Unknown provider: {provider}")
