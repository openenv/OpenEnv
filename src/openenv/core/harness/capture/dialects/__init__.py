"""Transform manager — dispatches to the right transformer by API type."""

from ..detection import APIType
from .anthropic import AnthropicTransformer
from .base import BaseTransformer
from .google import GoogleTransformer
from .openai_chat import OpenAIChatTransformer
from .openai_responses import OpenAIResponsesTransformer


class TransformManager:
    """Route to the correct transformer based on detected API type."""

    def __init__(self):
        self._transformers: dict[APIType, BaseTransformer] = {
            APIType.ANTHROPIC: AnthropicTransformer(),
            APIType.OPENAI_CHAT: OpenAIChatTransformer(),
            APIType.OPENAI_RESPONSES: OpenAIResponsesTransformer(),
            APIType.GOOGLE: GoogleTransformer(),
        }

    def get(self, api_type: APIType) -> BaseTransformer:
        return self._transformers[api_type]
