"""Model abstractions — platform-agnostic model interface."""

from rai_toolkit.models.base import BaseModel, ModelResponse
from rai_toolkit.models.openai_compatible import OpenAICompatibleModel

__all__ = ["BaseModel", "ModelResponse", "OpenAICompatibleModel"]
