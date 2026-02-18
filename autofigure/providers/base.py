"""统一的 LLM 调用接口"""

from typing import Optional, List, Any
from PIL import Image

from ..config import ProviderType
from .bianxie import call_bianxie_text, call_bianxie_multimodal, call_bianxie_image_generation
from .openrouter import call_openrouter_text, call_openrouter_multimodal, call_openrouter_image_generation
from .local import call_local_image_generation, call_kimi_multimodal


def call_llm_text(
    prompt: str,
    api_key: str,
    model: str,
    base_url: str,
    provider: ProviderType,
    max_tokens: int = 16000,
    temperature: float = 0.7,
) -> Optional[str]:
    """统一的文本 LLM 调用接口"""
    if provider == "bianxie":
        return call_bianxie_text(prompt, api_key, model, base_url, max_tokens, temperature)
    elif provider == "openrouter":
        return call_openrouter_text(prompt, api_key, model, base_url, max_tokens, temperature)
    else:
        return call_openrouter_text(prompt, api_key, model, base_url, max_tokens, temperature)


def call_llm_multimodal(
    contents: List[Any],
    api_key: str,
    model: str,
    base_url: str,
    provider: ProviderType,
    max_tokens: int = 16000,
    temperature: float = 0.7,
) -> Optional[str]:
    """统一的多模态 LLM 调用接口"""
    if provider == "bianxie":
        return call_bianxie_multimodal(contents, api_key, model, base_url, max_tokens, temperature)
    elif provider == "local":
        return call_kimi_multimodal(contents, api_key, model, base_url, max_tokens, temperature)
    else:
        return call_openrouter_multimodal(contents, api_key, model, base_url, max_tokens, temperature)


def call_llm_image_generation(
    prompt: str,
    api_key: str,
    model: str,
    base_url: str,
    provider: ProviderType,
    reference_image: Optional[Image.Image] = None,
    local_img_path: Optional[str] = None,
) -> Optional[Image.Image]:
    """统一的图像生成 LLM 调用接口"""
    if provider == "bianxie":
        return call_bianxie_image_generation(prompt, api_key, model, base_url, reference_image)
    elif provider == "openrouter":
        return call_openrouter_image_generation(prompt, api_key, model, base_url, reference_image)
    elif provider == "local":
        return call_local_image_generation(prompt, api_key, model, base_url, reference_image, local_img_path=local_img_path)
    else:
        raise ValueError(f"Unsupported provider: {provider}")

