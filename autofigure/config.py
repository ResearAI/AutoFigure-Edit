"""配置文件"""

from typing import Literal

# Provider 类型定义
ProviderType = Literal["openrouter", "bianxie", "local", "kimi"]
PlaceholderMode = Literal["none", "box", "label"]

# Provider 配置
PROVIDER_CONFIGS = {
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "default_image_model": "google/gemini-3-pro-image-preview",
        "default_svg_model": "google/gemini-3-pro-preview",
    },
    "bianxie": {
        "base_url": "https://api.bianxie.ai/v1",
        "default_image_model": "gemini-3-pro-image-preview",
        "default_svg_model": "gemini-3-pro-preview",
    },
    "local": {
        'base_url': 'xxxx',
        "default_image_model": "",
        "default_svg_model": "xxx",
    },
}

# SAM3 API 配置
SAM3_FAL_API_URL = "https://fal.run/fal-ai/sam-3/image"
SAM3_ROBOFLOW_API_URL = "https://serverless.roboflow.com/sam3/concept_segment"
SAM3_API_TIMEOUT = 300

# 全局设置
USE_REFERENCE_IMAGE = False
REFERENCE_IMAGE_PATH = None

