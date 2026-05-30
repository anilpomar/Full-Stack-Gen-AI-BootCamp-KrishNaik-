"""
Central configuration for LLM Playground.

Each modality (e.g. "Text-Text", "Text-Image") maps to a dict of providers.
Each provider lists the models supported for that modality.

To add a new model, just append it to the appropriate list - the UI picks
it up automatically.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Modality combinations exposed as tabs in the UI
# ---------------------------------------------------------------------------
MODALITY_TABS: list[str] = [
    "Text-Text",
    "Text-Image",
    "Image-Text",
    "Image-Image",
    "Text-Audio",
    "Audio-Text",
    "Audio-Audio",
    "Text-Video",
    "Video-Text",
]

# Inputs / Outputs (used by the "modality selector" controls)
INPUT_MODALITIES: list[str] = ["Text", "Image", "Audio", "Video"]
OUTPUT_MODALITIES: list[str] = ["Text", "Image", "Audio", "Video"]

# All providers shown in the dropdown
PROVIDERS: list[str] = [
    "Huggingface",
    "Google Gemini",
    "Groq",
    "OpenRouter",
    "OpenAI",
]

# ---------------------------------------------------------------------------
# Models per (modality, provider)
# Sourced from the Access_Different_LLMs notebooks plus a few well-known IDs.
# ---------------------------------------------------------------------------
MODELS: dict[str, dict[str, list[str]]] = {
    "Text-Text": {
        "Huggingface": [
            "meta-llama/Llama-3.1-8B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.3",
            "deepseek-ai/DeepSeek-V3-0324",
        ],
        "Google Gemini": [
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-1.5-flash",
        ],
        "Groq": [
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile",
            "mixtral-8x7b-32768",
        ],
        "OpenRouter": [
            "openai/gpt-4o-mini",
            "google/gemini-2.5-flash",
            "meta-llama/llama-3.1-8b-instruct",
        ],
        "OpenAI": [
            "gpt-4o-mini",
            "gpt-4o",
            "gpt-4-turbo",
        ],
    },
    "Text-Image": {
        "Huggingface": [
            "stabilityai/stable-diffusion-3-medium",
            "black-forest-labs/FLUX.1-schnell",
        ],
        "Google Gemini": [
            "gemini-2.5-flash-image",
        ],
        "OpenRouter": [
            "google/gemini-2.5-flash-image",
        ],
        "OpenAI": [
            "dall-e-3",
            "gpt-image-1",
        ],
        # Groq currently has no image-generation models.
        "Groq": [],
    },
    "Image-Text": {
        "Huggingface": [
            "CohereLabs/aya-vision-32b",
            "Qwen/Qwen2-VL-7B-Instruct",
        ],
        "Google Gemini": [
            "gemini-2.5-flash",
            "gemini-2.5-pro",
        ],
        "OpenRouter": [
            "openai/gpt-4o-mini",
            "google/gemini-2.5-flash",
        ],
        "OpenAI": [
            "gpt-4o-mini",
            "gpt-4o",
        ],
        "Groq": [
            "llama-3.2-11b-vision-preview",
        ],
    },
    # The notebook explicitly skipped Image-Image / Audio-Audio / Video-Video.
    # The UI still shows tabs, but the handlers return a "coming soon" message.
    "Image-Image": {p: [] for p in PROVIDERS},
    "Text-Audio": {
        "OpenAI": ["tts-1", "tts-1-hd"],
        "Huggingface": [], "Google Gemini": [], "Groq": [], "OpenRouter": [],
    },
    "Audio-Text": {
        "OpenAI": ["whisper-1"],
        "Groq": ["whisper-large-v3"],
        "Huggingface": ["openai/whisper-large-v3"],
        "Google Gemini": [], "OpenRouter": [],
    },
    "Audio-Audio": {p: [] for p in PROVIDERS},
    "Text-Video": {p: [] for p in PROVIDERS},
    "Video-Text": {p: [] for p in PROVIDERS},
}


# Which providers support each modality (derived from MODELS)
def providers_for(modality: str) -> list[str]:
    return [p for p, models in MODELS.get(modality, {}).items() if models]


def models_for(modality: str, provider: str) -> list[str]:
    return MODELS.get(modality, {}).get(provider, [])


# ---------------------------------------------------------------------------
# Required environment variables per provider
# ---------------------------------------------------------------------------
PROVIDER_ENV_VAR: dict[str, str] = {
    "Huggingface": "HUGGINGFACEHUB_API_TOKEN",
    "Google Gemini": "GOOGLE_API_KEY",
    "Groq": "GROQ_API_KEY",
    "OpenRouter": "OPENROUTER_API_KEY",
    "OpenAI": "OPENAI_API_KEY",
}
