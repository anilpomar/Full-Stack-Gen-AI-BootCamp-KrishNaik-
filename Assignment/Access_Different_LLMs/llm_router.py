"""
LLM Router.

Routes a (modality, provider, model, prompt, optional image bytes) request to
the right backend. Each handler is a thin wrapper around the corresponding
notebook in this folder.

Public entry point: ``run(modality, provider, model, prompt, image_bytes,
image_mime, audio_bytes)``.

Return shape is always a dict:
    {"type": "text" | "image" | "audio", "data": <str | bytes>, "note": str}

So the UI can render any modality uniformly.
"""

from __future__ import annotations

import base64
import io
import os
from typing import Optional

from dotenv import load_dotenv

from config import PROVIDER_ENV_VAR

load_dotenv(override=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class LLMError(RuntimeError):
    """Raised when a backend call cannot proceed (missing key, bad input)."""


def _require_key(provider: str) -> str:
    env_var = PROVIDER_ENV_VAR.get(provider)
    if not env_var:
        raise LLMError(f"Unknown provider: {provider}")
    key = os.getenv(env_var)
    if not key:
        raise LLMError(
            f"{env_var} is not set. Add it to your .env file before calling "
            f"{provider}."
        )
    return key


def _image_to_data_url(image_bytes: bytes, mime: str = "image/jpeg") -> str:
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def _ok(kind: str, data, note: str = "") -> dict:
    return {"type": kind, "data": data, "note": note}


# ---------------------------------------------------------------------------
# Text -> Text
# ---------------------------------------------------------------------------
def _text_to_text(provider: str, model: str, prompt: str) -> dict:
    from langchain_core.messages import HumanMessage, SystemMessage

    system_msg = SystemMessage(content="You are a helpful assistant.")
    human_msg = HumanMessage(content=prompt)

    if provider == "Huggingface":
        _require_key(provider)
        from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

        llm = HuggingFaceEndpoint(
            repo_id=model, task="text-generation", max_new_tokens=512
        )
        chat = ChatHuggingFace(llm=llm)
        return _ok("text", chat.invoke([system_msg, human_msg]).content)

    if provider == "Groq":
        _require_key(provider)
        from langchain_groq import ChatGroq

        llm = ChatGroq(model=model)
        return _ok("text", llm.invoke([system_msg, human_msg]).content)

    if provider == "Google Gemini":
        key = _require_key(provider)
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(
            model=model, google_api_key=key, max_output_tokens=2000
        )
        return _ok("text", llm.invoke([system_msg, human_msg]).content)

    if provider == "OpenAI":
        _require_key(provider)
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model=model)
        return _ok("text", llm.invoke([system_msg, human_msg]).content)

    if provider == "OpenRouter":
        key = _require_key(provider)
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=model, api_key=key, base_url="https://openrouter.ai/api/v1"
        )
        return _ok("text", llm.invoke([system_msg, human_msg]).content)

    raise LLMError(f"Provider '{provider}' is not supported for Text-Text.")


# ---------------------------------------------------------------------------
# Text -> Image
# ---------------------------------------------------------------------------
def _text_to_image(provider: str, model: str, prompt: str) -> dict:
    if provider == "Huggingface":
        _require_key(provider)
        from huggingface_hub import InferenceClient

        client = InferenceClient(token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))
        image = client.text_to_image(prompt=prompt, model=model)
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return _ok("image", buf.getvalue())

    if provider == "Google Gemini":
        key = _require_key(provider)
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(model=model, google_api_key=key)
        response = llm.invoke(
            prompt,
            generation_config={"response_modalities": ["TEXT", "IMAGE"]},
        )
        image_block = next(
            (
                b
                for b in response.content
                if isinstance(b, dict) and b.get("image_url")
            ),
            None,
        )
        if image_block is None:
            raise LLMError("Gemini returned no image block in the response.")
        b64 = image_block["image_url"]["url"].split(",", 1)[-1]
        return _ok("image", base64.b64decode(b64))

    if provider == "OpenRouter":
        # OpenRouter image-gen via raw HTTP - the langchain wrapper drops the
        # images field. See LLMsTextToImageDiffAPIs.ipynb for the why.
        key = _require_key(provider)
        import requests

        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "modalities": ["image"],
                "image_config": {"aspect_ratio": "16:9", "image_size": "1K"},
            },
            timeout=120,
        )
        if resp.status_code != 200:
            raise LLMError(
                f"OpenRouter returned {resp.status_code}: {resp.text[:200]}"
            )
        data = resp.json()
        message = data.get("choices", [{}])[0].get("message", {})
        images = message.get("images") or []
        if not images:
            raise LLMError(
                "OpenRouter response did not include any images. "
                f"Text content: {message.get('content', '')[:200]}"
            )
        b64 = images[0]["image_url"]["url"].split(",", 1)[-1]
        return _ok("image", base64.b64decode(b64))

    if provider == "OpenAI":
        _require_key(provider)
        from openai import OpenAI

        client = OpenAI()
        result = client.images.generate(model=model, prompt=prompt, n=1)
        # Either b64_json or a URL depending on the model
        item = result.data[0]
        if getattr(item, "b64_json", None):
            return _ok("image", base64.b64decode(item.b64_json))
        if getattr(item, "url", None):
            import requests

            return _ok("image", requests.get(item.url, timeout=60).content)
        raise LLMError("OpenAI image response had no data.")

    raise LLMError(f"Provider '{provider}' is not supported for Text-Image.")


# ---------------------------------------------------------------------------
# Image -> Text
# ---------------------------------------------------------------------------
def _image_to_text(
    provider: str,
    model: str,
    prompt: str,
    image_bytes: bytes,
    image_mime: str,
) -> dict:
    if not image_bytes:
        raise LLMError("Image-Text requires an uploaded image.")

    from langchain_core.messages import HumanMessage

    data_url = _image_to_data_url(image_bytes, image_mime)
    message = HumanMessage(
        content=[
            {"type": "image_url", "image_url": {"url": data_url}},
            {"type": "text", "text": prompt},
        ]
    )

    if provider == "Huggingface":
        _require_key(provider)
        from langchain_openai import ChatOpenAI
        from pydantic import SecretStr

        # HF Inference router needs model_id:provider; assume cohere for aya,
        # hf-inference for everything else (matches the notebook).
        router_model = (
            f"{model}:cohere" if "aya" in model.lower() else f"{model}:hf-inference"
        )
        llm = ChatOpenAI(
            model=router_model,
            base_url="https://router.huggingface.co/v1",
            api_key=SecretStr(os.getenv("HUGGINGFACEHUB_API_TOKEN") or ""),
            max_tokens=512,
        )
        return _ok("text", llm.invoke([message]).content)

    if provider == "Google Gemini":
        key = _require_key(provider)
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(model=model, google_api_key=key)
        return _ok("text", llm.invoke([message]).content)

    if provider == "OpenAI":
        _require_key(provider)
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model=model)
        return _ok("text", llm.invoke([message]).content)

    if provider == "OpenRouter":
        key = _require_key(provider)
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=model, api_key=key, base_url="https://openrouter.ai/api/v1"
        )
        return _ok("text", llm.invoke([message]).content)

    if provider == "Groq":
        _require_key(provider)
        from langchain_groq import ChatGroq

        llm = ChatGroq(model=model)
        return _ok("text", llm.invoke([message]).content)

    raise LLMError(f"Provider '{provider}' is not supported for Image-Text.")


# ---------------------------------------------------------------------------
# Text -> Audio (TTS)
# ---------------------------------------------------------------------------
def _text_to_audio(provider: str, model: str, prompt: str) -> dict:
    if provider == "OpenAI":
        _require_key(provider)
        from openai import OpenAI

        client = OpenAI()
        speech = client.audio.speech.create(model=model, voice="alloy", input=prompt)
        return _ok("audio", speech.read())

    raise LLMError(f"Provider '{provider}' is not supported for Text-Audio.")


# ---------------------------------------------------------------------------
# Audio -> Text (STT)
# ---------------------------------------------------------------------------
def _audio_to_text(
    provider: str, model: str, prompt: str, audio_bytes: bytes
) -> dict:
    if not audio_bytes:
        raise LLMError("Audio-Text requires an uploaded audio file.")

    if provider == "OpenAI":
        _require_key(provider)
        from openai import OpenAI

        client = OpenAI()
        result = client.audio.transcriptions.create(
            model=model, file=("audio.mp3", audio_bytes)
        )
        return _ok("text", result.text)

    if provider == "Groq":
        key = _require_key(provider)
        from groq import Groq

        client = Groq(api_key=key)
        result = client.audio.transcriptions.create(
            model=model, file=("audio.mp3", audio_bytes)
        )
        return _ok("text", result.text)

    if provider == "Huggingface":
        key = _require_key(provider)
        from huggingface_hub import InferenceClient

        client = InferenceClient(token=key)
        text = client.automatic_speech_recognition(audio=audio_bytes, model=model)
        return _ok("text", getattr(text, "text", str(text)))

    raise LLMError(f"Provider '{provider}' is not supported for Audio-Text.")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def run(
    modality: str,
    provider: str,
    model: str,
    prompt: str,
    image_bytes: Optional[bytes] = None,
    image_mime: str = "image/jpeg",
    audio_bytes: Optional[bytes] = None,
) -> dict:
    """Dispatch a request to the appropriate provider handler."""
    if modality == "Text-Text":
        return _text_to_text(provider, model, prompt)
    if modality == "Text-Image":
        return _text_to_image(provider, model, prompt)
    if modality == "Image-Text":
        return _image_to_text(provider, model, prompt, image_bytes or b"", image_mime)
    if modality == "Text-Audio":
        return _text_to_audio(provider, model, prompt)
    if modality == "Audio-Text":
        return _audio_to_text(provider, model, prompt, audio_bytes or b"")
    if modality in {"Image-Image", "Audio-Audio", "Text-Video", "Video-Text"}:
        return _ok(
            "text",
            "",
            note=(
                f"{modality} is not yet implemented. The course notebooks "
                "explicitly skip this modality."
            ),
        )
    raise LLMError(f"Unknown modality: {modality}")
