"""
LLM Multi-Modal Hub - Streamlit UI

Lets the user pick:
  * Input modality (Text / Image / Audio / Video)
  * Output modality (Text / Image / Audio / Video)
  * LLM provider (Huggingface / Google Gemini / Groq / OpenRouter / OpenAI)
  * Model (filtered to what the provider supports for that modality)
  * Prompt (and optionally an uploaded image / audio)

Submit is disabled until the user has selected a modality combination AND
provided a non-empty prompt.

Run from this folder:
    streamlit run app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

# Make sibling modules importable when run via `streamlit run app.py`
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    INPUT_MODALITIES,
    MODALITY_TABS,
    OUTPUT_MODALITIES,
    PROVIDER_ENV_VAR,
    PROVIDERS,
    models_for,
    providers_for,
)
from llm_router import LLMError, run


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="LLM Multi-Modal Hub",
    page_icon="🤖",
    layout="wide",
)

st.title("LLM Multi-Modal Hub")
st.caption(
    "Pick an input/output modality, a provider, a model, type a prompt, hit Go."
)


# ---------------------------------------------------------------------------
# Sidebar - global controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Modality & Provider")

    input_modality = st.selectbox("Input modality", INPUT_MODALITIES, index=0)
    output_modality = st.selectbox("Output modality", OUTPUT_MODALITIES, index=0)

    selected_combo = f"{input_modality}-{output_modality}"

    if selected_combo not in MODALITY_TABS:
        st.warning(
            f"`{selected_combo}` is not a supported combination. "
            f"Pick one of: {', '.join(MODALITY_TABS)}"
        )

    available_providers = providers_for(selected_combo)
    provider_options = available_providers if available_providers else PROVIDERS
    provider = st.selectbox("LLM provider", provider_options, index=0)

    models = models_for(selected_combo, provider)
    model = st.selectbox(
        "Model",
        models if models else ["(no models configured)"],
        index=0,
        disabled=not models,
    )

    # Show env-var status so the user knows whether a key is loaded
    env_var = PROVIDER_ENV_VAR[provider]
    if os.getenv(env_var):
        st.success(f"`{env_var}` is set")
    else:
        st.error(f"`{env_var}` is NOT set - add it to your .env")


# ---------------------------------------------------------------------------
# Modality tabs (for spec compliance: tabs for every supported combination)
# ---------------------------------------------------------------------------
st.subheader("Pipelines")
tab_objs = st.tabs(MODALITY_TABS)
for tab, label in zip(tab_objs, MODALITY_TABS):
    with tab:
        if label == selected_combo:
            st.success(f"Active: **{label}**")
        else:
            st.info(
                f"To use **{label}**, set Input/Output to "
                f"`{label.split('-')[0]}` / `{label.split('-')[1]}` in the sidebar."
            )


# ---------------------------------------------------------------------------
# Main input area
# ---------------------------------------------------------------------------
st.subheader("Prompt")

prompt = st.text_area(
    "Your prompt",
    placeholder="Ask anything, or describe the image/audio/video you want...",
    height=140,
    key="prompt_input",
)

# Image uploader is shown when input modality is Image OR the user just wants
# to attach an image to the prompt (the spec asks: "Prompt should accept
# images as well.").
uploaded_image = None
uploaded_audio = None
image_bytes: bytes | None = None
audio_bytes: bytes | None = None
image_mime = "image/jpeg"

upload_col1, upload_col2 = st.columns(2)
with upload_col1:
    accept_image = input_modality == "Image" or st.checkbox(
        "Attach an image to the prompt", value=False, key="attach_image"
    )
    if accept_image:
        uploaded_image = st.file_uploader(
            "Upload image",
            type=["png", "jpg", "jpeg", "webp", "gif"],
            key="image_upload",
        )
        if uploaded_image is not None:
            image_bytes = uploaded_image.read()
            image_mime = uploaded_image.type or "image/jpeg"
            st.image(image_bytes, caption=uploaded_image.name, width=320)

with upload_col2:
    if input_modality == "Audio":
        uploaded_audio = st.file_uploader(
            "Upload audio",
            type=["mp3", "wav", "m4a", "ogg", "flac"],
            key="audio_upload",
        )
        if uploaded_audio is not None:
            audio_bytes = uploaded_audio.read()
            st.audio(audio_bytes)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
validation_errors: list[str] = []

if selected_combo not in MODALITY_TABS:
    validation_errors.append("Pick a supported input/output modality combination.")
if not prompt.strip():
    validation_errors.append("Prompt cannot be empty.")
if input_modality == "Image" and not image_bytes:
    validation_errors.append("Upload an image - input modality is Image.")
if input_modality == "Audio" and not audio_bytes:
    validation_errors.append("Upload an audio file - input modality is Audio.")
if not models_for(selected_combo, provider):
    validation_errors.append(
        f"No models configured for `{selected_combo}` on `{provider}`."
    )

submit_disabled = bool(validation_errors)

if validation_errors:
    with st.expander("Why is Go disabled?", expanded=False):
        for err in validation_errors:
            st.write(f"- {err}")

go = st.button(
    "Go",
    type="primary",
    disabled=submit_disabled,
    use_container_width=True,
)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
st.subheader("Output")
output_placeholder = st.container()

if go:
    with st.spinner(f"Calling {provider} - {model}..."):
        try:
            result = run(
                modality=selected_combo,
                provider=provider,
                model=model,
                prompt=prompt,
                image_bytes=image_bytes,
                image_mime=image_mime,
                audio_bytes=audio_bytes,
            )
        except LLMError as e:
            output_placeholder.error(str(e))
            st.stop()
        except Exception as e:  # noqa: BLE001 - surface anything else to the user
            output_placeholder.exception(e)
            st.stop()

    with output_placeholder:
        if result.get("note"):
            st.info(result["note"])
        if result["type"] == "text":
            if result["data"]:
                st.markdown(result["data"])
        elif result["type"] == "image":
            st.image(result["data"], caption=f"{provider} - {model}")
            st.download_button(
                "Download image",
                data=result["data"],
                file_name="output.png",
                mime="image/png",
            )
        elif result["type"] == "audio":
            st.audio(result["data"])
            st.download_button(
                "Download audio",
                data=result["data"],
                file_name="output.mp3",
                mime="audio/mpeg",
            )
        else:
            st.write(result["data"])
else:
    output_placeholder.caption(
        "Output appears here after you press **Go**."
    )
