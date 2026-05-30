# Access Different LLMs - Multi-Modal Hub

A Streamlit app that lets you talk to **5 different LLM providers** across **multiple modalities**
(Text, Image, Audio, Video) from a single UI. The app is a working,
production-style wrapper around the experimental notebooks in this folder.

---

## 1. Features

### UI
- **Sidebar controls** to pick:
  - Input modality (Text / Image / Audio / Video)
  - Output modality (Text / Image / Audio / Video)
  - LLM provider (Huggingface, Google Gemini, Groq, OpenRouter, OpenAI)
  - Model (auto-filtered to what the provider supports for that modality, loaded from [`config.py`](config.py))
- **Tabs** for every supported pipeline: `Text-Text`, `Text-Image`, `Image-Text`,
  `Image-Image`, `Text-Audio`, `Audio-Text`, `Audio-Audio`, `Text-Video`, `Video-Text`
- **Prompt box** (multi-line) for the user message
- **Image / audio uploaders** for image- and audio-input modalities; an
  "Attach an image to the prompt" checkbox is available for text inputs too
  (the spec asks: *"Prompt should accept images as well"*)
- **Output panel** that renders text (markdown), images (with download), or
  audio (with download), depending on the modality
- **Go button** to submit; disabled until validation passes

### Validation rules
The Go button is **disabled** unless **all** of these are true:
1. A supported input/output modality combination is selected
2. The prompt is non-empty
3. If input modality is Image -> an image is uploaded
4. If input modality is Audio -> an audio file is uploaded
5. The selected provider has at least one model configured for the selected modality

If Go is disabled, the *"Why is Go disabled?"* expander lists the specific reasons.

### Provider matrix

| Modality      | Huggingface | Google Gemini | Groq | OpenRouter | OpenAI |
| ------------- |:-----------:|:-------------:|:----:|:----------:|:------:|
| Text-Text     | OK | OK | OK | OK | OK |
| Text-Image    | OK | OK | -  | OK | OK |
| Image-Text    | OK | OK | OK | OK | OK |
| Text-Audio    | -  | -  | -  | -  | OK |
| Audio-Text    | OK | -  | OK | -  | OK |
| Image-Image / Audio-Audio / Video | - | - | - | - | - |

The course notebooks explicitly skip Image-Image and Audio-Audio, so the app
shows a "not implemented" note for those tabs instead of failing silently.

---

## 2. Project layout

```
Assignment/Access_Different_LLMs/
├── app.py                          # Streamlit UI
├── config.py                       # Modalities, providers, model lists, env-var map
├── llm_router.py                   # Dispatches a request to the right provider
├── requirements.txt
├── README.md                       # (this file)
├── LLMsTextToTextDiffAPIs.ipynb    # Source notebook for Text-Text logic
├── LLMsTextToImageDiffAPIs.ipynb   # Source notebook for Text-Image logic
├── LLMsImageToTextDiffAPIs.ipynb   # Source notebook for Image-Text logic
├── generated_image.png             # Sample artifact from notebooks
└── images/
    └── AIImage.jpg                 # Sample image used in notebooks
```

The app **does not execute notebook cells**. Instead, the exact provider
patterns from each notebook were lifted into [`llm_router.py`](llm_router.py)
so each call is a real Python function call (faster, safer, debuggable).

---

## 3. Setup

### a) Install dependencies

From the repository root (or this folder):

```powershell
pip install -r Assignment/Access_Different_LLMs/requirements.txt
```

### b) Create a `.env` at the repo root with the keys for any provider you want to use

```dotenv
HUGGINGFACEHUB_API_TOKEN=hf_xxx
GOOGLE_API_KEY=AIza_xxx
GROQ_API_KEY=gsk_xxx
OPENROUTER_API_KEY=sk-or-xxx
OPENAI_API_KEY=sk-xxx
```

You only need keys for the providers you intend to call. The sidebar shows
a green/red badge for the env-var of the currently selected provider, so
missing keys are obvious.

### c) Run the app

```powershell
streamlit run Assignment/Access_Different_LLMs/app.py
```

Streamlit will open `http://localhost:8501` in your browser.

---

## 4. How it works (request flow)

```
User picks modality + provider + model
        |
        v
Streamlit (app.py) validates input
        |
        v
llm_router.run(modality, provider, model, prompt, image_bytes, audio_bytes)
        |
        v
   per-modality handler -> per-provider branch
        |
        v
LangChain / direct SDK call -> Provider API
        |
        v
   {"type": "text"|"image"|"audio", "data": ...}
        |
        v
   Streamlit renders the result
```

### Why some providers use direct SDKs

- **OpenRouter text-to-image**: LangChain's `ChatOpenAI` parses only standard
  OpenAI message fields, so the `images` field from OpenRouter is dropped.
  We call `POST /api/v1/chat/completions` directly and pull `images[0]` from
  the response. (See `LLMsTextToImageDiffAPIs.ipynb` for the full explanation.)
- **Huggingface text-to-image**: `langchain_huggingface` doesn't expose an
  image-generation endpoint. We use `huggingface_hub.InferenceClient.text_to_image`.
- **Huggingface vision (image-to-text)**: The new HF *Inference Providers*
  system isn't supported by `HuggingFaceEndpoint`. We use `ChatOpenAI` pointed
  at `https://router.huggingface.co/v1` with `model:provider` IDs.

---

## 5. Adding a new model

Just append it to the right list in [`config.py`](config.py) - the UI picks
it up automatically on next reload:

```python
MODELS["Text-Text"]["Groq"].append("my-new-groq-model")
```

If you need a new provider, also:
1. Add it to the `PROVIDERS` list and the `PROVIDER_ENV_VAR` dict in `config.py`
2. Add a branch in the appropriate `_text_to_text` / `_text_to_image` / ... function in `llm_router.py`

## 6. Adding a new modality

1. Append the new combo (e.g. `"Video-Image"`) to `MODALITY_TABS` in `config.py`
2. Add an entry to `MODELS` with at least one provider + model
3. Add a handler in `llm_router.py` and dispatch from `run()`

---

## 7. Troubleshooting

| Symptom | Cause / Fix |
| --- | --- |
| Red `XYZ_API_KEY is NOT set` in the sidebar | The env-var isn't in your `.env`, or the app wasn't restarted after editing `.env`. |
| `Go` button is greyed out | Open the *"Why is Go disabled?"* expander - it lists the specific failing rule. |
| `OpenRouter returned 402` | OpenRouter account has no credits. Image-gen models cost money even for "free" model IDs. |
| `langchain_openrouter` import error | The notebook used a hypothetical `langchain_openrouter`. The app uses `langchain_openai` + `base_url=https://openrouter.ai/api/v1` instead. |
| Audio-Audio / Image-Image / *-Video tabs | Intentionally not implemented - the course notebooks skipped these. |

---

## 8. Reference - notebook -> app mapping

| Notebook | Cell / pattern | Lives in |
| --- | --- | --- |
| `LLMsTextToTextDiffAPIs.ipynb` | Huggingface / Groq / OpenRouter / Gemini chat | `llm_router._text_to_text` |
| `LLMsTextToImageDiffAPIs.ipynb` | Gemini, HF Stable Diffusion, OpenRouter image-gen | `llm_router._text_to_image` |
| `LLMsImageToTextDiffAPIs.ipynb` | HF aya-vision via router, Gemini vision | `llm_router._image_to_text` |
