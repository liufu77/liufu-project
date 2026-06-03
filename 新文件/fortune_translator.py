import json
import os
import re

import requests
import streamlit as st

st.set_page_config(page_title="Fortune Translator", layout="centered")
st.title("Fortune Translator")
st.write("从在线 fortune API 获取一句英文 fortune，并翻译成自然中文，附上三种语气的解释。")


def clean_fortune(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r", "\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    text = " ".join(lines)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"[-=_]{2,}", " ", text)
    text = re.sub(r"[\u200b\u200c\u200d]", "", text)
    text = re.sub(r"[“”]+", '"', text)
    text = re.sub(r"[‘’]+", "'", text)
    text = re.sub(r"[^\w\s\.,!\?\'\"\-:;()\/]+", " ", text)
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    text = re.sub(r"([.,!?;:]){2,}", r"\1", text)
    text = text.strip()
    if text and text[-1] not in ".!?":
        text += "."
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    return text


def fetch_online_fortune() -> str:
    candidates = [
        ("https://api.ef.gy/fortune?format=json", "fortune"),
        ("https://api.quotable.io/random?tags=life,wisdom&maxLength=120", "quotable"),
        ("https://api.adviceslip.com/advice", "advice"),
    ]
    last_error = None
    for url, kind in candidates:
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            if kind == "fortune":
                data = response.json()
                if isinstance(data, dict):
                    for key in ("fortune", "text", "quote"):
                        if key in data:
                            return str(data[key]).strip()
                return response.text.strip()
            if kind == "quotable":
                data = response.json()
                if isinstance(data, dict) and data.get("content"):
                    return str(data["content"]).strip()
            if kind == "advice":
                data = response.json()
                if isinstance(data, dict) and data.get("slip") and data["slip"].get("advice"):
                    return str(data["slip"]["advice"]).strip()
        except Exception as exc:
            last_error = exc
            continue
    raise RuntimeError(
        "无法从在线 fortune API 获取内容。请检查网络或切换到可用的 Base URL/API Key 后重试。"
    ) from last_error


def call_ai_translate(api_key: str, base_url: str, model: str, temperature: float, prompt: str) -> str:
    if not api_key:
        raise RuntimeError("缺少 API Key，请右侧输入 API Key。")
    if not base_url:
        raise RuntimeError("缺少 Base URL，请右侧输入 Base URL。")
    if not model:
        raise RuntimeError("缺少 Model，请右侧输入模型名称。")

    api_endpoint = base_url.rstrip("/") + "/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 800,
    }

    response = requests.post(api_endpoint, headers=headers, json=payload, timeout=25)
    if response.status_code >= 400:
        raise RuntimeError(f"AI 接口异常：{response.status_code}，返回内容：{response.text}")
    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeError("AI 返回的不是合法 JSON。") from exc

    if not data.get("choices"):
        raise RuntimeError("AI 返回格式异常，缺少 choices。")

    content = data["choices"][0].get("message", {}).get("content", "").strip()
    if not content:
        raise RuntimeError("AI 返回了空内容。")
    return content


def parse_translation(raw: str) -> dict:
    try:
        parsed = json.loads(raw)
        return {
            "chinese": parsed.get("chinese", "").strip(),
            "teacher": parsed.get("teacher", "").strip(),
            "friend": parsed.get("friend", "").strip(),
            "complaint": parsed.get("complaint", "").strip(),
        }
    except json.JSONDecodeError:
        result = {}
        for key in ("chinese", "teacher", "friend", "complaint"):
            pattern = re.compile(
                rf"{key}[:：]\s*(.+?)(?=(?:\n\s*(?:teacher|friend|complaint)[:：])|\Z)",
                re.IGNORECASE | re.DOTALL,
            )
            match = pattern.search(raw)
            result[key] = match.group(1).strip() if match else ""
        return result


def build_prompt(cleaned_fortune: str) -> str:
    return (
        "请将下面这句英文 fortune 翻译成自然中文，并分别用三种语气解释它的意思：\n"
        "1) 老师式\n"
        "2) 朋友式\n"
        "3) 吐槽式\n"
        "请按 JSON 格式输出，字段为 chinese、teacher、friend、complaint，值为中文文本。\n"
        "只输出 JSON，不要多余说明。\n\n"
        f"英文 fortune：\n{cleaned_fortune}"
    )


if "fortune_raw" not in st.session_state:
    st.session_state.fortune_raw = ""
if "fortune_clean" not in st.session_state:
    st.session_state.fortune_clean = ""
if "translation" not in st.session_state:
    st.session_state.translation = None
if "error_message" not in st.session_state:
    st.session_state.error_message = ""

st.sidebar.header("AI 配置")
default_api_key = os.environ.get("OPENAI_API_KEY", os.environ.get("API_KEY", ""))
default_base = os.environ.get("LLM_BASE_URL", "https://api.openai.com")
default_model = os.environ.get("LLM_MODEL", "gpt-4")
default_temp = float(os.environ.get("LLM_TEMPERATURE", "0.5"))
api_key = st.sidebar.text_input("API Key", value=default_api_key, type="password")
base_url = st.sidebar.text_input("Base URL", value=default_base)
model = st.sidebar.text_input("Model", value=default_model)
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=default_temp, step=0.05)

st.sidebar.markdown(
    "---\n"
    "如果你使用其它大模型服务，请确认 Base URL 支持 OpenAI /v1/chat/completions 接口。"
)

button_col1, button_col2 = st.columns(2)
with button_col1:
    go_button = st.button("获取 fortune")
with button_col2:
    refresh_button = st.button("换一句")

if go_button:
    if not st.session_state.fortune_raw:
        refresh_button = True
    else:
        st.info("已复用上一次获取的 fortune，点击“换一句”可重新请求。")

if refresh_button:
    st.session_state.error_message = ""
    try:
        raw = fetch_online_fortune()
        st.session_state.fortune_raw = raw
        st.session_state.fortune_clean = clean_fortune(raw)
        st.session_state.translation = None
    except Exception as exc:
        st.session_state.error_message = str(exc)
        st.session_state.fortune_raw = st.session_state.fortune_raw or ""
        st.session_state.fortune_clean = st.session_state.fortune_clean or ""
        st.session_state.translation = None

if st.session_state.fortune_clean and st.session_state.translation is None:
    try:
        prompt = build_prompt(st.session_state.fortune_clean)
        raw_ai = call_ai_translate(api_key, base_url, model, temperature, prompt)
        translation = parse_translation(raw_ai)
        st.session_state.translation = translation
    except Exception as exc:
        st.session_state.error_message = str(exc)
        st.session_state.translation = {"chinese": "", "teacher": "", "friend": "", "complaint": ""}

if st.session_state.error_message:
    st.error(st.session_state.error_message)

if st.session_state.fortune_raw:
    st.subheader("原始英文 fortune")
    st.code(st.session_state.fortune_raw)
    st.subheader("清洗后的英文")
    st.write(st.session_state.fortune_clean)

    if st.session_state.translation:
        st.subheader("中文翻译")
        st.write(st.session_state.translation.get("chinese", ""))
        st.subheader("老师式解释")
        st.write(st.session_state.translation.get("teacher", ""))
        st.subheader("朋友式解释")
        st.write(st.session_state.translation.get("friend", ""))
        st.subheader("吐槽式解释")
        st.write(st.session_state.translation.get("complaint", ""))
else:
    st.info("点击上方“获取 fortune”按钮，先从在线服务获取一条英文 fortune。")
