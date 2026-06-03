import os

import streamlit as st
from openai import OpenAI

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


if load_dotenv is not None:
    load_dotenv()


ROLE_PRESETS = {
    "通用助手": "你是一个有帮助的助手。回答清晰、自然、直接。",
    "Python老师": "你是一个严格但友好的 Python 老师。优先提示思路，不直接给完整答案。",
    "吐槽型朋友": "你是一个嘴上毒舌、其实很热心的朋友。语气有趣，但不要冒犯用户。",
}


def ask_ai(client: OpenAI, messages: list[dict[str, str]], system_prompt: str) -> str:
    response = client.chat.completions.create(
        model=os.getenv("LLM_MODEL", "deepseek-chat"),
        messages=[
            {"role": "system", "content": system_prompt},
            *messages,
        ],
        temperature=0.7,
    )
    return response.choices[0].message.content or ""


st.set_page_config(page_title="My AI Chatbot", page_icon="🤖", layout="centered")
st.title("🤖 My AI Chatbot")
st.caption("Step 5: introduce system prompt and role presets")

if "messages" not in st.session_state:
    st.session_state.messages = []

role = st.selectbox("Role Preset", options=list(ROLE_PRESETS))
system_prompt = ROLE_PRESETS[role]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_input = st.chat_input("请输入内容")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            reply = "请先在 .env 中配置 DEEPSEEK_API_KEY 或 OPENAI_API_KEY。"
            st.warning(reply)
        else:
            client = OpenAI(
                api_key=api_key,
                base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com"),
            )
            with st.spinner("Thinking..."):
                reply = ask_ai(client, st.session_state.messages, system_prompt)
            st.write(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})