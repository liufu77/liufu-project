import os

import streamlit as st
from openai import OpenAI

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


if load_dotenv is not None:
    load_dotenv()


def ask_ai(client: OpenAI, user_input: str) -> str:
    response = client.chat.completions.create(
        model=os.getenv("LLM_MODEL", "deepseek-chat"),
        messages=[
            {"role": "system", "content": "你是一个有帮助的助手。"},
            {"role": "user", "content": user_input},
        ],
        temperature=0.7,
    )
    return response.choices[0].message.content or ""


st.set_page_config(page_title="My AI Chatbot", page_icon="🤖", layout="centered")
st.title("🤖 My AI Chatbot")
st.caption("Step 3: connect the AI API")

if "messages" not in st.session_state:
    st.session_state.messages = []

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
                reply = ask_ai(client, user_input)
            st.write(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})