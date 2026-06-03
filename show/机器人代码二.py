import streamlit as st

st.set_page_config(page_title="My AI Chatbot", page_icon="🤖", layout="centered")
st.title("🤖 My AI Chatbot")
st.caption("Step 2: show history and add a placeholder reply")

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

    reply = "这里是 AI 回复（待接入）"
    st.session_state.messages.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.write(reply)