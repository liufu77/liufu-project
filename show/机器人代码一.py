import streamlit as st

st.set_page_config(page_title="我的AI聊天机器人", page_icon="🤖", layout="centered")
st.title("🤖 我的AI聊天机器人")
st.caption("Step 1: smallest possible chat page")

if "messages" not in st.session_state:
    st.session_state.messages = []

user_input = st.chat_input("请输入内容")

if user_input:
    st.write(f"你刚刚输入了：{user_input}")