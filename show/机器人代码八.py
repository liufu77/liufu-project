import os
import time
import streamlit as st
from openai import OpenAI
from openai import APIError, APIConnectionError, AuthenticationError
from streamlit.errors import StreamlitSecretNotFoundError

# ========== 初始化主题状态【新增】==========
if "theme" not in st.session_state:
    st.session_state.theme = "light"  # 默认浅色

def toggle_theme():
    """切换深浅色函数【新增】"""
    st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"
    st.rerun()
# ======================================

# =====================双主题CSS美化【自动根据深浅加载样式】=====================
if st.session_state.theme == "light":
    CSS = """
<style>
/* 全局页面基础样式-浅色 */
.main > .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width:900px;
}
h1 {
    font-weight:700 !important;
    letter-spacing:1px;
    color:#111;
}
/* 用户消息气泡 */
.user-bubble{
    background:#fef2f2;
    border-radius:12px;
    padding:12px 16px;
    margin:6px 0;
    border-left:6px solid #ef4444;
    color:#222;
}
/* AI消息气泡 */
.assistant-bubble{
    background:#fffaf0;
    border-radius:12px;
    padding:12px 16px;
    margin:6px 0;
    border-left:6px solid #f59e0b;
    color:#222;
}
/* 底部输入框美化 */
.stChatInput{
    border:2px solid #ef4444 !important;
    border-radius:10px !important;
}
.stChatInput:focus-within{
    border-color:#f59e0b !important;
}
/* 侧边栏样式 */
[data-testid="stSidebar"]{
    background-color:#f9fafb;
}
</style>
"""
else:
    CSS = """
<style>
/* 全局页面基础样式-深色 */
.stApp{
    background-color:#121418 !important;
    color:#e6e6e6 !important;
}
.main > .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width:900px;
}
h1 {
    font-weight:700 !important;
    letter-spacing:1px;
    color:#ffffff !important;
}
/* 用户消息气泡深色 */
.user-bubble{
    background:#272b33;
    border-radius:12px;
    padding:12px 16px;
    margin:6px 0;
    border-left:6px solid #f87171;
    color:#f1f1f1;
}
/* AI消息气泡深色 */
.assistant-bubble{
    background:#2c2a26;
    border-radius:12px;
    padding:12px 16px;
    margin:6px 0;
    border-left:6px solid #fb923c;
    color:#f1f1f1;
}
/* 底部输入框深色 */
.stChatInput{
    border:2px solid #f87171 !important;
    border-radius:10px !important;
    background:#22262d !important;
}
.stChatInput:focus-within{
    border-color:#fb923c !important;
}
/* 侧边栏深色 */
[data-testid="stSidebar"]{
    background-color:#1e2128 !important;
}
[data-testid="stSidebar"] * {
    color:#e2e2e2 !important;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)
# ======================================================================

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = lambda: None
load_dotenv()

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"
ROLE_PRESETS = {
    "通用助手": "你是一个有帮助的助手。回答清晰、自然、直接。",
    "Python老师": "你是一个严格但友好的 Python 老师。优先提示思路，不直接给完整答案。",
    "法语陪练": "你是一个法语陪练。请用简单法语回答，并在必要时附一行中文解释。",
    "旅行规划师": "你是一个高效的旅行规划师。给出具体、实用、可执行的建议。",
    "吐槽型朋友": "你是一个嘴上毒舌、其实很热心的朋友。语气有趣，但不要冒犯用户。",
    "文案写手": "擅长各类文案撰写，短句精炼、贴合使用场景，按需调整文风",
    "心理咨询倾听": "温和耐心倾听用户烦恼，共情优先，客观开导，不做医学诊断",
    "中小学解题辅导": "讲解题目分步拆解，通俗举例，引导自主思考"
}


def get_setting(name: str, default: str = "") -> str:
    env_val = os.getenv(name, "")
    if env_val.strip():
        return env_val.strip()
    try:
        if name in st.secrets:
            return str(st.secrets[name]).strip()
    except StreamlitSecretNotFoundError:
        pass
    return default.strip()


def ask_ai(
    client: OpenAI,
    model: str,
    messages: list[dict[str, str]],
    system_prompt: str,
    temperature: float,
) -> str:
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system_prompt}, *messages],
            temperature=temperature,
            stream=True
        )
        full_content = ""
        content_placeholder = st.empty()
        for chunk in stream:
            chunk_text = chunk.choices[0].delta.content or ""
            full_content += chunk_text
            content_placeholder.markdown(f'<div class="assistant-bubble">🤖 {full_content}</div>', unsafe_allow_html=True)
            time.sleep(0.015)
        return full_content
    except AuthenticationError:
        return "❌ API密钥错误，请检查侧边栏填入的Key是否正确"
    except APIConnectionError:
        return "❌ 接口连接失败，请检查BaseURL是否填写正确、网络可连通"
    except APIError as e:
        return f"❌ 接口请求异常：{str(e)}"
    except Exception as e:
        return f"❌ 未知错误：{str(e)}"


st.set_page_config(
    page_title="My AI Chatbot",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="auto"
)
st.title("🤖 My AI Chatbot")
st.caption("Step 6: classroom-ready version | 深浅双主题版")

# 会话初始化
init_state = {
    "messages": [],
    "selected_role": "通用助手",
    "last_role": "通用助手",
    "system_prompt": ROLE_PRESETS["通用助手"]
}
for key, default_val in init_state.items():
    if key not in st.session_state:
        st.session_state[key] = default_val

default_api_key = get_setting("DEEPSEEK_API_KEY") or get_setting("OPENAI_API_KEY")
default_base_url = get_setting("LLM_BASE_URL", DEFAULT_BASE_URL)
default_model = get_setting("LLM_MODEL", DEFAULT_MODEL)

# 侧边栏
with st.sidebar:
    # =====【新增主题切换按钮，侧边栏最上方】=====
    theme_btn_text = "🌙 切换深色模式" if st.session_state.theme == "light" else "☀️ 切换浅色模式"
    st.button(theme_btn_text, on_click=toggle_theme, use_container_width=True)
    st.divider()
    # =========================================

    st.markdown("### ⚙️ API配置")
    api_key = st.text_input("API Key", value=default_api_key, type="password", placeholder="填入Deepseek/OpenAI密钥")
    base_url = st.text_input("Base URL", value=default_base_url, placeholder="接口地址")
    model = st.text_input("Model", value=default_model, placeholder="模型名称")

    st.divider()
    st.markdown("### 🧑‍🏫 角色配置")
    role = st.selectbox("Role Preset", options=list(ROLE_PRESETS), key="selected_role")
    if role != st.session_state.last_role:
        st.session_state.system_prompt = ROLE_PRESETS[role]
        st.session_state.last_role = role

    system_prompt = st.text_area("System Prompt", key="system_prompt", height=140, placeholder="自定义AI人设指令")
    temperature = st.slider("Temperature（随机性0~1.5）", min_value=0.0, max_value=1.5, value=0.7, step=0.1)

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("清空对话", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("重置角色", use_container_width=True):
            st.session_state.selected_role = "通用助手"
            st.session_state.last_role = "通用助手"
            st.session_state.system_prompt = ROLE_PRESETS["通用助手"]
            st.rerun()

# 渲染历史聊天
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f'<div class="user-bubble">👤 {message["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="assistant-bubble">🤖 {message["content"]}</div>', unsafe_allow_html=True)

# 用户输入
user_input = st.chat_input("请输入内容，开始对话...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.markdown(f'<div class="user-bubble">👤 {user_input}</div>', unsafe_allow_html=True)

    if not api_key.strip():
        reply = "⚠️ 请先在侧边栏输入 API Key，或在环境变量/.env中配置DEEPSEEK_API_KEY / OPENAI_API_KEY"
        st.warning(reply)
    else:
        client = OpenAI(api_key=api_key.strip(), base_url=base_url.strip())
        with st.spinner("AI思考中..."):
            reply = ask_ai(
                client=client,
                model=model.strip(),
                messages=st.session_state.messages,
                system_prompt=system_prompt.strip(),
                temperature=temperature,
            )
    st.session_state.messages.append({"role": "assistant", "content": reply})