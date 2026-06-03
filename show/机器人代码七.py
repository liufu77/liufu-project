import os
import time
import streamlit as st
from openai import OpenAI
from openai import APIError, APIConnectionError, AuthenticationError  # ★优化1：导入OpenAI异常类，捕获接口报错
from streamlit.errors import StreamlitSecretNotFoundError

# ★优化2：优化dotenv容错逻辑，避免不存在时全局变量报错
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = lambda: None

# 优先加载.env环境变量
load_dotenv()

# 全局默认配置
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"
# ★优化3：角色预设扩展+规范文案，新增3种实用角色
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
    """★优化4：配置读取逻辑优化，优先环境变量→secrets→默认值，顺序规范"""
    # 1.先读系统环境变量
    env_val = os.getenv(name, "")
    if env_val.strip():
        return env_val.strip()
    # 2.再读取streamlit secrets
    try:
        if name in st.secrets:
            return str(st.secrets[name]).strip()
    except StreamlitSecretNotFoundError:
        pass
    # 3.兜底默认值
    return default.strip()


def ask_ai(
    client: OpenAI,
    model: str,
    messages: list[dict[str, str]],
    system_prompt: str,
    temperature: float,
) -> str:
    """★优化5：接口调用增加异常捕获+流式输出改造，解决长文本卡顿"""
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system_prompt}, *messages],
            temperature=temperature,
            stream=True  # 开启流式返回
        )
        full_content = ""
        content_placeholder = st.empty()
        for chunk in stream:
            chunk_text = chunk.choices[0].delta.content or ""
            full_content += chunk_text
            content_placeholder.markdown(full_content)
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


# ★优化6：页面配置提前置顶（st.set_page_config必须在所有页面元素最开头，原代码位置合规，补充额外配置）
st.set_page_config(
    page_title="My AI Chatbot",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="auto"
)
st.title("🤖 My AI Chatbot")
st.caption("Step 6: classroom-ready version | 优化增强版")

# ★优化7：会话状态统一初始化，合并冗余session_state判断
init_state = {
    "messages": [],
    "selected_role": "通用助手",
    "last_role": "通用助手",
    "system_prompt": ROLE_PRESETS["通用助手"]
}
for key, default_val in init_state.items():
    if key not in st.session_state:
        st.session_state[key] = default_val

# 读取默认密钥配置
default_api_key = get_setting("DEEPSEEK_API_KEY") or get_setting("OPENAI_API_KEY")
default_base_url = get_setting("LLM_BASE_URL", DEFAULT_BASE_URL)
default_model = get_setting("LLM_MODEL", DEFAULT_MODEL)

# 侧边栏配置区
with st.sidebar:
    st.markdown("### ⚙️ API配置")
    api_key = st.text_input("API Key", value=default_api_key, type="password", placeholder="填入Deepseek/OpenAI密钥")
    base_url = st.text_input("Base URL", value=default_base_url, placeholder="接口地址")
    model = st.text_input("Model", value=default_model, placeholder="模型名称")

    st.divider() # ★优化8：增加分割线，UI分层更清爽
    st.markdown("### 🧑‍🏫 角色配置")
    role = st.selectbox("Role Preset", options=list(ROLE_PRESETS), key="selected_role")
    # 切换角色自动更新系统提示词
    if role != st.session_state.last_role:
        st.session_state.system_prompt = ROLE_PRESETS[role]
        st.session_state.last_role = role

    system_prompt = st.text_area("System Prompt", key="system_prompt", height=140, placeholder="自定义AI人设指令")
    temperature = st.slider("Temperature（随机性0~1.5）", min_value=0.0, max_value=1.5, value=0.7, step=0.1)

    st.divider()
    # ★优化9：新增清空+重置配置双按钮
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

# 历史消息渲染
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# 用户输入框
user_input = st.chat_input("请输入内容，开始对话...")

if user_input:
    # 存入用户消息并渲染
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # AI回复区域
    with st.chat_message("assistant"):
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
    # AI回答入库
    st.session_state.messages.append({"role": "assistant", "content": reply})