import math
import streamlit as st

st.title("🔮 元素测试仪")

name = st.text_input("你的代号", "见习炼金术士")
fire = st.slider("火属性", 0, 10, 5)
water = st.slider("水属性", 0, 10, 5)

profiles = {
    "🔥 烈焰型": [10, 3],
    "💧 潮汐型": [3, 10],
    "🌪️ 平衡游侠": [7, 7],
}

if st.button("开始匹配"):
    user = [fire, water]
    min_dist = 999999.0
    best = ""
    for pname, coords in profiles.items():
        dist = math.sqrt(
            (user[0] - coords[0]) ** 2 + (user[1] - coords[1]) ** 2
        )
        if dist < min_dist:
            min_dist = dist
            best = pname
    st.write(f"{name}，你最接近：{best}")