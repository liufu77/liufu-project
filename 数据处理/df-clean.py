import pandas as pd
import streamlit as st
from pathlib import Path

# 读取原始数据
csv_path = Path("assets/nasa_power_beijing_2024_casefile.csv")
df_raw = pd.read_csv(csv_path, skiprows=12)
# 拷贝原始数据，用于清洗（不破坏原数据）
df_clean = df_raw.copy()

# ========== 数据清洗核心逻辑（课件规则落地） ==========
# 需要清洗的数值列：气温、湿度、降水、辐射
num_cols = ["T2M", "RH2M", "PRECTOTCORR", "ALLSKY_SFC_SW_DWN"]
for col in num_cols:
    # errors="coerce"：非法字符串转为 NaN（缺失值）
    df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

# 新增标记列：标记是否为可疑数据
df_clean["is_suspect"] = df_clean["QUALITY_FLAG"] != "ok"

# ========== Streamlit 页面：清洗前后对比 ==========
st.title("NASA数据侦探台 - P2-2 数据清洗（缺失值&异常处理）")

# 分两栏展示：原始数据 | 清洗后数据
col1, col2 = st.columns(2)
with col1:
    st.subheader("原始脏数据")
    st.dataframe(df_raw[df_raw["QUALITY_FLAG"] != "ok"])

with col2:
    st.subheader("清洗后数据")
    st.dataframe(df_clean[df_clean["QUALITY_FLAG"] != "ok"])

# 展示完整清洗后数据
st.subheader("完整清洗后数据表")
st.dataframe(df_clean.head(10))

# 清洗规则说明（对应课件"人话规则"）
st.subheader("清洗规则说明")
st.markdown("""
1. 气温/湿度空值：统一转为 NaN 保留，不随意填充；
2. 降水 `trace`、辐射 `MISSING`、带单位字符串（如0.21C）：转为缺失值 NaN；
3. 异常极值：仅标记，不直接删除；
4. 新增 `is_suspect` 列，永久标记可疑记录。
""")