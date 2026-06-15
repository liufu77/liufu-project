import pandas as pd
import streamlit as st
from pathlib import Path

# ===================== 1. 读取并清洗单站点数据（北京） =====================
csv_beijing = Path("assets/nasa_power_beijing_2024_casefile.csv")
df_raw = pd.read_csv(csv_beijing, skiprows=12)
df_clean = df_raw.copy()

# 数值列清洗
num_cols = ["T2M", "RH2M", "PRECTOTCORR", "ALLSKY_SFC_SW_DWN"]
for col in num_cols:
    df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

# 拼接标准日期列（YEAR/MO/DY 合成 datetime）
df_clean["date"] = pd.to_datetime(
    df_clean[["YEAR", "MO", "DY"]].rename(
        columns={"YEAR": "year", "MO": "month", "DY": "day"}
    )
)

# 衍生月份字段（用于后续分组统计）
df_clean["month_num"] = df_clean["date"].dt.month
df_clean["month_name"] = df_clean["date"].dt.strftime("%b")

# 简化列名（提升可读性）
df_clean = df_clean.rename(
    columns={
        "T2M": "temperature_c",
        "RH2M": "humidity_pct",
        "PRECTOTCORR": "precip_mm",
        "ALLSKY_SFC_SW_DWN": "solar_radiation"
    }
)

# 标记可疑数据
df_clean["is_suspect"] = df_clean["QUALITY_FLAG"] != "ok"

# ===================== 2. 读取多城市整洁数据 =====================
csv_cities = Path("assets/nasa_power_china_cities_2024_tidy.csv")
df_cities = pd.read_csv(csv_cities)

# ===================== Streamlit 仪表盘页面 =====================
st.title("NASA数据侦探台 完整版 | 脏数据处理+数据整洁化")

# 模块1：原始数据 & 清洗对比
st.header("一、数据清洗前后对比")
c1, c2 = st.columns(2)
with c1:
    st.subheader("原始脏数据")
    st.dataframe(df_raw[df_raw["QUALITY_FLAG"] != "ok"])
with c2:
    st.subheader("清洗后标准数据")
    st.dataframe(df_clean[df_clean["is_suspect"]])

# 模块2：整洁化后单站点数据
st.header("二、Tidy 整洁化数据（北京）")
st.dataframe(df_clean[["date", "month_num", "month_name", "temperature_c",
                       "humidity_pct", "precip_mm", "solar_radiation", "is_suspect"]].head(15))

# 模块3：多城市数据（为后续对比做准备）
st.header("三、多城市统一整洁数据")
st.write(f"城市数量：{df_cities['city'].nunique()} | 总观测记录：{len(df_cities)}")
st.dataframe(df_cities.head(10))

# 规则说明
st.header("四、整体处理规则")
st.markdown("""
1. **脏数据清洗**：非法文本转为缺失值，保留可疑标记，不随意删除数据；
2. **日期规整**：将年月日拼接为标准时间格式，衍生月份字段；
3. **列名优化**：缩写列名改为易懂英文，符合 Tidy Data 规范；
4. **数据结构**：一行=单条观测，一列=单种指标，适配后续分组统计。
""")