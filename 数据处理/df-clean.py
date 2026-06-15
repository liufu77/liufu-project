# 导入库
import pandas as pd
import streamlit as st
from pathlib import Path

# 1. 定义文件路径，读取原始数据（跳过前12行说明表头）
csv_path = Path("assets/nasa_power_beijing_2024_casefile.csv")
df_raw = pd.read_csv(csv_path, skiprows=12)

# 2. 页面标题
st.title("NASA数据侦探台 - P2-1 原始数据侦查")

# 3. 数据基础信息
st.subheader("1. 数据基本信息")
st.write(f"数据表行列数：{df_raw.shape}")
st.write("所有列名：", list(df_raw.columns))

# 4. 展示原始数据
st.subheader("2. 原始数据表格")
st.dataframe(df_raw)

# 5. 统计每列缺失值数量
st.subheader("3. 缺失值统计")
missing_count = df_raw.isna().sum().rename("缺失数量")
st.dataframe(missing_count)

# 6. 筛选嫌疑数据（QUALITY_FLAG 不为 ok 的脏数据）
st.subheader("4. 可疑记录筛选")
suspect_data = df_raw[df_raw["QUALITY_FLAG"] != "ok"]
st.dataframe(suspect_data)