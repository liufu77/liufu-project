import streamlit as st
import math
import time

# ==========页面配置&标题==========
st.set_page_config(page_title="元素人格测试仪", page_icon="🔮")
st.title("⚗️赛博炼金术士｜全维度元素人格测试仪")
st.caption("多维度性格采样+欧式距离算法自动匹配本命元素")
st.divider()

# ==========用户基础信息==========
user_name = st.text_input("输入炼金代号：", value="见习炼金术士")
# 新增性别选择
sex = st.radio("选择炼金派系：", ["火系炼金师", "水系炼金师", "无派系散修"], horizontal=True)
st.divider()

# ==========三维属性定义【行动力｜冷静度｜冒险度】P5三维拓展==========
action = 0    # 行动力
calm = 0      # 冷静度
adventure = 0 # 冒险度（新增第三维度）

# ==========5道测试单选题目（拓展题目数量）==========
st.subheader("📜五项人格测试问卷")
q1 = st.radio("1.发现陌生密闭洞穴？",["直接闯入","原地观察","找人结伴探索"])
q2 = st.radio("2.团队任务分配？",["主动带队","稳妥补位","创意变通"])
q3 = st.radio("3.突发意外故障？",["立刻补救","冷静复盘","换方案绕开"])
q4 = st.radio("4.空闲修炼选择？",["外出冒险历练","闭关看书炼金","随机随缘"])
q5 = st.radio("5.遇到从没见过的魔法生物？",["主动上前试探","远距离观望","绕道避开"])

# ==========选项对应计分规则==========
# 题目1
if q1 == "直接闯入":action +=3;adventure +=2
elif q1 == "原地观察":calm +=3
else:action +=1;calm +=1;adventure +=1
# 题目2
if q2 == "主动带队":action +=3
elif q2 == "稳妥补位":calm +=3
else:adventure +=3
# 题目3
if q3 == "立刻补救":action +=3
elif q3 == "冷静复盘":calm +=3
else:adventure +=3
# 题目4
if q4 == "外出冒险历练":adventure +=3
elif q4 == "闭关看书炼金":calm +=3
else:action +=2;adventure +=1
# 题目5
if q5 == "主动上前试探":adventure +=3;action +=2
elif q5 == "远距离观望":calm +=3
else:calm +=2;adventure +=1

# 用户三维坐标
user_vec = [action, calm, adventure]

# ==========字典存储全元素模板库 P6字典+循环==========
element_profiles = {
    "🔥烈焰型": [9,2,6],
    "💧潮汐型": [2,9,3],
    "🌪️风暴型": [7,4,9],
    "🪨大地型": [3,8,2],
    "⚡惊雷均衡型": [6,6,6]
}
st.divider()

# ==========距离计算+打擂台筛选最优模板函数==========
def get_best_result(user_data, profile_dict):
    min_distance = 99999.0
    best_name = ""
    all_dist_info = {} # 存储全部元素距离用于展示
    for elem_name, elem_vec in profile_dict.items():
        # 三维欧式距离公式
        dist = math.sqrt((user_data[0]-elem_vec[0])**2 + (user_data[1]-elem_vec[1])**2 + (user_data[2]-elem_vec[2])**2)
        all_dist_info[elem_name] = round(dist,2)
        # 擂台比武更新最小值
        if dist < min_distance:
            min_distance = dist
            best_name = elem_name
    return best_name, min_distance, all_dist_info

# ==========提交按钮触发计算==========
if st.button("🔍水晶球占卜·匹配本命元素", type="primary"):
    # 加载动画
    with st.spinner("水晶球运算中，正在解析人格数据..."):
        time.sleep(2)
    best_elem, min_d, all_data = get_best_result(user_vec, element_profiles)

    # 数据展示
    st.success(f"【{user_name}】最终本命元素：{best_elem}")
    st.info(f"三维属性得分：行动力{action}｜冷静度{calm}｜冒险度{adventure}")

    # 逐个输出所有元素距离
    st.subheader("📊全元素相似度距离明细（数值越小越匹配）")
    for name,d in all_data.items():
        st.write(f"{name}：距离 = {d}")

    # 不同元素对应不同特效
    if "烈焰" in best_elem or "惊雷" in best_elem:
        st.balloons()
    elif "潮汐" in best_elem or "大地" in best_elem:
        st.snow()
    st.toast("✅炼金档案已自动保存！")

# ==========侧边栏拓展：手动微调属性滑块 P5滑块组件==========
with st.sidebar:
    st.subheader("⚙️手动微调属性（自定义测试）")
    fix_act = st.slider("行动力",0,12,action)
    fix_cal = st.slider("冷静度",0,12,calm)
    fix_adv = st.slider("冒险度",0,12,adventure)
    if st.button("使用自定义分数重新测算"):
        new_user = [fix_act,fix_cal,fix_adv]
        res_name,res_dis,all_dis = get_best_result(new_user,element_profiles)
        st.success(f"自定义测算结果：{res_name}")