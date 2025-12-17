#!/usr/bin/env python
# coding: utf-8

# In[2]:


# app.py（部署到公网的代码，需与模型文件lgb_occupation_stress_model.pkl放在同一目录）
import streamlit as st
import joblib
import pandas as pd
import os

# -------------------------- 页面配置与样式 --------------------------
st.set_page_config(page_title="职业紧张风险筛查", layout="centered")

# 自定义样式
st.markdown(
    """
    <div style='text-align: center; margin-bottom: 10px;'>
        <h3 style='color: #2E4057; font-weight: bold;'>互联网员工职业紧张风险快速筛查评估系统</h3>
    </div>
    <div style='text-align: left; margin-bottom: 30px;'>
        <p style='color: #666; font-size: 14px; text-indent: 2em;'>系统通过收集工作时长、疲劳积蓄程度等信息，评估职业紧张程度等级（1-4级），是初步筛查工具而非正式医学诊断。填写便捷，结果可为调整工作生活节奏提供参考！</p >
    </div>
    <p style='font-size: 18px; font-weight: bold;'>请输入员工信息</p >
    """,
    unsafe_allow_html=True
)

# -------------------------- 加载训练好的模型 --------------------------
# 异常处理：确保模型文件存在
model_path = "hlw_occupation_stress_model.pkl"
if not os.path.exists(model_path):
    st.error(f"模型文件 {model_path} 不存在，请先运行train_model.py生成模型！")
    st.stop()

# 加载模型
model = joblib.load(model_path)

# -------------------------- 用户输入组件 --------------------------
# 1. 年龄
age_options = ['20-25','26-35','36-60']
age = st.radio("年龄", age_options, horizontal=False)
age = age_options.index(age) + 1

# 2. 教育程度
edu_options = ["初中及以下", "高中或中专", "大专或高职", "大学本科", "研究生及以上"]
edu = st.radio("教育程度", edu_options, horizontal=False)
edu = edu_options.index(edu) + 1

# 3. 收入水平
income_options = ["少于3000元", "3000-4999元", "5000-6999元", "7000-8999元", "9000-10999元", "11000元及以上"]
income = st.radio("收入水平", income_options, horizontal=False)
income = income_options.index(income) + 1

# 4. 工龄
work_age_options = ['1-10','11-20','≥21']
work_age = st.radio("工龄（年）", work_age_options, horizontal=False)
work_age = work_age_options.index(work_age) + 1

# 5. 周均工作时间
weekly_hours_options = ['35~40','41~48','49~54','≥55']
weekly_hours = st.radio("周均工作时间（小时）", weekly_hours_options, horizontal=False)
weekly_hours = weekly_hours_options.index(weekly_hours) + 1

# 6. 日均加班时间
daily_overtime = st.number_input("日均加班时间（小时）", value=1, min_value=0, max_value=10)

# 7. 是否轮班
shift = st.radio("是否轮班", ["否", "是"], horizontal=False)
shift = 1 if shift == "是" else 0

# 8. 是否夜班
night_shift = st.radio("是否夜班", ["否", "是"], horizontal=False)
night_shift = 1 if night_shift == "是" else 0

# 9. 是否吸烟
smoke = st.radio("是否吸烟", ["是", "否"], horizontal=False)
smoke = 1 if smoke == "是" else 0

# 10. 高强度锻炼
high_exercise_options = ["无", "偶尔，1~3次/月", "有，1~3次/周", "经常，4~6次/周", "每天"]
high_exercise = st.radio("高强度锻炼（持续至少30分钟）", high_exercise_options, horizontal=False)
high_exercise = high_exercise_options.index(high_exercise) + 1

# 11. 生活满意度（改为1~10分自评分，映射到原标签0/1）
st.markdown("<p style='font-size: 16px; margin-top: 15px;'>生活满意度（1~10分，1=最低，10=最高）：</p>", unsafe_allow_html=True)
life_satisfaction_score = st.slider(
    "生活满意度评分",  # 滑块标签（仅在侧边显示，不影响页面样式）
    min_value=1, 
    max_value=10, 
    value=5,  # 默认值
    label_visibility="collapsed"  # 隐藏滑块默认标签，用自定义markdown显示
)
# 评分映射到原标签：1-5分→0（较低），6-10分→1（较高）
life_satisfaction = 0 if 1 <= life_satisfaction_score <= 5 else 1

# 12. 疲劳积蓄程度（改为1~10分自评分，映射到原标签1/2/3/4）
st.markdown("<p style='font-size: 16px; margin-top: 15px;'>疲劳积蓄程度（1~10分，1=最低，10=最高）：</p>", unsafe_allow_html=True)
fatigue_score = st.slider(
    "疲劳积蓄程度评分",
    min_value=1,
    max_value=10,
    value=5,
    label_visibility="collapsed"
)
# 评分映射到原标签：1→1，2-3→2，4-5→3，6-10→4
if fatigue_score == 1:
    fatigue_degree = 1
elif 2 <= fatigue_score <= 3:
    fatigue_degree = 2
elif 4 <= fatigue_score <= 5:
    fatigue_degree = 3
else:  # 6-10分
    fatigue_degree = 4

# -------------------------- 数据整理与预测 --------------------------
# 收集输入特征
input_data = pd.DataFrame({
    '疲劳积蓄程度': [fatigue_degree],
    '周均工作时间': [weekly_hours],
    '生活满意度': [life_satisfaction],
    '教育程度': [edu],
    '收入水平': [income],
    '日均加班时间': [daily_overtime],
    '是否轮班': [shift],
    '年龄': [age],
    '工龄': [work_age],
    '是否夜班': [night_shift],
    '是否吸烟': [smoke],
    '高强度锻炼': [high_exercise]
})

# 预测按钮
if st.button("进行职业紧张风险评估"):
    prediction = model.predict(input_data)[0]
    prediction = int(prediction)  # 确保为整数
    st.subheader("评估结果")
    # 不同等级对应不同样式和说明
    if prediction == 1:
        st.success(f"该员工的职业紧张程度等级为：{prediction}级")
        st.write("**说明**：低紧张程度，当前工作生活节奏较为健康，建议保持。")
    elif prediction == 2:
        st.info(f"该员工的职业紧张程度等级为：{prediction}级")
        st.write("**说明**：中低紧张程度，建议适当放松，减少不必要的加班。")
    elif prediction == 3:
        st.warning(f"该员工的职业紧张程度等级为：{prediction}级")
        st.write("**说明**：中高紧张程度，建议调整工作节奏，增加休息和锻炼时间。")
    elif prediction == 4:
        st.error(f"该员工的职业紧张程度等级为：{prediction}级")
        st.write("**说明**：高紧张程度，建议及时调整工作状态，必要时寻求专业心理疏导。")

# -------------------------- 模型信息与等级说明 --------------------------
st.subheader("模型说明")
st.write("模型基于LightGBM多分类算法训练，适用于互联网行业员工职业紧张程度评估。")
# 若需展示准确率，可将训练时的准确率保存为txt文件，再读取（避免重新训练）
# 示例：可在train_model.py中保存准确率，这里读取
# if os.path.exists("accuracy.txt"):
#     with open("accuracy.txt", "r") as f:
#         accuracy = f.read()
#     st.write(f"模型在测试集上的准确率：{accuracy}")

# 职业紧张程度等级说明
st.markdown(
    """
    <div style='margin-top: 20px; padding: 10px; background-color: #f0f2f6; border-radius: 5px;'>
        <p style='font-size: 14px; color: #333;'><strong>职业紧张程度等级说明：</strong></p>
        <p style='font-size: 13px; color: #666;'>1级：低紧张程度 | 2级：中低紧张程度 | 3级：中高紧张程度 | 4级：高紧张程度</p>
        <p style='font-size: 13px; color: #666; margin-top: 8px;'><strong>评分映射说明：</strong></p>
        <p style='font-size: 12px; color: #666;'>• 疲劳积蓄程度：1分→1级，2-3分→2级，4-5分→3级，6-10分→4级</p>
        <p style='font-size: 12px; color: #666;'>• 生活满意度：1-5分→较低（0），6-10分→较高（1）</p>
    </div>
    """,
    unsafe_allow_html=True
)


# In[ ]:




