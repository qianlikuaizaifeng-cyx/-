import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 设置页面标题
st.set_page_config(page_title="儿童腺病毒肺炎预测", page_icon="🏥", layout="centered")

# 页面标题
st.title("儿童腺病毒肺炎预测系统")
st.write("基于机器学习模型的腺病毒感染预测工具")

# 加载模型
try:
    rf_model = joblib.load('RandomForest_model.sav')
    xgb_model = joblib.load('XGBoost_model.sav')
    lgb_model = joblib.load('LightGBM_model.sav')
    model_loaded = True
except Exception as e:
    st.error(f"模型加载失败: {e}")
    model_loaded = False

# 输入表单
st.sidebar.header("输入患者信息")

# 热峰输入
fever_peak = st.sidebar.number_input("热峰 (摄氏度)", min_value=35.0, max_value=42.0, step=0.1, value=38.5)

# IgG输入
gg_level = st.sidebar.number_input("IgG水平", min_value=0.0, max_value=20.0, step=0.1, value=5.0)

# 合并感染输入
co_infection = st.sidebar.selectbox("合并感染", ["无", "有"], index=0)
co_infection = 1 if co_infection == "有" else 0

# 基础疾病输入
basic_disease = st.sidebar.selectbox("基础疾病", ["无", "有"], index=0)
basic_disease = 1 if basic_disease == "有" else 0

# 模型选择
model_choice = st.sidebar.selectbox("选择预测模型", ["XGBoost (推荐)", "LightGBM", "RandomForest", "Logistic回归"], index=0)

# 预测按钮
if st.sidebar.button("预测") and model_loaded:
    # 准备输入数据
    input_data = pd.DataFrame({
        '热峰': [fever_peak],
        'IgG': [gg_level],
        '合并感染': [co_infection],
        '基础疾病': [basic_disease]
    })
    
    # 根据选择的模型进行预测
    if model_choice == "XGBoost (推荐)":
        prediction = xgb_model.predict(input_data)[0]
        model_name = "XGBoost"
    elif model_choice == "LightGBM":
        prediction = lgb_model.predict(input_data)[0]
        model_name = "LightGBM"
    elif model_choice == "RandomForest":
        prediction = rf_model.predict(input_data)[0]
        model_name = "RandomForest"
    else:  # Logistic回归
        # 使用用户提供的回归系数
        def logistic_prediction(row):
            fever = 1  # 假设发热=1
            P = (-1.318 * fever) + (1.726 * row['合并感染']) + (0.657 * row['基础疾病']) + (0.867 * row['热峰']) + (0.149 * row['IgG']) - 35.538
            prob = 1 / (1 + np.exp(-P))
            return 1 if prob >= 0.5 else 0
        
        prediction = logistic_prediction(input_data.iloc[0])
        model_name = "Logistic回归"
    
    # 显示预测结果
    st.subheader("预测结果")
    st.write(f"使用模型: {model_name}")
    
    if prediction == 1:
        st.error("⚠️ 预测结果: 可能感染腺病毒")
    else:
        st.success("✅ 预测结果: 未感染腺病毒")
    
    # 显示输入参数
    st.subheader("输入参数")
    st.write(f"热峰: {fever_peak} °C")
    st.write(f"IgG水平: {gg_level}")
    st.write(f"合并感染: {'有' if co_infection == 1 else '无'}")
    st.write(f"基础疾病: {'有' if basic_disease == 1 else '无'}")

# 模型性能信息
st.subheader("模型性能")
performance_data = {
    "模型": ["XGBoost", "LightGBM", "RandomForest", "Logistic回归"],
    "准确率": ["85.00%", "80.00%", "70.00%", "60.00%"],
    "F1分数": ["0.8421", "0.8000", "0.6250", "0.4286"]
}
performance_df = pd.DataFrame(performance_data)
st.dataframe(performance_df, use_container_width=True)

# 说明信息
st.subheader("使用说明")
st.write("1. 在左侧输入患者的相关信息")
st.write("2. 选择预测模型（推荐使用XGBoost）")
st.write("3. 点击'预测'按钮获取预测结果")
st.write("4. 预测结果仅供参考，最终诊断请以医生判断为准")

# 免责声明
st.caption("免责声明: 本工具仅用于辅助诊断，不能替代专业医生的诊断和治疗建议。")