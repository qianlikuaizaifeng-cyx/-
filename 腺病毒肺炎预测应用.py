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

# 合并感染（并发症）输入
co_infection = st.sidebar.selectbox("并发症（合并感染）", ["无", "有"], index=0)
co_infection = 1 if co_infection == "有" else 0

# 基础疾病输入
basic_disease = st.sidebar.selectbox("基础疾病", ["无", "有"], index=0)
basic_disease = 1 if basic_disease == "有" else 0

# 预测按钮
if st.sidebar.button("预测") and model_loaded:
    # 准备输入数据
    input_data = pd.DataFrame({
        '热峰': [fever_peak],
        '合并感染': [co_infection],
        '基础疾病': [basic_disease]
    })
    
    # 使用所有模型进行预测
    predictions = {}
    
    # XGBoost模型
    predictions["XGBoost"] = xgb_model.predict(input_data)[0]
    
    # LightGBM模型
    predictions["LightGBM"] = lgb_model.predict(input_data)[0]
    
    # RandomForest模型
    predictions["RandomForest"] = rf_model.predict(input_data)[0]
    
    # Logistic回归模型（使用新公式）
    def logistic_prediction(row):
        # Logistic(P) = 0.810×热峰（℃）+ 0.705×并发症（1=有，0=无）+ 0.890×基础疾病（1=有，0=无）- 30.205
        P = (0.810 * row['热峰']) + (0.705 * row['合并感染']) + (0.890 * row['基础疾病']) - 30.205
        prob = 1 / (1 + np.exp(-P))
        return 1 if prob >= 0.5 else 0
    
    predictions["Logistic回归"] = logistic_prediction(input_data.iloc[0])
    
    # 显示预测结果
    st.subheader("预测结果")
    
    # 创建结果表格
    result_data = []
    for model_name, prediction in predictions.items():
        result = "可能感染腺病毒" if prediction == 1 else "未感染腺病毒"
        result_data.append([model_name, result])
    
    result_df = pd.DataFrame(result_data, columns=["模型", "预测结果"])
    st.dataframe(result_df, use_container_width=True)
    
    # 显示输入参数
    st.subheader("输入参数")
    st.write(f"热峰: {fever_peak} °C")
    st.write(f"并发症（合并感染）: {'有' if co_infection == 1 else '无'}")
    st.write(f"基础疾病: {'有' if basic_disease == 1 else '无'}")

# 模型性能信息
st.subheader("模型性能")
performance_data = {
    "模型": ["XGBoost", "LightGBM", "RandomForest", "Logistic回归"],
    "准确率": ["65.00%", "70.00%", "70.00%", "60.00%"],
    "精确率": ["71.43%", "75.00%", "75.00%", "55.56%"],
    "召回率": ["50.00%", "60.00%", "60.00%", "100.00%"],
    "F1分数": ["0.5882", "0.6667", "0.6667", "0.7143"]
}
performance_df = pd.DataFrame(performance_data)
st.dataframe(performance_df, use_container_width=True)

# 说明信息
st.subheader("使用说明")
st.write("1. 在左侧输入患者的相关信息")
st.write("2. 点击'预测'按钮获取预测结果")
st.write("3. 预测结果仅供参考，最终诊断请以医生判断为准")

# 免责声明

st.caption("免责声明: 本工具仅用于辅助诊断，不能替代专业医生的诊断和治疗建议。")
