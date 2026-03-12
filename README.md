# 儿童腺病毒肺炎预测系统

基于机器学习模型的腺病毒感染预测工具，使用Streamlit构建的网页应用。

## 功能特性

- 支持输入热峰、IgG水平、合并感染和基础疾病等参数
- 提供4个预测模型：XGBoost、LightGBM、RandomForest和Logistic回归
- 实时显示预测结果和模型性能
- 响应式设计，支持不同设备访问

## 技术栈

- Python 3.13
- Streamlit 1.55.0
- Pandas 2.3.3
- NumPy 2.4.3
- Scikit-learn 1.8.0
- XGBoost 3.2.0
- LightGBM 4.6.0

## 本地运行

1. 安装依赖
   ```bash
   pip install -r requirements.txt
   ```

2. 运行应用
   ```bash
   streamlit run 腺病毒肺炎预测应用.py
   ```

3. 在浏览器中访问 http://localhost:8501

## 部署到Streamlit Cloud

1. 在GitHub上创建一个新仓库
2. 将以下文件上传到仓库：
   - 腺病毒肺炎预测应用.py
   - requirements.txt
   - .gitignore
   - README.md
3. 访问 https://share.streamlit.io/ 并连接到GitHub仓库
4. 选择主分支和主文件（腺病毒肺炎预测应用.py）
5. 点击"Deploy"按钮进行部署

## 模型性能

| 模型 | 准确率 | F1分数 |
|------|-------|--------|
| XGBoost | 85.00% | 0.8421 |
| LightGBM | 80.00% | 0.8000 |
| RandomForest | 70.00% | 0.6250 |
| Logistic回归 | 60.00% | 0.4286 |

## 免责声明

本工具仅用于辅助诊断，不能替代专业医生的诊断和治疗建议。
