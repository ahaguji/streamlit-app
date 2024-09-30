import streamlit as st
import pandas as pd
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt

# 加载模型和标准化器
gbc_model = joblib.load('gbc_model222.pkl')
scaler = joblib.load('scaler222.pkl')

feature_names = ['DBIL', 'ALT', 'AST', 'HBsAg', 'HBeAg', 'DNA', 'PCT', 'GGT']

# 创建Streamlit应用
st.title('FibroPredict Kids')
st.write('This application can help you predict the probability of significant liver fibrosis in children with CHB')
# 创建用户输入控件
DBIL = st.number_input('DBIL (umol/L)', min_value=0.00, max_value=100.00, value=0.00, step=0.01, key='DBIL')
ALT = st.number_input('ALT(IU/L)', min_value=0.00, max_value=1000.00, value=0.00, step=0.01, key='ALT')
AST = st.number_input('AST (IU/L)', min_value=0.00, max_value=1000.00, value=0.00, step=0.01, key='AST')
HBsAg = st.number_input('HBsAg (IU/mL)', min_value=0.000, max_value=60000.000, value=0.000, step=0.001, key='HBsAg')
HBeAg = st.number_input('HBeAg (COI)', min_value=0.000, max_value=10000.000, value=0.000, step=0.001, key='HBeAg')
DNA = st.number_input('DNA (Copies/mL)', min_value=0.00, max_value=52000.00, value=0.00, step=0.01, key='DNA')
PCT = st.number_input('PCT (%)', min_value=0.00, max_value=100.00, value=0.00, step=0.01, key='PCT')
GGT = st.number_input('GGT (IU/L)', min_value=0.00, max_value=200.00, value=0.00, step=0.01, key='GGT')

# 计算log10(DNA)值
log10_DNA = np.log10(DNA) if DNA > 0 else 0

feature_values = [DBIL, ALT, AST, HBsAg, HBeAg, log10_DNA, PCT, GGT]
features = np.array([feature_values])

# 创建一个按钮进行预测
if st.button('Predict'):
    # 检查是否所有输入都已经提供
    if DBIL == 0 or ALT == 0 or AST == 0 or HBsAg == 0 or HBeAg == 0 or DNA == 0 or PCT == 0 or GGT == 0:
        st.write("Please fill in all fields")
    else:
    # 获取用户输入并创建数据框
     user_data = pd.DataFrame({
        'DBIL': [DBIL],
        'ALT': [ALT],
        'AST': [AST],
        'HBsAg': [HBsAg],
        'HBeAg': [HBeAg],
        'DNA': [log10_DNA],
        'PCT': [PCT],
        'GGT': [GGT]
    })
    
    # 对用户输入的数据进行标准化处理
    user_data_scaled = scaler.transform(user_data)
    
    # 进行预测
    prediction_prob = gbc_model.predict_proba(user_data_scaled,)[0, 1]
    
    # 显示预测结果
    st.write(f'The probability of significant liver fibrosis is: {prediction_prob * 100:.2f}%')
    # Generate advice based on prediction results    
    if prediction_prob >=0.38:        
        advice = (            f'According to our model, CHB children with a predicted probability greater than 38% have a high risk of significant liver fibrosis. '            
                              f'The model predicts that the probability of having heart significant liver fibrosis is {prediction_prob * 100:.2f}%. '           
                              'While this is just an estimate, it suggests that this patient may be at significant risk. '           
                              'I recommend that this patient undergo a liver biopsy as soon as possible for further evaluation and '         
                              'to ensure accurate diagnosis and necessary treatment.'        )    
    else:        
        advice = (            f'According to our model, CHB children with a predicted probability greater than 38% have a high risk of significant liver fibrosis. '       
                              f'The model predicts that your probability of not having heart disease is {prediction_prob * 100:.2f}%. '           
                              'However, maintaining a healthy lifestyle is still very important.'            
                              'I recommend regular check-ups to monitor your liver health, '            
                              'and to seek medical advice promptly if you experience any symptoms.'        )
    st.write(advice)
    # Calculate SHAP values and display force plot    
    explainer = shap.TreeExplainer(gbc_model)    
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))
    shap.force_plot(explainer.expected_value, shap_values[0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)    
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png")