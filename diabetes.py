import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import numpy as np

# 한글 폰트 설정
plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

# 데이터 로딩 및 전처리
data = pd.read_csv("diabest_cut.csv")
data['bmi'] = data['bmi'].fillna(data['bmi'].mean())
data['age'] = data['age'].fillna(data['age'].mean())
data['diabetes_risk'] = ((data['bmi'] >= 30) | (data['HbA1c_level'] >= 6.5)).astype(int)
data['high_glucose_risk'] = (data['blood_glucose_level'] >= 180).astype(int)

# 인코딩
gender_map = {'Male': 1, 'Female': 0}
data['gender'] = data['gender'].map(gender_map)
data['smoking_history'] = data['smoking_history'].replace({'ever': 'former', 'not current': 'former', 'No Info': 'never'})
le_smoke = LabelEncoder()
data['smoking_history'] = le_smoke.fit_transform(data['smoking_history'])

# 학습 데이터 분리
X = data.drop(columns=['diabetes'])
y = data['diabetes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 스케일링 및 SMOTE
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# Streamlit 인터페이스 구성
st.title("📊 당뇨 예측 시스템")

tab1, tab2, tab3, tab4 = st.tabs(["데이터 탐색", "예측 결과", "모델 성능", "인사이트 요약"])

# 사이드바 입력
st.sidebar.header("👤 신규 이용자 정보 입력")
age = st.sidebar.number_input("나이", 18, 100, 30)
bmi = st.sidebar.number_input("BMI", 10.0, 50.0, 25.0)
HbA1c_level = st.sidebar.number_input("HbA1c 수치", 4.0, 15.0, 5.5)
blood_glucose_level = st.sidebar.number_input("혈당 수치", 50, 300, 100)
gender = st.sidebar.selectbox("성별", ['여성', '남성'])
smoking = st.sidebar.selectbox("흡연 이력", ['No Info', 'ever', 'not current', 'never', 'former', 'current'])

# 탭 1: 데이터 탐색
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# EDA 섹션 시작
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# EDA 섹션 시작
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# EDA 섹션 시작
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# EDA 섹션 시작
with tab1:
    st.subheader("📊 데이터 탐색 (EDA)")
    
    # 데이터 샘플
    st.write("### 데이터 샘플")
    st.write(data.head())

    # EDA 설명
    st.write("### EDA 설명")
    st.markdown("""
    - 이 섹션에서는 각 변수의 분포를 살펴보고, 데이터 간 상관 관계를 분석하여 당뇨병의 위험 요인을 이해합니다.
    - 데이터의 주요 변수에 대한 시각화 및 인사이트를 제공합니다.
    """)

    # 변수 선택 및 최적화된 시각화 (expander로 감싸기)
    with st.expander("📈 변수별 시각화 보기"):
        selected_column = st.selectbox("보고 싶은 변수를 선택하세요:", 
                                       ['성별', '흡연 이력', '고혈압', '심장병', 'BMI', '나이', 'HbA1c 수준', '혈당 수치', '당뇨병 여부', '당뇨병 위험도', '고혈당 위험도'])
        
        # 변수 이름 매핑
        column_mapping = {
            '성별': 'gender',
            '흡연 이력': 'smoking_history',
            '고혈압': 'hypertension',
            '심장병': 'heart_disease',
            'BMI': 'bmi',
            '나이': 'age',
            'HbA1c 수준': 'HbA1c_level',
            '혈당 수치': 'blood_glucose_level',
            '당뇨병 여부': 'diabetes',
            '당뇨병 위험도': 'diabetes_risk',
            '고혈당 위험도': 'high_glucose_risk'
        }
        
        # 성별 (0, 1) 변수에 대한 시각화
        if selected_column == '성별':
            gender_counts = data[column_mapping['성별']].value_counts()
            
            # 파이 차트 그리기
            fig1, ax1 = plt.subplots(figsize=(8, 5))
            ax1.pie(gender_counts, labels=['여성', '남성'], autopct='%1.1f%%', startangle=90, colors=['lightblue', 'lightgreen'])
            ax1.set_title(f'{selected_column} 분포')
            st.pyplot(fig1)
            
            st.write("### 성별 분석 요약")
            st.markdown("""
            - 데이터에서 남성과 여성의 분포가 거의 비슷합니다.
            - 성별은 당뇨병 위험도에 큰 영향을 미치지 않는 변수일 수 있습니다.
            """)
        
        # 흡연 이력 (current, former, never) 변수에 대한 시각화
        elif selected_column == '흡연 이력':
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            sns.countplot(x=data[column_mapping[selected_column]], palette='Set3', ax=ax2)
            ax2.set_title(f'{selected_column} 분포')
            ax2.set_xlabel(f'{selected_column}')
            ax2.set_ylabel('빈도')
            ax2.set_xticklabels(['현재 흡연', '흡연 경험 있음', '흡연 안함'], rotation=0)
            st.pyplot(fig2)
            st.write("### 흡연 이력 분석 요약")
            st.markdown("""
            - 흡연 경험이 있는 사람들보다 흡연하지 않는 사람이 더 많은 경향이 있습니다.
            - 흡연이 당뇨병에 미치는 영향은 다른 변수들과 함께 추가 분석이 필요할 수 있습니다.
            """)

        # 고혈압, 심장병 변수에 대한 시각화
        elif selected_column in ['고혈압', '심장병']:
            fig3, ax3 = plt.subplots(figsize=(8, 5))
            sns.countplot(x=data[column_mapping[selected_column]], palette='Set1', ax=ax3)
            ax3.set_title(f'{selected_column} 분포')
            ax3.set_xlabel(f'{selected_column}')
            ax3.set_ylabel('빈도')
            ax3.set_xticklabels(['0', '1'], rotation=0)
            st.pyplot(fig3)
            st.write(f"### {selected_column} 분석 요약")
            st.markdown("""
            - 고혈압과 심장병은 당뇨병 발생에 중요한 변수입니다. 이 변수들에 대한 추가 분석이 필요합니다.
            """)

        # BMI에 대한 박스플롯 시각화
        elif selected_column == 'BMI':
            fig4, ax4 = plt.subplots(figsize=(8, 5))
            sns.histplot(data[column_mapping[selected_column]], kde=True, color='lightgreen', ax=ax4)
            ax4.set_title(f'{selected_column} 분포')  # 타이틀 통일성 있게 수정
            ax4.set_xlabel(f'{selected_column}')
            ax4.set_ylabel('빈도')
            st.pyplot(fig4)
            
            st.write("### BMI 분석 요약")
            st.markdown("""
            - BMI의 분포는 일부 높은 값들에 이상치가 존재하는 것을 확인할 수 있습니다. 고BMI 값은 당뇨병의 위험 요소로 작용할 수 있습니다.
            """)


        # 나이에 대한 히스토그램
        elif selected_column == '나이':
            fig5, ax5 = plt.subplots(figsize=(8, 5))
            sns.histplot(data[column_mapping[selected_column]], kde=True, color='lightgreen', ax=ax5)
            ax5.set_title(f'{selected_column} 분포')
            ax5.set_xlabel(f'{selected_column}')
            ax5.set_ylabel('빈도')
            st.pyplot(fig5)
            st.write("### 나이 분석 요약")
            st.markdown("""
            - 나이는 당뇨병의 주요 위험 요소 중 하나입니다. 나이가 많을수록 당뇨병의 위험이 높아질 수 있습니다.
            """)

        # HbA1c 수준에 대한 히스토그램
        elif selected_column == 'HbA1c 수준':
            fig6, ax6 = plt.subplots(figsize=(8, 5))
            sns.histplot(data[column_mapping[selected_column]], kde=True, color='lightcoral', ax=ax6)
            ax6.set_title(f'{selected_column} 분포')
            ax6.set_xlabel(f'{selected_column}')
            ax6.set_ylabel('빈도')
            st.pyplot(fig6)
            st.write("### HbA1c 수준 분석 요약")
            st.markdown("""
            - HbA1c 수치는 당뇨병 환자의 혈당 조절 상태를 나타냅니다. 높은 HbA1c 수치는 당뇨병이 잘 관리되지 않음을 의미합니다.
            """)

        # 혈당 수치에 대한 히스토그램
        elif selected_column == '혈당 수치':
            fig7, ax7 = plt.subplots(figsize=(8, 5))
            sns.histplot(data[column_mapping[selected_column]], kde=True, color='lightblue', ax=ax7)
            ax7.set_title(f'{selected_column} 분포')
            ax7.set_xlabel(f'{selected_column}')
            ax7.set_ylabel('빈도')
            st.pyplot(fig7)
            st.write("### 혈당 수치 분석 요약")
            st.markdown("""
            - 혈당 수치는 당뇨병의 상태를 나타내는 중요한 변수입니다. 높을수록 당뇨병 위험이 증가합니다.
            """)

        # 당뇨병 여부에 대한 막대 차트
        elif selected_column == '당뇨병 여부':
            fig8, ax8 = plt.subplots(figsize=(8, 5))
            sns.countplot(x=data[column_mapping[selected_column]], palette='coolwarm', ax=ax8)
            ax8.set_title(f'{selected_column} 분포')
            ax8.set_xlabel(f'{selected_column}')
            ax8.set_ylabel('빈도')
            ax8.set_xticklabels(['없음', '있음'], rotation=0)
            st.pyplot(fig8)
            st.write("### 당뇨병 여부 분석 요약")
            st.markdown("""
            - 당뇨병 여부는 당뇨병 위험도와 직접적인 관련이 있습니다.
            """)

        # 당뇨병 위험도나 고혈당 위험도에 대한 막대 차트
        elif selected_column == '당뇨병 위험도' or selected_column == '고혈당 위험도':
            fig9, ax9 = plt.subplots(figsize=(8, 5))
            sns.countplot(x=data[column_mapping[selected_column]], palette='Set2', ax=ax9)
            ax9.set_title(f'{selected_column} 분포')
            ax9.set_xlabel(f'{selected_column}')
            ax9.set_ylabel('빈도')
            ax9.set_xticklabels(['위험 없음', '위험 있음'], rotation=0)
            st.pyplot(fig9)
            st.write(f"### {selected_column} 분석 요약")
            st.markdown("""
            - 당뇨병과 고혈당 위험도는 밀접하게 연관되어 있습니다. 이 변수들은 당뇨병 예방에 중요한 지표로 활용될 수 있습니다.
            """)

    # 상관 행렬 (expander로 감싸기)
    with st.expander("🔍 상관 행렬 보기"):
        st.write("### 변수 간 상관 관계 (상관 행렬)")
        fig10 = plt.figure(figsize=(10, 8))
        corr = data.corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, annot_kws={'size': 10})
        plt.title('상관 행렬 (Correlation Matrix) 분포', fontsize=15)
        st.pyplot(fig10)
        
        # 인사이트 요약: 상관 행렬
        st.write("### 상관 관계 분석")
        st.markdown("""
        - **BMI**와 **HbA1c_level**은 높은 상관 관계를 보입니다. 이는 비만이 당뇨와 밀접하게 연관되어 있음을 의미합니다.
        - **혈당 수치**와 **HbA1c level** 간에도 높은 상관관계를 보이며, 이는 혈당 수치가 높을수록 HbA1c가 증가함을 의미합니다.
        - **연령**은 다른 변수들과 약한 상관 관계를 보이지만, 나이가 들수록 당뇨 위험이 증가하는 경향을 보입니다.
        """)


    # 변수별 시각화 선택
    with st.expander("📈 변수별 시각화 보기"):
        # 차트 선택
        chart_option = st.selectbox("보고 싶은 변수를 선택하세요:", 
                                   ['흡연 여부와 연령대에 따른 당뇨병 유병률', 
                                    'BMI와 연령대에 따른 당뇨병 유병률',
                                    '고혈압 여부와 연령대에 따른 당뇨병 유병률',
                                    '심장병 여부와 연령대에 따른 당뇨병 유병률',
                                    'BMI와 흡연 여부에 따른 당뇨병 유병률',
                                    'BMI와 고혈압에 따른 당뇨병 유병률',
                                    'BMI와 심장병 여부에 따른 당뇨병 유병률'])
        
        # 흡연 여부와 연령대에 따른 당뇨병 유병률
        if chart_option == '흡연 여부와 연령대에 따른 당뇨병 유병률':
            # 연령대 범주화
            bins = [18, 30, 40, 50, 60, 70, 80, 100]
            labels = ['18-30', '31-40', '41-50', '51-60', '61-70', '71-80', '80+']
            data['age_group'] = pd.cut(data['age'], bins=bins, labels=labels, right=False)
    
            # 그룹화 및 평균
            df_smoking_age = data.groupby(['age_group', 'smoking_history'])['diabetes'].mean().reset_index()
    
            # 시각화
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='age_group', y='diabetes', hue='smoking_history', data=df_smoking_age, ax=ax, palette='Set2')
            ax.set_title('흡연 여부와 연령대에 따른 당뇨병 유병률', fontsize=16)
            ax.set_xlabel('연령대')
            ax.set_ylabel('당뇨병 유병률')
            st.pyplot(fig)
    
            st.write("""
            - 흡연 여부와 연령대가 당뇨병 유병률에 미치는 영향을 분석할 수 있습니다.
            - 연령대가 증가할수록 당뇨병 유병률이 높아지며, 흡연 여부가 유병률에 영향을 미치는 경향이 있을 수 있습니다.
            """)
    
        # BMI와 연령대에 따른 당뇨병 유병률
        elif chart_option == 'BMI와 연령대에 따른 당뇨병 유병률':
            # 연령대 범주화
            bins = [18, 30, 40, 50, 60, 70, 80, 100]
            labels = ['18-30', '31-40', '41-50', '51-60', '61-70', '71-80', '80+']
            data['age_group'] = pd.cut(data['age'], bins=bins, labels=labels, right=False)
    
            # BMI 범주화
            bins_bmi = [0, 18.5, 24.9, 29.9, 40]
            labels_bmi = ['저체중', '정상', '과체중', '비만']
            data['bmi_category'] = pd.cut(data['bmi'], bins=bins_bmi, labels=labels_bmi)
    
            # 그룹화 및 평균
            df_bmi_age = data.groupby(['age_group', 'bmi_category'])['diabetes'].mean().reset_index()
    
            # 시각화
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='age_group', y='diabetes', hue='bmi_category', data=df_bmi_age, ax=ax, palette='coolwarm')
            ax.set_title('BMI와 연령대에 따른 당뇨병 유병률', fontsize=16)
            ax.set_xlabel('연령대')
            ax.set_ylabel('당뇨병 유병률')
            st.pyplot(fig)
    
            st.write("""
            - BMI 범주 전체에 대해 당뇨병 유병률을 비교할 수 있습니다.
            - 비만일수록, 그리고 연령이 증가할수록 유병률이 높아질 수 있습니다.
            """)

        # 고혈압 여부와 연령대에 따른 당뇨병 유병률
        elif chart_option == '고혈압 여부와 연령대에 따른 당뇨병 유병률':
            # 연령대 범주화
            bins = [18, 30, 40, 50, 60, 70, 80, 100]
            labels = ['18-30', '31-40', '41-50', '51-60', '61-70', '71-80', '80+']
            data['age_group'] = pd.cut(data['age'], bins=bins, labels=labels, right=False)
    
            # 그룹화 및 평균
            df_hypertension_age = data.groupby(['age_group', 'hypertension'])['diabetes'].mean().reset_index()
    
            # 시각화
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='age_group', y='diabetes', hue='hypertension', data=df_hypertension_age, ax=ax, palette='coolwarm')
            ax.set_title('고혈압 여부와 연령대에 따른 당뇨병 유병률', fontsize=16)
            ax.set_xlabel('연령대')
            ax.set_ylabel('당뇨병 유병률')
            st.pyplot(fig)
    
            st.write("""
            - 고혈압 여부와 연령대가 당뇨병 유병률에 미치는 영향을 분석할 수 있습니다.
            """)
    
        # 심장병 여부와 연령대에 따른 당뇨병 유병률
        elif chart_option == '심장병 여부와 연령대에 따른 당뇨병 유병률':
            # 연령대 범주화
            bins = [18, 30, 40, 50, 60, 70, 80, 100]
            labels = ['18-30', '31-40', '41-50', '51-60', '61-70', '71-80', '80+']
            data['age_group'] = pd.cut(data['age'], bins=bins, labels=labels, right=False)
    
            # 그룹화 및 평균
            df_heart_disease_age = data.groupby(['age_group', 'heart_disease'])['diabetes'].mean().reset_index()
    
            # 시각화
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='age_group', y='diabetes', hue='heart_disease', data=df_heart_disease_age, ax=ax, palette='Set2')
            ax.set_title('심장병 여부와 연령대에 따른 당뇨병 유병률', fontsize=16)
            ax.set_xlabel('연령대')
            ax.set_ylabel('당뇨병 유병률')
            st.pyplot(fig)
    
            st.write("""
            - 심장병 여부와 연령대가 당뇨병 유병률에 미치는 영향을 분석할 수 있습니다.
            """)
    
        # BMI와 흡연 여부에 따른 당뇨병 유병률
        elif chart_option == 'BMI와 흡연 여부에 따른 당뇨병 유병률':
            # BMI 범주화
            bins_bmi = [0, 18.5, 24.9, 29.9, 40]
            labels_bmi = ['저체중', '정상', '과체중', '비만']
            data['bmi_category'] = pd.cut(data['bmi'], bins=bins_bmi, labels=labels_bmi)
    
            # 그룹화 및 평균
            df_bmi_smoking = data.groupby(['bmi_category', 'smoking_history'])['diabetes'].mean().reset_index()
    
            # 시각화
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='bmi_category', y='diabetes', hue='smoking_history', data=df_bmi_smoking, ax=ax, palette='coolwarm')
            ax.set_title('BMI와 흡연 여부에 따른 당뇨병 유병률', fontsize=16)
            ax.set_xlabel('BMI 범주')
            ax.set_ylabel('당뇨병 유병률')
            st.pyplot(fig)
    
            st.write("""
            - BMI와 흡연 여부에 따른 당뇨병 유병률을 비교할 수 있습니다.
            """)
    
        # BMI와 고혈압에 따른 당뇨병 유병률
        elif chart_option == 'BMI와 고혈압에 따른 당뇨병 유병률':
            # BMI 범주화
            bins_bmi = [0, 18.5, 24.9, 29.9, 40]
            labels_bmi = ['저체중', '정상', '과체중', '비만']
            data['bmi_category'] = pd.cut(data['bmi'], bins=bins_bmi, labels=labels_bmi)
    
            # 그룹화 및 평균
            df_bmi_hypertension = data.groupby(['bmi_category', 'hypertension'])['diabetes'].mean().reset_index()
    
            # 시각화
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='bmi_category', y='diabetes', hue='hypertension', data=df_bmi_hypertension, ax=ax, palette='Set2')
            ax.set_title('BMI와 고혈압에 따른 당뇨병 유병률', fontsize=16)
            ax.set_xlabel('BMI 범주')
            ax.set_ylabel('당뇨병 유병률')
            st.pyplot(fig)
    
            st.write("""
            - BMI와 고혈압에 따른 당뇨병 유병률을 비교할 수 있습니다.
            """)

        # BMI와 심장병 여부에 따른 당뇨병 유병률
        elif chart_option == 'BMI와 심장병 여부에 따른 당뇨병 유병률':
            # BMI 범주화
            bins_bmi = [0, 18.5, 24.9, 29.9, 40]
            labels_bmi = ['저체중', '정상', '과체중', '비만']
            data['bmi_category'] = pd.cut(data['bmi'], bins=bins_bmi, labels=labels_bmi)
        
            # 그룹화 및 평균
            df_bmi_heart = data.groupby(['bmi_category', 'heart_disease'])['diabetes'].mean().reset_index()
        
            # 시각화
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='bmi_category', y='diabetes', hue='heart_disease', data=df_bmi_heart, ax=ax, palette='PuBu')
            ax.set_title('BMI와 심장병 여부에 따른 당뇨병 유병률', fontsize=16)
            ax.set_xlabel('BMI 범주')
            ax.set_ylabel('당뇨병 유병률')
            st.pyplot(fig)
        
            st.write("""
            - BMI와 심장병 여부에 따른 당뇨병 유병률을 비교할 수 있습니다.
            - 심장병이 있는 사람에서 비만일수록 당뇨병 유병률이 높을 수 있습니다.
            """)







# 탭 2: 예측
# with tab2:
#     st.subheader("🩺 신규 환자 예측 결과")
    
#     gender_val = 0 if gender == '여성' else 1
#     smoking_clean = {'ever': 'former', 'not current': 'former', 'No Info': 'never'}.get(smoking, smoking)
#     smoking_val = le_smoke.transform([smoking_clean])[0]

#     new_data = pd.DataFrame({
#         'age': [age],
#         'bmi': [bmi],
#         'HbA1c_level': [HbA1c_level],
#         'blood_glucose_level': [blood_glucose_level],
#         'gender': [gender_val],
#         'smoking_history': [smoking_val],
#         'diabetes_risk': [int(bmi >= 30 or HbA1c_level >= 6.5)],
#         'high_glucose_risk': [int(blood_glucose_level >= 180)]
#     })

#     for col in X.columns:
#         if col not in new_data.columns:
#             new_data[col] = 0

#     new_data_scaled = scaler.transform(new_data[X.columns])

#     model_option = st.selectbox("📌 사용할 모델을 선택하세요", ["XGBoost", "Random Forest", "Logistic Regression"], key="model_choice")

#     if model_option == "XGBoost":
#         model = XGBClassifier(random_state=42)
#         param_grid = {'n_estimators': [100], 'max_depth': [3]}
#     elif model_option == "Random Forest":
#         model = RandomForestClassifier(random_state=42)
#         param_grid = {'n_estimators': [100], 'max_depth': [5]}
#     else:
#         model = LogisticRegression(max_iter=1000, random_state=42)
#         param_grid = {'C': [1], 'solver': ['lbfgs']}

#     grid = GridSearchCV(model, param_grid, cv=3, scoring='accuracy')
#     grid.fit(X_train_resampled, y_train_resampled)

#     new_pred = grid.best_estimator_.predict(new_data_scaled)
#     new_proba = grid.best_estimator_.predict_proba(new_data_scaled)[:, 1]

#     st.success(f"✅ 예측된 결과: **{'당뇨' if new_pred[0] == 1 else '비당뇨'}**")
#     st.info(f"📈 당뇨 확률: **{new_proba[0]:.2%}**")

with tab2:
    st.subheader("🩺 신규 환자 예측 결과")

    gender_val = 0 if gender == '여성' else 1
    smoking_clean = {'ever': 'former', 'not current': 'former', 'No Info': 'never'}.get(smoking, smoking)
    smoking_val = le_smoke.transform([smoking_clean])[0]

    new_data = pd.DataFrame({
        'age': [age],
        'bmi': [bmi],
        'HbA1c_level': [HbA1c_level],
        'blood_glucose_level': [blood_glucose_level],
        'gender': [gender_val],
        'smoking_history': [smoking_val],
        'diabetes_risk': [int(bmi >= 30 or HbA1c_level >= 6.5)],
        'high_glucose_risk': [int(blood_glucose_level >= 180)]
    })

    for col in X.columns:
        if col not in new_data.columns:
            new_data[col] = 0

    new_data_scaled = scaler.transform(new_data[X.columns])

    st.markdown("### 📌 사용할 모델을 선택하세요 (복수 선택 가능)")
    selected_models = st.multiselect("모델 선택", ["XGBoost", "Random Forest", "Logistic Regression"], default=["XGBoost"])

    for model_option in selected_models:
        if model_option == "XGBoost":
            model = XGBClassifier(random_state=42)
            param_grid = {'n_estimators': [100], 'max_depth': [3]}
        elif model_option == "Random Forest":
            model = RandomForestClassifier(random_state=42)
            param_grid = {'n_estimators': [100], 'max_depth': [5]}
        else:
            model = LogisticRegression(max_iter=1000, random_state=42)
            param_grid = {'C': [1], 'solver': ['lbfgs']}

        grid = GridSearchCV(model, param_grid, cv=3, scoring='accuracy')
        grid.fit(X_train_resampled, y_train_resampled)

        new_pred = grid.best_estimator_.predict(new_data_scaled)
        new_proba = grid.best_estimator_.predict_proba(new_data_scaled)[:, 1]

        st.write(f"### 🔍 {model_option} 모델 결과")
        st.success(f"✅ 예측된 결과: **{'당뇨' if new_pred[0] == 1 else '비당뇨'}**")
        st.info(f"📈 당뇨 확률: **{new_proba[0]:.2%}**")


# 탭 3: 모델 성능
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

# with tab3:
#     st.subheader(f"📉 {model_option} 성능 분석")
    
#     # 정확도 출력
#     accuracy = grid.best_estimator_.score(X_test_scaled, y_test)
#     st.write(f"✅ 정확도 (Accuracy): **{accuracy:.2%}**")
    
#     # 분류 리포트 계산
#     y_pred = grid.best_estimator_.predict(X_test_scaled)
#     report_dict = classification_report(y_test, y_pred, target_names=['비당뇨', '당뇨'], output_dict=True)
#     report_df = pd.DataFrame(report_dict).transpose()
    
#     # 정수 지원 수치를 정수로 변환
#     report_df['support'] = report_df['support'].astype(int)
    
#     # 분류 리포트 표 형태로 출력
#     st.subheader("📋 분류 리포트")
#     st.dataframe(report_df.style.format({"precision": "{:.2f}", "recall": "{:.2f}", "f1-score": "{:.2f}"}))
    
#     # 혼동 행렬 시각화
#     st.subheader("🧾 혼동 행렬")
#     cm = confusion_matrix(y_test, y_pred)
#     fig2, ax = plt.subplots(figsize=(5, 4))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['비당뇨', '당뇨'], yticklabels=['비당뇨', '당뇨'])
#     plt.xlabel("예측값")
#     plt.ylabel("실제값")
#     st.pyplot(fig2)

#     # ROC 곡선 시각화
#     st.subheader("📉 ROC 곡선")
#     y_proba = grid.best_estimator_.predict_proba(X_test_scaled)[:, 1]
#     fpr, tpr, thresholds = roc_curve(y_test, y_proba)
#     roc_auc = auc(fpr, tpr)
    
#     fig3, ax3 = plt.subplots(figsize=(6, 5))
#     ax3.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
#     ax3.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#     ax3.set_xlim([0.0, 1.0])
#     ax3.set_ylim([0.0, 1.05])
#     ax3.set_xlabel('False Positive Rate')
#     ax3.set_ylabel('True Positive Rate')
#     ax3.set_title('Receiver Operating Characteristic (ROC)')
#     ax3.legend(loc="lower right")
#     st.pyplot(fig3)

# 탭 3: 모델 성능
with tab3:
    st.subheader("📊 모델 성능")
    
    # 모델 설정
    models = {
        "로지스틱 회귀": LogisticRegression(),
        "랜덤 포레스트": RandomForestClassifier(),
        "XGBoost": XGBClassifier()
    }

    # 스타일 정의 (각 모델에 대해 다른 스타일)
    model_styles = {
        "로지스틱 회귀": {"color": "blue", "linestyle": "-", "linewidth": 2},
        "랜덤 포레스트": {"color": "green", "linestyle": "--", "linewidth": 2},
        "XGBoost": {"color": "red", "linestyle": "-.", "linewidth": 2}
    }
    
    # 모델 학습 및 예측
    for model_name, model in models.items():
        # 모델 학습
        model.fit(X_train_resampled, y_train_resampled)
        
        # 예측 확률 계산 (양성 클래스의 확률)
        y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]
        
        # ROC Curve 계산
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        
        # ROC Curve 시각화
        st.write(f"### {model_name}의 ROC Curve")
        
        fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
        
        # 스타일에 맞게 ROC Curve 그리기
        ax_roc.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.4f})',
                    color=model_styles[model_name]["color"],
                    linestyle=model_styles[model_name]["linestyle"],
                    linewidth=model_styles[model_name]["linewidth"])
        
        # 랜덤 모델의 기준선 (대각선)
        ax_roc.plot([0, 1], [0, 1], color='gray', linestyle='--')
        
        ax_roc.set_xlim([0.0, 1.0])
        ax_roc.set_ylim([0.0, 1.05])
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.set_title(f'{model_name} - Receiver Operating Characteristic (ROC) Curve')
        ax_roc.legend(loc='lower right')
        
        st.pyplot(fig_roc)
    
    # 모델 성능에 대한 설명
    st.write("### 모델 성능에 대한 설명")
    st.markdown("""
    - **정확도**는 전체 데이터 중 맞게 예측한 비율을 나타냅니다. 하지만 불균형 데이터에서는 다른 지표들에 비해 과대 평가될 수 있습니다.
    - **F1-Score**는 정밀도와 재현율의 조화 평균으로, 불균형 클래스 문제에서 중요합니다.
    - **AUC (Area Under Curve)**는 ROC 곡선 아래의 면적을 나타내며, 1에 가까울수록 좋은 성능을 나타냅니다.
    - **혼동 행렬**은 예측한 클래스와 실제 클래스 간의 비교를 시각화한 결과로, 잘못 분류된 샘플을 확인하는 데 유용합니다.
    """)







# 탭 4: 인사이트 요약
with tab4:
    st.subheader("🔍 인사이트 요약")
    st.markdown("""
    - **HbA1c 수치**, **BMI**, **혈당 수치**가 높을수록 당뇨 확률이 증가합니다.
    - **흡연 이력**은 간접적으로 당뇨와 관련이 있을 수 있습니다.
    - XGBoost 모델이 전반적으로 가장 높은 성능을 보였습니다.
    """)
