import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# 페이지 기본 설정
st.set_page_config(page_title="서울시 따릉이 대여건수 예측", page_icon="🚲", layout="wide")

st.title("🚲 날씨 기반 서울시 공공자전거(따릉이) 일별 대여건수 예측")
st.markdown("상반기 기상 데이터(기온, 강수량, 습도)와 요일 정보를 학습한 **다중 선형회귀(Linear Regression)** 모델입니다.")

# 1. 데이터 로드 및 전처리 함수 (파일명을 간결한 영문으로 고정)
@st.cache_data
def load_and_merge_data():
    bike_file = "bike.csv"
    weather_file = "weather.csv"
    
    df_bike = pd.read_csv(bike_file, encoding="cp949")
    df_weather = pd.read_csv(weather_file, encoding="cp949")
    
    # 날짜 기준 병합
    df = pd.merge(df_bike, df_weather, left_on="대여일자", right_on="일시", how="inner")
    
    # 강수량 결측치(비가 안 온 날)는 0으로 채움
    df["일강수량(mm)"] = df["일강수량(mm)"].fillna(0)
    
    # 날짜에서 요일 및 주말 여부(토=5, 일=6) 추출
    df["날짜"] = pd.to_datetime(df["대여일자"])
    df["요일번호"] = df["날짜"].dt.dayofweek
    df["주말여부"] = df["요일번호"].apply(lambda x: 1 if x >= 5 else 0)
    df["구분"] = df["주말여부"].apply(lambda x: "주말" if x == 1 else "평일")
    
    return df

try:
    df = load_and_merge_data()
except Exception as e:
    st.error(f"데이터 파일을 불러오는 중 오류가 발생했습니다. 깃허브에 bike.csv와 weather.csv 파일이 잘 올라가 있는지 확인하세요.\n오류 내용: {e}")
    st.stop()

# 2. 선형회귀 모델 학습
features = ["평균기온(°C)", "일강수량(mm)", "평균 상대습도(%)", "주말여부"]
X = df[features]
y = df["대여건수"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# 3. 사이드바: 사용자 입력 시뮬레이터
st.sidebar.header("🌤️ 내일의 날씨 조건 입력")
temp = st.sidebar.slider("평균기온 (°C)", min_value=-15.0, max_value=35.0, value=20.0, step=0.5)
rain = st.sidebar.slider("일강수량 (mm)", min_value=0.0, max_value=100.0, value=0.0, step=1.0)
humidity = st.sidebar.slider("평균 상대습도 (%)", min_value=10.0, max_value=100.0, value=55.0, step=1.0)
is_weekend = st.sidebar.radio("요일 구분", options=["평일", "주말"])

weekend_val = 1 if is_weekend == "주말" else 0
input_data = pd.DataFrame([[temp, rain, humidity, weekend_val]], columns=features)
predicted_rentals = max(0, int(model.predict(input_data)[0]))

# 4. 메인 화면: 예측 결과 및 모델 성능 지표
col1, col2, col3 = st.columns(3)
col1.metric("예상 일일 대여건수", f"{predicted_rentals:,} 건")
col2.metric("모델 설명력 (R² Score)", f"{r2:.2f}")
col3.metric("평균 절대 오차 (MAE)", f"{int(mae):,} 건")

st.divider()

# 5. 시각화 차트 영역
left_col, right_col = st.columns(2)

with left_col:
    st.subheader("📈 평균기온과 대여건수의 상관관계")
    fig1 = px.scatter(
        df,
        x="평균기온(°C)",
        y="대여건수",
        color="구분",
        labels={"대여건수": "일별 대여건수 (건)"}
    )
    st.plotly_chart(fig1, use_container_width=True)

with right_col:
    st.subheader("🎯 실제 대여건수 vs 선형회귀 예측값")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=y_test, y=y_pred, mode="markers", name="테스트 데이터",
        marker=dict(color="#1f77b4", size=8, opacity=0.7)
    ))
    min_val, max_val = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    fig2.add_trace(go.Scatter(
        x=[min_val, max_val], y=[min_val, max_val], mode="lines", name="완벽 예측선 (y=x)",
        line=dict(color="red", dash="dash")
    ))
    fig2.update_layout(xaxis_title="실제 대여건수 (건)", yaxis_title="예측 대여건수 (건)")
    st.plotly_chart(fig2, use_container_width=True)

# 6. 회귀계수 확인
with st.expander("🔍 선형회귀 모델의 변수별 영향력(가중치) 확인"):
    coef_df = pd.DataFrame({
        "변수명": features,
        "회귀계수 (가중치)": [round(c, 1) for c in model.coef_]
    })
    st.dataframe(coef_df, use_container_width=True)
