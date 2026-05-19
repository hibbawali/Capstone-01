import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Ad Sales Predictor",
    page_icon="📈",
    layout="centered"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0d0d0d;
    color: #f0f0f0;
}

.main { background-color: #0d0d0d; }

h1, h2, h3 {
    font-family: 'Space Mono', monospace;
    color: #c8f135;
}

.stButton > button {
    background-color: #c8f135;
    color: #0d0d0d;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    border: none;
    border-radius: 4px;
    padding: 0.6rem 2rem;
    width: 100%;
    font-size: 1rem;
    cursor: pointer;
    transition: 0.2s;
}
.stButton > button:hover {
    background-color: #aacc00;
    transform: scale(1.02);
}

.result-box {
    background: #1a1a1a;
    border: 2px solid #c8f135;
    border-radius: 8px;
    padding: 1.5rem;
    text-align: center;
    margin-top: 1rem;
}
.result-box h2 {
    font-size: 2.5rem;
    margin: 0;
    color: #c8f135;
}
.result-box p {
    color: #aaa;
    margin: 0.3rem 0 0 0;
    font-size: 0.9rem;
}

.metric-row {
    display: flex;
    gap: 1rem;
    margin-top: 1rem;
}
.metric-card {
    flex: 1;
    background: #1a1a1a;
    border: 1px solid #333;
    border-radius: 8px;
    padding: 1rem;
    text-align: center;
}
.metric-card h4 {
    color: #c8f135;
    font-family: 'Space Mono', monospace;
    margin: 0 0 0.3rem 0;
    font-size: 1.3rem;
}
.metric-card p {
    color: #888;
    margin: 0;
    font-size: 0.8rem;
}

.tag {
    display: inline-block;
    background: #1e2a00;
    color: #c8f135;
    border: 1px solid #c8f135;
    border-radius: 20px;
    padding: 0.2rem 0.8rem;
    font-size: 0.75rem;
    font-family: 'Space Mono', monospace;
    margin-bottom: 0.5rem;
}

hr { border-color: #222; }
</style>
""", unsafe_allow_html=True)


# ── Load & Train Model ────────────────────────────────────────────────────────
@st.cache_resource
def load_and_train():
    data = pd.read_csv("data.csv")
    X = data[['TV', 'Radio', 'Newspaper']]
    y = data['Sales']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    poly = PolynomialFeatures(degree=2)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly  = poly.transform(X_test)
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    y_pred = model.predict(X_test_poly)
    r2  = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    return model, poly, r2, mse, data

model, poly, r2, mse, data = load_and_train()


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<span class="tag">ML · Polynomial Regression · Degree 2</span>', unsafe_allow_html=True)
st.title("📈 Ad Sales Predictor")
st.markdown("Predict product **sales revenue** from your TV, Radio & Newspaper advertising budgets.")
st.markdown("---")


# ── Input Sliders ─────────────────────────────────────────────────────────────
st.subheader("Enter Ad Budgets")

col1, col2, col3 = st.columns(3)
with col1:
    tv = st.slider("📺 TV Budget", min_value=0.0, max_value=300.0, value=150.0, step=1.0)
with col2:
    radio = st.slider("📻 Radio Budget", min_value=0.0, max_value=50.0, value=25.0, step=0.5)
with col3:
    newspaper = st.slider("📰 Newspaper Budget", min_value=0.0, max_value=120.0, value=30.0, step=1.0)

st.markdown("---")


# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("🚀 Predict Sales"):
    input_data = np.array([[tv, radio, newspaper]])
    input_poly  = poly.transform(input_data)
    prediction  = model.predict(input_poly)[0]

    st.markdown(f"""
    <div class="result-box">
        <p>Predicted Sales</p>
        <h2>${prediction:.2f}K</h2>
        <p>Based on TV={tv}, Radio={radio}, Newspaper={newspaper}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-card">
            <h4>{r2*100:.1f}%</h4>
            <p>Model Accuracy (R²)</p>
        </div>
        <div class="metric-card">
            <h4>{mse:.2f}</h4>
            <p>Mean Squared Error</p>
        </div>
        <div class="metric-card">
            <h4>{len(data)}</h4>
            <p>Training Samples</p>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Data Explorer ─────────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("🔍 View Raw Dataset"):
    st.dataframe(data.style.background_gradient(subset=['Sales'], cmap='YlGn'), use_container_width=True)

st.markdown("<br><p style='text-align:center;color:#444;font-size:0.8rem;font-family:Space Mono'>Built by Anaya · Capstone-01 · hibbawali</p>", unsafe_allow_html=True)
