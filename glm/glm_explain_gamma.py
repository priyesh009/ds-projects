import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy import stats

# Page configuration
st.set_page_config(
    page_title="Gamma Distribution Explorer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #43cea2 0%, #185a9d 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    
    .info-box {
        background-color: #f0f2f6;
        border-left: 4px solid #43cea2;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.5rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #43cea2, #185a9d);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Helper: Generate Gamma data
def generate_gamma_data(data_type="insurance_claims", seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    if data_type == "insurance_claims":
        shape, scale = 2.0, 1000.0  # claim severity
        X = np.linspace(0, 8000, 100)
        y_theoretical = stats.gamma.pdf(X, a=shape, scale=scale)
        noise = np.random.normal(0, 0.02 * y_theoretical.max(), len(X))
        y = np.maximum(y_theoretical + noise, 0)
        x_label, y_label = "Claim Amount", "Density"
        
    elif data_type == "repair_costs":
        shape, scale = 3.0, 500.0
        X = np.linspace(0, 6000, 100)
        y_theoretical = stats.gamma.pdf(X, a=shape, scale=scale)
        noise = np.random.normal(0, 0.02 * y_theoretical.max(), len(X))
        y = np.maximum(y_theoretical + noise, 0)
        x_label, y_label = "Repair Cost", "Density"
        
    elif data_type == "waiting_times":
        shape, scale = 5.0, 2.0
        X = np.linspace(0, 40, 100)
        y_theoretical = stats.gamma.pdf(X, a=shape, scale=scale)
        noise = np.random.normal(0, 0.02 * y_theoretical.max(), len(X))
        y = np.maximum(y_theoretical + noise, 0)
        x_label, y_label = "Waiting Time (minutes)", "Density"
        
    else:
        shape, scale = 2.0, 2.0
        X = np.linspace(0, 20, 100)
        y_theoretical = stats.gamma.pdf(X, a=shape, scale=scale)
        y = y_theoretical
        x_label, y_label = "X", "Density"
    
    return X, y, x_label, y_label, shape, scale

# Initialize session state
if 'gamma_data' not in st.session_state:
    X_sample, y_sample, x_label, y_label, true_shape, true_scale = generate_gamma_data("insurance_claims", seed=42)
    st.session_state.gamma_data = pd.DataFrame({'x': X_sample, 'y': y_sample})
    st.session_state.x_label = x_label
    st.session_state.y_label = y_label
    st.session_state.true_shape = true_shape
    st.session_state.true_scale = true_scale

# Main header
st.markdown('<h1 class="main-header">📈 Gamma Distribution Explorer</h1>', unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
<strong>Gamma Distribution</strong> models continuous, positive-only variables.  
Commonly used for modeling claim severity, waiting times, and other skewed data.
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("🔧 Controls")
scenario = st.sidebar.selectbox(
    "Choose scenario:",
    ["insurance_claims", "repair_costs", "waiting_times"],
    format_func=lambda x: {
        "insurance_claims": "🏠 Insurance Claim Severity",
        "repair_costs": "🔧 Repair Costs",
        "waiting_times": "⏳ Waiting Times"
    }[x]
)

if st.sidebar.button("🎲 Generate New Sample") or 'scenario' not in st.session_state or st.session_state.scenario != scenario:
    X_sample, y_sample, x_label, y_label, true_shape, true_scale = generate_gamma_data(scenario, seed=np.random.randint(1, 1000))
    st.session_state.gamma_data = pd.DataFrame({'x': X_sample, 'y': y_sample})
    st.session_state.x_label = x_label
    st.session_state.y_label = y_label
    st.session_state.scenario = scenario
    st.session_state.true_shape = true_shape
    st.session_state.true_scale = true_scale

shape_param = st.sidebar.slider("Shape (α)", 0.5, 10.0, st.session_state.true_shape, 0.1)
scale_param = st.sidebar.slider("Scale (θ)", 0.5, 2000.0, st.session_state.true_scale, 50.0)

# Main content
col1, col2 = st.columns([2,1])

with col1:
    df = st.session_state.gamma_data
    X = df['x'].values
    y = df['y'].values
    
    # Theoretical gamma curve
    y_theoretical = stats.gamma.pdf(X, a=shape_param, scale=scale_param)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=X, y=y, mode='markers',
        marker=dict(size=8, color='red', opacity=0.7),
        name="Observed"
    ))
    fig.add_trace(go.Scatter(
        x=X, y=y_theoretical, mode='lines',
        line=dict(color='blue', width=3),
        name=f"Gamma PDF (α={shape_param:.2f}, θ={scale_param:.2f})"
    ))
    fig.update_layout(
        title="Gamma Probability Density Function Fit",
        xaxis_title=st.session_state.x_label,
        yaxis_title=st.session_state.y_label,
        template="plotly_white",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Fit diagnostics
    st.subheader("📊 Model Diagnostics")
    ss_res = np.sum((y - y_theoretical) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    rmse = np.sqrt(np.mean((y - y_theoretical) ** 2))
    
    metric_col1, metric_col2 = st.columns(2)
    with metric_col1:
        st.metric("R²", f"{r_squared:.3f}")
    with metric_col2:
        st.metric("RMSE", f"{rmse:.4f}")

with col2:
    st.subheader("📚 Theory")
    with st.expander("🔢 Mathematical Foundation"):
        st.markdown(r"""
        **Gamma PDF:**
        $$
        f(x; \alpha, \theta) = \frac{1}{\Gamma(\alpha)\theta^\alpha} x^{\alpha-1} e^{-x/\theta}, \quad x > 0
        $$
        
        - Shape parameter: $\alpha$ (controls skewness)
        - Scale parameter: $\theta$ (stretches the distribution)
        
        **Mean:** $E[X] = \alpha \theta$  
        **Variance:** $Var[X] = \alpha \theta^2$
        """)
    
    with st.expander("🎯 Real-World Applications"):
        st.markdown("""
        - 🏠 **Insurance**: Claim severity modeling  
        - 🔧 **Repairs**: Cost distribution for damages  
        - ⏳ **Operations**: Waiting times between events  
        """)
    
    with st.expander("📊 Interpretation"):
        st.markdown("""
        - Larger **α** → more symmetric distribution  
        - Larger **θ** → stretches the curve to the right  
        - Useful for **positively skewed continuous data**
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
<p><strong>Gamma Distribution Explorer</strong> | Built with Streamlit & Plotly</p>
</div>
""", unsafe_allow_html=True)
