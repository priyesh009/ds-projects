import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import poisson
from scipy.special import gamma as gamma_func  # Correct gamma function
import math

# Page configuration
st.set_page_config(
    page_title="Tweedie Distribution Explorer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    text-align: center;
    font-size: 3rem;
    font-weight: bold;
    margin-bottom: 1rem;
}
.info-box {
    background-color: #f0f2f6;
    border-left: 4px solid #667eea;
    padding: 1rem;
    margin: 1rem 0;
    border-radius: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# Tweedie helper function
def tweedie_pmf(x, mu, phi, p):
    """Approximate Tweedie PMF for visualization"""
    x = np.array(x)
    if p == 1:
        return poisson.pmf(x, mu)
    elif p == 2:
        # Gamma PDF for continuous case
        from scipy.stats import gamma as gamma_dist
        shape = 1/phi
        scale = mu * phi
        return gamma_dist.pdf(x, a=shape, scale=scale)
    else:
        # Compound Poisson-Gamma approximation
        a = (2 - p) / (p - 1)
        scale = phi * mu**(2 - p) / (2 - p)
        y = (1 / (gamma_func(a) * scale**a)) * x**(a - 1) * np.exp(-x / scale)
        return y

# Generate sample Tweedie data
def generate_tweedie_data(mu=1.0, phi=1.0, p=1.5, max_x=20, seed=None):
    if seed is not None:
        np.random.seed(seed)
    x = np.arange(0, max_x + 1)
    y = tweedie_pmf(x, mu, phi, p)
    y = y + np.random.normal(0, 0.02 * y.max(), len(x))  # add small noise
    y = np.maximum(y, 0)
    return x, y

# Initialize session state
if 'data' not in st.session_state:
    X_sample, y_sample = generate_tweedie_data()
    st.session_state.data = pd.DataFrame({'x': X_sample, 'y': y_sample})
    st.session_state.mu = 1.0
    st.session_state.phi = 1.0
    st.session_state.p = 1.5

# Main app
st.markdown('<h1 class="main-header">🎯 Tweedie Distribution Explorer</h1>', unsafe_allow_html=True)
st.markdown("""
<div class="info-box">
The Tweedie distribution is a flexible family of distributions for modeling non-negative data,
including count data and continuous positive data with mass at zero.
</div>
""", unsafe_allow_html=True)

# Sidebar controls
st.sidebar.header("Controls")
manual_mode = st.sidebar.checkbox("Manual parameter adjustment", value=False)

if manual_mode:
    mu = st.sidebar.slider("Mean (μ)", 0.1, 10.0, 1.0, 0.1)
    phi = st.sidebar.slider("Dispersion (φ)", 0.1, 5.0, 1.0, 0.1)
    p = st.sidebar.slider("Power (p)", 1.0, 2.0, 1.5, 0.05)
else:
    mu = st.session_state.mu
    phi = st.session_state.phi
    p = st.session_state.p

if st.sidebar.button("🎲 Generate New Sample"):
    X_sample, y_sample = generate_tweedie_data(mu=mu, phi=phi, p=p, max_x=20, seed=np.random.randint(1,1000))
    st.session_state.data = pd.DataFrame({'x': X_sample, 'y': y_sample})
    st.session_state.mu = mu
    st.session_state.phi = phi
    st.session_state.p = p

current_data = st.session_state.data

# Plotting
col1, col2 = st.columns([2,1])
with col1:
    X = current_data['x'].values
    y = current_data['y'].values
    
    x_range = np.arange(0, max(20, X.max() + 1))
    y_theoretical = tweedie_pmf(x_range, mu, phi, p)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X, y=y, mode='markers', marker=dict(size=12, color='red'), name='Observed'))
    fig.add_trace(go.Scatter(x=x_range, y=y_theoretical, mode='lines+markers',
                             line=dict(color='blue', width=3),
                             name=f'Tweedie PMF (μ={mu}, φ={phi}, p={p})'))
    
    fig.update_layout(title='Tweedie Distribution Fit',
                      xaxis_title='x',
                      yaxis_title='Probability / Density',
                      template='plotly_white',
                      height=500)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Basic diagnostics
    expected_value = np.sum(X*y)/np.sum(y)
    st.write(f"**Approx. Expected Value:** {expected_value:.2f}")
    st.write(f"**Parameters used:** μ={mu}, φ={phi}, p={p}")

with col2:
    st.subheader("📚 Theory")
    st.markdown(f"""
    **Tweedie Distribution (Power {p})**

    - Mean: μ
    - Variance: φ * μ^p
    - Special cases:
        - p=1 → Poisson (count data)
        - p=2 → Gamma (continuous positive)
        - 1<p<2 → Compound Poisson-Gamma (common in insurance claims)

    **Uses**: Modeling non-negative data with spikes at zero, e.g., insurance claims, rainfall, or loss data.
    """)

    with st.expander("🔍 View Data"):
        st.dataframe(current_data)
