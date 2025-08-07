import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy import stats
from scipy.optimize import minimize
import math

# Page configuration
st.set_page_config(
    page_title="Poisson Distribution Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
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
    
    .metric-card {
        background: linear-gradient(135deg, #667eea, #764ba2);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .stExpander > div:first-child {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
def poisson_pmf(y, lam):
    """Poisson probability mass function"""
    return np.exp(y * np.log(lam) - lam - gammaln(y + 1))

def gammaln(x):
    """Log gamma function approximation"""
    return np.array([math.lgamma(xi) for xi in x])

def generate_poisson_data(data_type="insurance_claims", seed=None):
    """Generate realistic Poisson regression data for different scenarios"""
    if seed is not None:
        np.random.seed(seed)
    
    if data_type == "insurance_claims":
        # Number of claims (x-axis) vs probability of occurrence (y-axis)
        lambda_param = 2.5  # Average number of claims
        max_claims = 10
        
        # Generate theoretical probabilities for each number of claims
        X = np.arange(0, max_claims + 1)  # Claims: 0, 1, 2, ..., 10
        y_theoretical = stats.poisson.pmf(X, lambda_param)  # Theoretical probabilities
        
        # Add some noise to simulate observed data
        noise_factor = 0.15
        y = y_theoretical + np.random.normal(0, noise_factor * y_theoretical.max(), len(X))
        y = np.maximum(y, 0)  # Ensure non-negative probabilities
        
        x_label, y_label = "Number of Claims", "Probability"
        true_lambda = lambda_param
        
    elif data_type == "hospital_visits":
        # Hospital visits (x-axis) vs probability (y-axis)
        lambda_param = 1.8
        max_visits = 8
        
        X = np.arange(0, max_visits + 1)
        y_theoretical = stats.poisson.pmf(X, lambda_param)
        
        noise_factor = 0.12
        y = y_theoretical + np.random.normal(0, noise_factor * y_theoretical.max(), len(X))
        y = np.maximum(y, 0)
        
        x_label, y_label = "Hospital Visits per Year", "Probability"
        true_lambda = lambda_param
        
    elif data_type == "defect_counts":
        # Defect counts (x-axis) vs probability (y-axis)
        lambda_param = 3.2
        max_defects = 12
        
        X = np.arange(0, max_defects + 1)
        y_theoretical = stats.poisson.pmf(X, lambda_param)
        
        noise_factor = 0.10
        y = y_theoretical + np.random.normal(0, noise_factor * y_theoretical.max(), len(X))
        y = np.maximum(y, 0)
        
        x_label, y_label = "Defects per Batch", "Probability"
        true_lambda = lambda_param
        
    elif data_type == "website_clicks":
        # Click counts (x-axis) vs probability (y-axis)
        lambda_param = 4.5
        max_clicks = 15
        
        X = np.arange(0, max_clicks + 1)
        y_theoretical = stats.poisson.pmf(X, lambda_param)
        
        noise_factor = 0.08
        y = y_theoretical + np.random.normal(0, noise_factor * y_theoretical.max(), len(X))
        y = np.maximum(y, 0)
        
        x_label, y_label = "Clicks per Campaign", "Probability"
        true_lambda = lambda_param
        
    elif data_type == "call_center":
        # Calls per hour (x-axis) vs probability (y-axis)
        lambda_param = 6.0
        max_calls = 18
        
        X = np.arange(0, max_calls + 1)
        y_theoretical = stats.poisson.pmf(X, lambda_param)
        
        noise_factor = 0.06
        y = y_theoretical + np.random.normal(0, noise_factor * y_theoretical.max(), len(X))
        y = np.maximum(y, 0)
        
        x_label, y_label = "Calls per Hour", "Probability"
        true_lambda = lambda_param
        
    elif data_type == "customer_purchases":
        # Purchases (x-axis) vs probability (y-axis)
        lambda_param = 1.2
        max_purchases = 6
        
        X = np.arange(0, max_purchases + 1)
        y_theoretical = stats.poisson.pmf(X, lambda_param)
        
        noise_factor = 0.20
        y = y_theoretical + np.random.normal(0, noise_factor * y_theoretical.max(), len(X))
        y = np.maximum(y, 0)
        
        x_label, y_label = "Purchases per Week", "Probability"
        true_lambda = lambda_param
        
    else:  # Default case
        lambda_param = 2.0
        max_count = 8
        X = np.arange(0, max_count + 1)
        y_theoretical = stats.poisson.pmf(X, lambda_param)
        
        noise_factor = 0.15
        y = y_theoretical + np.random.normal(0, noise_factor * y_theoretical.max(), len(X))
        y = np.maximum(y, 0)
        
        x_label, y_label = "Count", "Probability"
        true_lambda = lambda_param
    
    return X.astype(int), y, x_label, y_label, true_lambda

# Initialize session state
if 'data' not in st.session_state:
    X_sample, y_sample, x_label, y_label, true_lambda = generate_poisson_data("insurance_claims", seed=42)
    st.session_state.data = pd.DataFrame({'x': X_sample, 'y': y_sample})
    st.session_state.x_label = x_label
    st.session_state.y_label = y_label
    st.session_state.true_lambda = true_lambda

if 'manual_points' not in st.session_state:
    st.session_state.manual_points = pd.DataFrame(columns=['x', 'y'])

# Main app
st.markdown('<h1 class="main-header">🎯 Poisson Distribution Explorer</h1>', unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
<strong>Poisson Distribution</strong> models the probability of count events occurring in fixed intervals. 
This visualization shows the relationship between number of events (x-axis) and their probability of occurrence (y-axis).
</div>
""", unsafe_allow_html=True)

# Sidebar controls
st.sidebar.header("🔧 Controls")

# Data source selection
data_source = st.sidebar.radio(
    "Choose data source:",
    #["Generated Sample", "Manual Input", "Upload CSV"]
    ["Generated Sample"]
)

# Model parameters
st.sidebar.subheader("Model Parameters")
manual_mode = st.sidebar.checkbox("Manual λ adjustment", value=False)

if manual_mode:
    lambda_param = st.sidebar.slider("Lambda (λ)", 0.1, 10.0, 2.5, 0.1)
else:
    lambda_param = None

# Data handling based on source
if data_source == "Generated Sample":
    scenario = st.sidebar.selectbox(
        "Choose scenario:",
        [
            "insurance_claims",
            "hospital_visits", 
            "defect_counts",
            "website_clicks",
            "call_center",
            "customer_purchases"
        ],
        format_func=lambda x: {
            "insurance_claims": "📋 Insurance Claims by Count",
            "hospital_visits": "🏥 Hospital Visits Distribution", 
            "defect_counts": "🏭 Manufacturing Defects",
            "website_clicks": "💻 Website Click Distribution",
            "call_center": "☎️ Call Center Volume",
            "customer_purchases": "🛒 Customer Purchase Frequency"
        }[x]
    )
    
    if st.sidebar.button("🎲 Generate New Sample") or 'scenario' not in st.session_state or st.session_state.scenario != scenario:
        X_sample, y_sample, x_label, y_label, true_lambda = generate_poisson_data(
            scenario, seed=np.random.randint(1, 1000)
        )
        st.session_state.data = pd.DataFrame({'x': X_sample, 'y': y_sample})
        st.session_state.x_label = x_label
        st.session_state.y_label = y_label
        st.session_state.scenario = scenario
        st.session_state.true_lambda = true_lambda
    
    current_data = st.session_state.data

elif data_source == "Manual Input":
    st.sidebar.subheader("Add Data Points")
    
    # Use current labels if available
    x_label = getattr(st.session_state, 'x_label', 'Count Value')
    y_label = getattr(st.session_state, 'y_label', 'Probability')
    
    new_x = st.sidebar.number_input(x_label, 0, 20, 2, 1)  # Count values
    new_y = st.sidebar.number_input(y_label, 0.0, 1.0, 0.1, 0.01)  # Probability values
    
    col1, col2 = st.sidebar.columns(2)
    if col1.button("➕ Add Point"):
        new_point = pd.DataFrame({'x': [int(new_x)], 'y': [float(new_y)]})
        st.session_state.manual_points = pd.concat([st.session_state.manual_points, new_point], 
                                                  ignore_index=True)
    
    if col2.button("🗑️ Clear All"):
        st.session_state.manual_points = pd.DataFrame(columns=['x', 'y'])
    
    current_data = st.session_state.manual_points
    
    # Set default labels for manual input
    if 'x_label' not in st.session_state:
        st.session_state.x_label = 'Count Value'
        st.session_state.y_label = 'Probability'

else:  # Upload CSV
    uploaded_file = st.sidebar.file_uploader("Choose CSV file", type="csv")
    if uploaded_file is not None:
        current_data = pd.read_csv(uploaded_file)
        if 'x' not in current_data.columns or 'y' not in current_data.columns:
            st.error("CSV must contain 'x' and 'y' columns")
            current_data = pd.DataFrame(columns=['x', 'y'])
    else:
        current_data = pd.DataFrame(columns=['x', 'y'])

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    if len(current_data) > 0:
        X = current_data['x'].values
        y = current_data['y'].values
        
        # Fit model if not in manual mode
        if not manual_mode:
            # For probability data, fit lambda parameter using method of moments
            if np.sum(y) > 0:
                fitted_lambda = np.sum(X * y) / np.sum(y)  # Weighted mean
            else:
                fitted_lambda = 1.0
        else:
            fitted_lambda = lambda_param
        
        # Create theoretical Poisson curve
        x_range = np.arange(0, max(15, X.max() + 3))
        y_theoretical = stats.poisson.pmf(x_range, fitted_lambda)
        
        # Main plot
        fig = go.Figure()
        
        # Add observed data points
        fig.add_trace(go.Scatter(
            x=X, y=y,
            mode='markers',
            marker=dict(size=12, color='red', opacity=0.8, 
                       line=dict(width=2, color='white')),
            name='Observed Data',
            hovertemplate=f'{getattr(st.session_state, "x_label", "Count")}: %{{x}}<br>{getattr(st.session_state, "y_label", "Probability")}: %{{y:.4f}}<extra></extra>'
        ))
        
        # Add theoretical Poisson PMF
        fig.add_trace(go.Scatter(
            x=x_range, y=y_theoretical,
            mode='lines+markers',
            line=dict(color='blue', width=3),
            marker=dict(size=6, color='blue'),
            name=f'Poisson PMF (λ = {fitted_lambda:.2f})',
            hovertemplate=f'{getattr(st.session_state, "x_label", "Count")}: %{{x}}<br>Theoretical Prob: %{{y:.4f}}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Poisson Probability Mass Function Fit',
            xaxis_title=getattr(st.session_state, 'x_label', 'Count'),
            yaxis_title=getattr(st.session_state, 'y_label', 'Probability'),
            height=500,
            hovermode='closest',
            showlegend=True,
            template='plotly_white',
            yaxis=dict(range=[0, max(0.5, y.max() * 1.1) if len(y) > 0 else 0.5])
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Model diagnostics for probability data
        st.subheader("📊 Model Diagnostics")
        
        # Calculate goodness of fit metrics
        if len(X) > 0:
            # Chi-square goodness of fit test
            expected_probs = stats.poisson.pmf(X, fitted_lambda)
            # Avoid division by zero
            expected_probs = np.maximum(expected_probs, 1e-10)
            
            # Calculate chi-square statistic
            chi_square = np.sum((y - expected_probs)**2 / expected_probs)
            degrees_freedom = len(X) - 1  # One parameter estimated
            
            # Calculate R-squared equivalent for probability fit
            ss_res = np.sum((y - expected_probs)**2)
            ss_tot = np.sum((y - np.mean(y))**2) if np.std(y) > 0 else 1
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Root Mean Square Error
            rmse = np.sqrt(np.mean((y - expected_probs)**2))
        else:
            chi_square = 0
            r_squared = 0
            rmse = 0
        
        # Display metrics in columns
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.metric("Fitted λ", f"{fitted_lambda:.3f}")
        with metric_col2:
            st.metric("Chi-Square", f"{chi_square:.3f}")
        with metric_col3:
            st.metric("R²", f"{r_squared:.3f}")
        with metric_col4:
            st.metric("RMSE", f"{rmse:.4f}")
        
        # Residual analysis for probability data
        if len(X) > 0:
            expected_probs = stats.poisson.pmf(X, fitted_lambda)
            residuals = y - expected_probs
            
            fig_residuals = make_subplots(rows=1, cols=2, 
                                        subplot_titles=('Residuals vs Count', 'Observed vs Expected'))
            
            # Residuals vs Count
            fig_residuals.add_trace(
                go.Scatter(x=X, y=residuals, mode='markers',
                          marker=dict(color='green', opacity=0.6, size=8),
                          name='Residuals'),
                row=1, col=1
            )
            fig_residuals.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
            
            # Observed vs Expected probabilities
            fig_residuals.add_trace(
                go.Scatter(x=expected_probs, y=y, mode='markers',
                          marker=dict(color='purple', opacity=0.6, size=8),
                          name='Observed vs Expected'),
                row=1, col=2
            )
            
            # Perfect fit line
            min_val = min(min(expected_probs), min(y))
            max_val = max(max(expected_probs), max(y))
            fig_residuals.add_trace(
                go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines',
                          line=dict(color='red', dash='dash'),
                          name='Perfect Fit'),
                row=1, col=2
            )
            
            fig_residuals.update_layout(height=400, showlegend=False, template='plotly_white')
            fig_residuals.update_xaxes(title_text="Count", row=1, col=1)
            fig_residuals.update_yaxes(title_text="Residuals", row=1, col=1)
            fig_residuals.update_xaxes(title_text="Expected Probability", row=1, col=2)
            fig_residuals.update_yaxes(title_text="Observed Probability", row=1, col=2)
            
            st.plotly_chart(fig_residuals, use_container_width=True)
    
    else:
        st.info("📝 Add some data points to see the Poisson distribution analysis!")

with col2:
    # Mathematical explanation
    st.subheader("📚 Theory")
    
    with st.expander("🔢 Mathematical Foundation"):
        st.markdown("""
        **Poisson Probability Mass Function:**
        $$P(X = k) = \\frac{\\lambda^k e^{-\\lambda}}{k!}$$
        
        **Key Properties:**
        - $E[X] = \\lambda$ (Expected value)
        - $Var[X] = \\lambda$ (Variance)
        - $k = 0, 1, 2, 3, ...$ (Non-negative integers)
        
        **Parameter Estimation:**
        $$\\hat{\\lambda} = \\bar{X} = \\frac{1}{n}\\sum_{i=1}^n X_i$$
        """)
    
    with st.expander("🎯 Real-World Applications"):
        st.markdown("""
        **Poisson distributions model count events:**
        
        📋 **Insurance Claims**: Claims per policy per year
        - λ ≈ 2.5 means average 2-3 claims annually
        - P(0 claims) ≈ 0.08, P(1 claim) ≈ 0.21
        
        🏥 **Healthcare**: Hospital visits per patient
        - λ ≈ 1.8 for patients with chronic conditions
        - Helps predict resource allocation
        
        🏭 **Manufacturing**: Defects per production batch
        - λ ≈ 3.2 defects per 1000 units
        - Quality control and process improvement
        
        💻 **Web Analytics**: Clicks per email campaign
        - λ ≈ 4.5 clicks per 1000 emails sent
        - Campaign effectiveness measurement
        
        ☎️ **Call Centers**: Calls received per hour
        - λ ≈ 6 calls during peak hours
        - Staffing optimization
        
        🛒 **Retail**: Customer purchases per week
        - λ ≈ 1.2 for regular customers
        - Inventory and marketing planning
        """)
    
    with st.expander("📊 Understanding the Visualization"):
        st.markdown("""
        **What you're seeing:**
        
        🔴 **Red dots**: Observed probability data points
        - X-axis: Count values (0, 1, 2, 3, ...)
        - Y-axis: Probability of each count occurring
        
        🔵 **Blue line**: Theoretical Poisson PMF
        - Fitted using maximum likelihood estimation
        - Shows what probabilities "should be" given λ
        
        **Interpreting the fit:**
        - **Good fit**: Blue line passes close to red dots
        - **Poor fit**: Large gaps between observed and theoretical
        - **Parameter λ**: Controls the shape and peak of distribution
        
        **Key insights:**
        - Higher λ → distribution shifts right (more likely high counts)
        - Lower λ → distribution peaks at 0 or 1 (rare events)
        - Sum of all probabilities should equal 1.0
        """)
    
    with st.expander("📊 Model Assessment"):
        st.markdown("""
        **Goodness of Fit Metrics:**
        
        - **Chi-Square**: Measures how well theoretical fits observed
          - Lower values indicate better fit
          - Compare to critical value for significance
        
        - **R²**: Proportion of variance explained (0 to 1)
          - Higher values indicate better fit
          - R² > 0.8 suggests excellent fit
        
        - **RMSE**: Root Mean Square Error of probabilities
          - Lower values indicate better predictions
          - Measures average prediction error
        
        - **Parameter λ**: Fitted rate parameter
          - Should be close to true λ if known
          - Represents expected count value
        
        **Visual diagnostics:**
        - **Residuals plot**: Should show random scatter around zero
        - **Observed vs Expected**: Points should fall on diagonal line
        """)
    
    # Data summary
    if len(current_data) > 0:
        st.subheader("📈 Data Summary")
        
        # Show scenario-specific information
        if hasattr(st.session_state, 'scenario'):
            scenario_info = {
                "insurance_claims": "📋 **Insurance Claims Distribution**: Probability mass function of claim counts per policy",
                "hospital_visits": "🏥 **Healthcare Analytics**: Distribution of patient visit frequencies", 
                "defect_counts": "🏭 **Quality Control**: Probability distribution of defects per batch",
                "website_clicks": "💻 **Digital Marketing**: Click count distribution per campaign",
                "call_center": "☎️ **Operations Research**: Call volume probability distribution",
                "customer_purchases": "🛒 **Customer Behavior**: Purchase frequency distribution analysis"
            }
            
            if st.session_state.scenario in scenario_info:
                st.markdown(scenario_info[st.session_state.scenario])
                
                # Show true vs fitted lambda
                if hasattr(st.session_state, 'true_lambda'):
                    st.write(f"**True λ parameter:** {st.session_state.true_lambda:.3f}")
                    
                    if not manual_mode and len(X) > 0:
                        st.write(f"**Fitted λ parameter:** {fitted_lambda:.3f}")
                        param_diff = abs(fitted_lambda - st.session_state.true_lambda)
                        st.write(f"**Parameter error:** |Δλ| = {param_diff:.3f}")
                        
                        # Goodness of fit interpretation
                        if param_diff < 0.1:
                            st.success("✅ Excellent fit! Parameter very close to true value.")
                        elif param_diff < 0.3:
                            st.info("ℹ️ Good fit! Parameter reasonably close to true value.")
                        else:
                            st.warning("⚠️ Parameter differs significantly from true value.")
        
        st.write(f"**Data points:** {len(current_data)}")
        st.write(f"**Count range:** [{int(current_data['x'].min())}, {int(current_data['x'].max())}]")
        st.write(f"**Probability range:** [{current_data['y'].min():.4f}, {current_data['y'].max():.4f}]")
        st.write(f"**Sum of probabilities:** {current_data['y'].sum():.3f}")
        
        # Check if probabilities sum to approximately 1
        prob_sum = current_data['y'].sum()
        if abs(prob_sum - 1.0) < 0.1:
            st.success("✅ **Valid probability distribution!** Sum ≈ 1.0")
        elif prob_sum > 1.1:
            st.warning("⚠️ **Probabilities sum > 1.0** Consider normalizing the data.")
        else:
            st.info("ℹ️ **Partial distribution** Sum < 1.0 (may be truncated data)")
        
        # Additional insights for probability data
        if len(current_data) > 0:
            # Find mode (most likely value)
            mode_idx = current_data['y'].idxmax()
            mode_count = current_data.loc[mode_idx, 'x']
            mode_prob = current_data.loc[mode_idx, 'y']
            st.write(f"**Most likely count:** {mode_count} (probability: {mode_prob:.3f})")
            
            # Calculate expected value (if probabilities are normalized)
            if prob_sum > 0:
                expected_value = np.sum(current_data['x'] * current_data['y']) / prob_sum
                st.write(f"**Expected value:** {expected_value:.2f}")
        
        # Show data table
        with st.expander("🔍 View Data"):
            # Sort by x for better readability and add cumulative probability
            display_data = current_data.sort_values('x').reset_index(drop=True)
            display_data['cumulative_prob'] = display_data['y'].cumsum()
            st.dataframe(display_data, use_container_width=True)

# Footer with additional information
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
<p><strong>Poisson Distribution Explorer</strong> | Built with Streamlit & Plotly</p>
<p>Visualize and understand the Poisson probability mass function with real-world count data scenarios.</p>
</div>
""", unsafe_allow_html=True)