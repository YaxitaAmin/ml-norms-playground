import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_classification, make_regression
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import seaborn as sns
import time
from sklearn.metrics import roc_curve, auc

# Page config
st.set_page_config(
    page_title="Vector Norms Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #2c3e50;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .insight-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-left: 4px solid #3498db;
        margin: 1rem 0;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .section-header {
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">Vector Norms Analysis Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.1rem; color: #7f8c8d;">Comprehensive Analysis of Vector Norms in Data Science Applications</p>', unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Select Analysis Module",
    ["Interactive Visualizer", "Recommendation Systems", "Fraud Detection", "Machine Learning Regularization", "Performance Benchmark"]
)

# Utility functions
def calculate_norms(vector):
    """Calculate different norms for a vector"""
    l1 = np.linalg.norm(vector, ord=1)
    l2 = np.linalg.norm(vector, ord=2)
    l_inf = np.linalg.norm(vector, ord=np.inf)
    return l1, l2, l_inf

def create_norm_balls_3d():
    """Create 3D visualization of norm balls"""
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('L1 Norm (Manhattan)', 'L2 Norm (Euclidean)', 'L∞ Norm (Chebyshev)'),
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}, {'type': 'scatter3d'}]]
    )
    
    # Generate points for each norm ball
    theta = np.linspace(0, 2*np.pi, 100)
    phi = np.linspace(0, np.pi, 50)
    
    # L1 norm (octahedron)
    t = np.linspace(-1, 1, 50)
    x1, y1, z1 = [], [], []
    for i in range(len(t)):
        for j in range(len(t)):
            remaining = 1 - abs(t[i]) - abs(t[j])
            if remaining >= 0:
                x1.extend([t[i], t[i]])
                y1.extend([t[j], t[j]])
                z1.extend([remaining, -remaining])
    
    # L2 norm (sphere)
    x2, y2, z2 = [], [], []
    for p in phi:
        for t in theta:
            x2.append(np.sin(p) * np.cos(t))
            y2.append(np.sin(p) * np.sin(t))
            z2.append(np.cos(p))
    
    # L∞ norm (cube)
    x3, y3, z3 = [], [], []
    for i in [-1, 1]:
        for j in np.linspace(-1, 1, 20):
            for k in np.linspace(-1, 1, 20):
                x3.extend([i, j, k])
                y3.extend([j, k, i])
                z3.extend([k, i, j])
    
    fig.add_trace(go.Scatter3d(x=x1, y=y1, z=z1, mode='markers', 
                               marker=dict(size=2, color='red'), name='L1'), row=1, col=1)
    fig.add_trace(go.Scatter3d(x=x2, y=y2, z=z2, mode='markers', 
                               marker=dict(size=2, color='blue'), name='L2'), row=1, col=2)
    fig.add_trace(go.Scatter3d(x=x3, y=y3, z=z3, mode='markers', 
                               marker=dict(size=2, color='green'), name='L∞'), row=1, col=3)
    
    fig.update_layout(height=500, showlegend=False)
    return fig

# PAGE 1: Interactive Visualizer
if page == "Interactive Visualizer":
    st.markdown('<h2 class="section-header">Interactive Norm Visualizer</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Vector Input Controls")
        
        # Vector input
        x = st.slider("X component", -10.0, 10.0, 3.0, 0.1)
        y = st.slider("Y component", -10.0, 10.0, 4.0, 0.1)
        z = st.slider("Z component", -10.0, 10.0, 5.0, 0.1)
        
        vector = np.array([x, y, z])
        l1, l2, l_inf = calculate_norms(vector)
        
        # Display norms with style
        st.markdown(f'<div class="metric-card"><h3>L1 Norm: {l1:.3f}</h3><p>Manhattan Distance</p></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-card"><h3>L2 Norm: {l2:.3f}</h3><p>Euclidean Distance</p></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-card"><h3>L∞ Norm: {l_inf:.3f}</h3><p>Maximum Component</p></div>', unsafe_allow_html=True)
        
        # Insights
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown("**Mathematical Properties:**")
        st.markdown(f"• L1 = {l1:.2f} (sum of absolute values)")
        st.markdown(f"• L2 = {l2:.2f} (√(x²+y²+z²))")
        st.markdown(f"• L∞ = {l_inf:.2f} (largest component)")
        st.markdown("• L1 ≥ L2 ≥ L∞ (triangle inequality)")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.subheader("3D Norm Balls Visualization")
        fig_3d = create_norm_balls_3d()
        st.plotly_chart(fig_3d, use_container_width=True)
        
        # 2D comparison
        fig_2d = go.Figure()
        
        # L1 norm (diamond)
        diamond_x = [1, 0, -1, 0, 1]
        diamond_y = [0, 1, 0, -1, 0]
        fig_2d.add_trace(go.Scatter(x=diamond_x, y=diamond_y, mode='lines', 
                                   name='L1 Norm', line=dict(color='red', width=3)))
        
        # L2 norm (circle)
        theta = np.linspace(0, 2*np.pi, 100)
        circle_x = np.cos(theta)
        circle_y = np.sin(theta)
        fig_2d.add_trace(go.Scatter(x=circle_x, y=circle_y, mode='lines', 
                                   name='L2 Norm', line=dict(color='blue', width=3)))
        
        # L∞ norm (square)
        square_x = [-1, 1, 1, -1, -1]
        square_y = [-1, -1, 1, 1, -1]
        fig_2d.add_trace(go.Scatter(x=square_x, y=square_y, mode='lines', 
                                   name='L∞ Norm', line=dict(color='green', width=3)))
        
        # Add the current vector
        fig_2d.add_trace(go.Scatter(x=[0, x], y=[0, y], mode='lines+markers', 
                                   name='Current Vector', line=dict(color='purple', width=4),
                                   marker=dict(size=10)))
        
        fig_2d.update_layout(
            title="2D Norm Comparison",
            xaxis_title="X Component", yaxis_title="Y Component",
            showlegend=True, height=400
        )
        st.plotly_chart(fig_2d, use_container_width=True)

# PAGE 2: Recommendation Systems
elif page == "Recommendation Systems":
    st.markdown('<h2 class="section-header">Recommendation System Analysis</h2>', unsafe_allow_html=True)
    
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.markdown("**Application:** Analyzing which norm best captures user similarity for content recommendation systems")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Generate synthetic user data
    np.random.seed(42)
    genres = ['Action', 'Comedy', 'Drama', 'Horror', 'Romance', 'Sci-Fi', 'Thriller']
    n_users = 1000
    
    # Create user preference matrix
    user_preferences = np.random.rand(n_users, len(genres)) * 5  # 0-5 rating scale
    
    # Create sample users
    target_user = np.array([4.5, 2.0, 3.5, 1.0, 4.0, 3.0, 2.5])  # Target user preferences
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("User Preference Configuration")
        
        # Interactive user preference input
        user_prefs = {}
        for i, genre in enumerate(genres):
            user_prefs[genre] = st.slider(f"{genre} Rating", 0.0, 5.0, float(target_user[i]), 0.1)
        
        current_user = np.array(list(user_prefs.values()))
        
        # Calculate similarities using different norms
        l1_similarities = []
        l2_similarities = []
        linf_similarities = []
        
        for user in user_preferences:
            diff = current_user - user
            l1_similarities.append(1 / (1 + np.linalg.norm(diff, ord=1)))
            l2_similarities.append(1 / (1 + np.linalg.norm(diff, ord=2)))
            linf_similarities.append(1 / (1 + np.linalg.norm(diff, ord=np.inf)))
        
        # Find top recommendations
        top_l1 = np.argsort(l1_similarities)[-10:][::-1]
        top_l2 = np.argsort(l2_similarities)[-10:][::-1]
        top_linf = np.argsort(linf_similarities)[-10:][::-1]
        
        st.markdown("**Top Similar Users:**")
        st.markdown(f"• L1 Norm: User {top_l1[0]} (similarity: {l1_similarities[top_l1[0]]:.3f})")
        st.markdown(f"• L2 Norm: User {top_l2[0]} (similarity: {l2_similarities[top_l2[0]]:.3f})")
        st.markdown(f"• L∞ Norm: User {top_linf[0]} (similarity: {linf_similarities[top_linf[0]]:.3f})")
        
        # Average similarity scores
        avg_l1 = np.mean(l1_similarities)
        avg_l2 = np.mean(l2_similarities)
        avg_linf = np.mean(linf_similarities)
        
        st.markdown("**Average Similarity Scores:**")
        st.markdown(f"• L1: {avg_l1:.3f}")
        st.markdown(f"• L2: {avg_l2:.3f}")
        st.markdown(f"• L∞: {avg_linf:.3f}")
    
    with col2:
        st.subheader("Recommendation Analysis")
        
        # Create recommendation comparison chart
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Your Preferences', 'L1 Top Match', 'L2 Top Match', 'L∞ Top Match'),
            specs=[[{'type': 'polar'}, {'type': 'polar'}], [{'type': 'polar'}, {'type': 'polar'}]]
        )
        
        # Your preferences
        fig.add_trace(go.Scatterpolar(
            r=current_user, theta=genres, fill='toself',
            name='Your Preferences', line=dict(color='purple')
        ), row=1, col=1)
        
        # L1 recommendations
        fig.add_trace(go.Scatterpolar(
            r=user_preferences[top_l1[0]], theta=genres, fill='toself',
            name='L1 Match', line=dict(color='red')
        ), row=1, col=2)
        
        # L2 recommendations
        fig.add_trace(go.Scatterpolar(
            r=user_preferences[top_l2[0]], theta=genres, fill='toself',
            name='L2 Match', line=dict(color='blue')
        ), row=2, col=1)
        
        # L∞ recommendations
        fig.add_trace(go.Scatterpolar(
            r=user_preferences[top_linf[0]], theta=genres, fill='toself',
            name='L∞ Match', line=dict(color='green')
        ), row=2, col=2)
        
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Similarity distribution
        fig_dist = go.Figure()
        
        fig_dist.add_trace(go.Histogram(x=l1_similarities, name='L1 Similarities', 
                                       opacity=0.7, nbinsx=30, marker_color='red'))
        fig_dist.add_trace(go.Histogram(x=l2_similarities, name='L2 Similarities', 
                                       opacity=0.7, nbinsx=30, marker_color='blue'))
        fig_dist.add_trace(go.Histogram(x=linf_similarities, name='L∞ Similarities', 
                                       opacity=0.7, nbinsx=30, marker_color='green'))
        
        fig_dist.update_layout(
            title="Similarity Score Distributions",
            xaxis_title="Similarity Score",
            yaxis_title="Frequency",
            barmode='overlay',
            height=400
        )
        st.plotly_chart(fig_dist, use_container_width=True)
        
        # Insights
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown("**Norm Characteristics for Recommendations:**")
        st.markdown("• **L1 Norm:** Emphasizes overall taste alignment across all genres")
        st.markdown("• **L2 Norm:** Provides balanced similarity measure (industry standard)")
        st.markdown("• **L∞ Norm:** Focuses on users with similar strongest preferences")
        st.markdown("• **L2 typically performs best** for recommendation systems due to its balanced approach")
        st.markdown('</div>', unsafe_allow_html=True)

# PAGE 3: Fraud Detection
elif page == "Fraud Detection":
    st.markdown('<h2 class="section-header">Fraud Detection Analysis</h2>', unsafe_allow_html=True)
    
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.markdown("**Application:** Comparing vector norms for anomaly detection in financial transactions")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Generate synthetic transaction data
    np.random.seed(42)
    n_normal = 1000
    n_fraud = 50
    
    # Normal transactions
    normal_transactions = np.random.multivariate_normal([50, 100], [[100, 20], [20, 400]], n_normal)
    
    # Fraudulent transactions (outliers)
    fraud_transactions = np.random.multivariate_normal([200, 500], [[500, 100], [100, 1000]], n_fraud)
    
    all_transactions = np.vstack([normal_transactions, fraud_transactions])
    labels = np.hstack([np.zeros(n_normal), np.ones(n_fraud)])
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Detection Configuration")
        
        threshold = st.slider("Anomaly Threshold", 0.1, 5.0, 2.0, 0.1)
        
        # Calculate anomaly scores using different norms
        center = np.mean(normal_transactions, axis=0)
        
        l1_scores = [np.linalg.norm(tx - center, ord=1) for tx in all_transactions]
        l2_scores = [np.linalg.norm(tx - center, ord=2) for tx in all_transactions]
        linf_scores = [np.linalg.norm(tx - center, ord=np.inf) for tx in all_transactions]
        
        # Normalize scores
        l1_scores = np.array(l1_scores) / np.max(l1_scores) * 3
        l2_scores = np.array(l2_scores) / np.max(l2_scores) * 3
        linf_scores = np.array(linf_scores) / np.max(linf_scores) * 3
        
        # Calculate detection metrics
        l1_predictions = l1_scores > threshold
        l2_predictions = l2_scores > threshold
        linf_predictions = linf_scores > threshold
        
        l1_accuracy = accuracy_score(labels, l1_predictions)
        l2_accuracy = accuracy_score(labels, l2_predictions)
        linf_accuracy = accuracy_score(labels, linf_predictions)
        
        st.markdown("**Detection Accuracy:**")
        st.markdown(f'<div class="metric-card"><h4>L1 Norm: {l1_accuracy:.3f}</h4></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-card"><h4>L2 Norm: {l2_accuracy:.3f}</h4></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-card"><h4>L∞ Norm: {linf_accuracy:.3f}</h4></div>', unsafe_allow_html=True)
        
        # Calculate precision and recall
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        l1_precision = precision_score(labels, l1_predictions)
        l2_precision = precision_score(labels, l2_predictions)
        linf_precision = precision_score(labels, linf_predictions)
        
        l1_recall = recall_score(labels, l1_predictions)
        l2_recall = recall_score(labels, l2_predictions)
        linf_recall = recall_score(labels, linf_predictions)
        
        st.markdown("**Precision Scores:**")
        st.markdown(f"• L1: {l1_precision:.3f}")
        st.markdown(f"• L2: {l2_precision:.3f}")
        st.markdown(f"• L∞: {linf_precision:.3f}")
        
        st.markdown("**Recall Scores:**")
        st.markdown(f"• L1: {l1_recall:.3f}")
        st.markdown(f"• L2: {l2_recall:.3f}")
        st.markdown(f"• L∞: {linf_recall:.3f}")
    
    with col2:
        st.subheader("Fraud Detection Visualization")
        
        # Create scatter plot
        fig = go.Figure()
        
        # Normal transactions
        fig.add_trace(go.Scatter(
            x=normal_transactions[:, 0], y=normal_transactions[:, 1],
            mode='markers', name='Normal Transactions',
            marker=dict(color='blue', size=6, opacity=0.6)
        ))
        
        # Fraudulent transactions
        fig.add_trace(go.Scatter(
            x=fraud_transactions[:, 0], y=fraud_transactions[:, 1],
            mode='markers', name='Fraudulent Transactions',
            marker=dict(color='red', size=8, symbol='x')
        ))
        
        # Add threshold circles for different norms
        theta = np.linspace(0, 2*np.pi, 100)
        
        # L2 threshold circle
        circle_x = center[0] + threshold * 50 * np.cos(theta)
        circle_y = center[1] + threshold * 50 * np.sin(theta)
        fig.add_trace(go.Scatter(
            x=circle_x, y=circle_y, mode='lines',
            name='L2 Threshold', line=dict(color='green', dash='dash', width=2)
        ))
        
        fig.update_layout(
            title="Transaction Anomaly Detection",
            xaxis_title="Transaction Amount ($)",
            yaxis_title="Account Balance ($)",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # ROC curves comparison
        try:
            fpr_l1, tpr_l1, _ = roc_curve(labels, l1_scores)
            fpr_l2, tpr_l2, _ = roc_curve(labels, l2_scores)
            fpr_linf, tpr_linf, _ = roc_curve(labels, linf_scores)
            
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr_l1, y=tpr_l1, name=f'L1 Norm (AUC: {auc(fpr_l1, tpr_l1):.3f})', line=dict(color='red')))
            fig_roc.add_trace(go.Scatter(x=fpr_l2, y=tpr_l2, name=f'L2 Norm (AUC: {auc(fpr_l2, tpr_l2):.3f})', line=dict(color='blue')))
            fig_roc.add_trace(go.Scatter(x=fpr_linf, y=tpr_linf, name=f'L∞ Norm (AUC: {auc(fpr_linf, tpr_linf):.3f})', line=dict(color='green')))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Classifier', line=dict(dash='dash', color='gray')))
            
            fig_roc.update_layout(
                title="ROC Curves Comparison",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                height=400
            )
            st.plotly_chart(fig_roc, use_container_width=True)
        except:
            st.error("Could not generate ROC curves due to data limitations")

# PAGE 4: ML Playground
elif page == "Machine Learning Regularization":
    st.markdown('<h2 class="section-header">Machine Learning Regularization Analysis</h2>', unsafe_allow_html=True)
    
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.markdown("**Application:** Comparing L1 (Lasso) vs L2 (Ridge) regularization effects on regression models")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Generate synthetic regression data
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    noise = 0.1
    
    X, y = make_regression(n_samples=n_samples, n_features=n_features, n_informative=5, 
                          noise=noise, random_state=42)
    
    # Add some irrelevant features
    X_noise = np.random.randn(n_samples, n_features//2) * 0.1
    X = np.hstack([X, X_noise])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Regularization Configuration")
        
        alpha = st.slider("Regularization Strength (α)", 0.001, 10.0, 1.0, 0.001, format="%.3f")
        l1_ratio = st.slider("L1 Ratio (ElasticNet)", 0.0, 1.0, 0.5, 0.01)
        
        # Train models with different regularizations
        models = {
            'Linear': LinearRegression(),
            'Ridge (L2)': Ridge(alpha=alpha),
            'Lasso (L1)': Lasso(alpha=alpha, max_iter=2000),
            'ElasticNet': ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=2000)
        }
        
        results = {}
        
        for name, model in models.items():
            try:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                # Count non-zero coefficients
                if hasattr(model, 'coef_'):
                    non_zero_coefs = np.sum(np.abs(model.coef_) > 0.001)
                else:
                    non_zero_coefs = len(model.coef_) if hasattr(model, 'coef_') else 0
                
                results[name] = {
                    'MSE': mse,
                    'R²': r2,
                    'Non-zero Coefficients': non_zero_coefs,
                    'Coefficients': model.coef_ if hasattr(model, 'coef_') else None
                }
            except Exception as e:
                st.error(f"Error training {name}: {str(e)}")
                results[name] = {'MSE': np.nan, 'R²': np.nan, 'Non-zero Coefficients': 0, 'Coefficients': None}
        
        # Display results
        st.markdown("**Model Performance:**")
        for name, result in results.items():
            if not np.isnan(result['MSE']):
                st.markdown(f'<div class="metric-card"><h4>{name}</h4><p>MSE: {result["MSE"]:.3f} | R²: {result["R²"]:.3f}</p></div>', unsafe_allow_html=True)
        
        # Feature selection info
        st.markdown("**Feature Selection:**")
        for name, result in results.items():
            if result['Non-zero Coefficients'] > 0:
                st.markdown(f"• {name}: {result['Non-zero Coefficients']}/{X.shape[1]} features")
    
    with col2:
        st.subheader("Model Comparison Analysis")
        
        # Create coefficient comparison plot
        feature_names = [f'Feature {i+1}' for i in range(X.shape[1])]
        
        fig = go.Figure()
        
        colors = ['blue', 'red', 'green', 'orange']
        for i, (name, result) in enumerate(results.items()):
            if result['Coefficients'] is not None:
                fig.add_trace(go.Bar(
                    x=feature_names, y=result['Coefficients'],
                    name=name, marker_color=colors[i % len(colors)], opacity=0.7
                ))
        
        fig.update_layout(
            title="Model Coefficients Comparison",
            xaxis_title="Features",
            yaxis_title="Coefficient Value",
            barmode='group',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Regularization path visualization
        alphas = np.logspace(-4, 1, 30)
        ridge_coefs = []
        lasso_coefs = []
        ridge_scores = []
        lasso_scores = []
        
# Regularization path visualization
        alphas = np.logspace(-4, 1, 30)
        ridge_coefs = []
        lasso_coefs = []
        ridge_scores = []
        lasso_scores = []
        
        for a in alphas:
            try:
                ridge_temp = Ridge(alpha=a)
                lasso_temp = Lasso(alpha=a, max_iter=2000)
                
                ridge_temp.fit(X_train_scaled, y_train)
                lasso_temp.fit(X_train_scaled, y_train)
                
                ridge_pred = ridge_temp.predict(X_test_scaled)
                lasso_pred = lasso_temp.predict(X_test_scaled)
                
                ridge_coefs.append(ridge_temp.coef_)
                lasso_coefs.append(lasso_temp.coef_)
                
                ridge_scores.append(r2_score(y_test, ridge_pred))
                lasso_scores.append(r2_score(y_test, lasso_pred))
                
            except:
                ridge_coefs.append(np.zeros(X.shape[1]))
                lasso_coefs.append(np.zeros(X.shape[1]))
                ridge_scores.append(0)
                lasso_scores.append(0)
        
        # Plot regularization path
        fig_path = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Ridge Regularization Path', 'Lasso Regularization Path', 
                          'Ridge R² Score', 'Lasso R² Score'),
            specs=[[{'secondary_y': False}, {'secondary_y': False}],
                   [{'secondary_y': False}, {'secondary_y': False}]]
        )
        
        # Ridge path
        ridge_coefs_array = np.array(ridge_coefs)
        for i in range(min(10, ridge_coefs_array.shape[1])):  # Show first 10 features
            fig_path.add_trace(go.Scatter(
                x=alphas, y=ridge_coefs_array[:, i],
                name=f'Ridge Feature {i+1}', mode='lines'
            ), row=1, col=1)
        
        # Lasso path
        lasso_coefs_array = np.array(lasso_coefs)
        for i in range(min(10, lasso_coefs_array.shape[1])):  # Show first 10 features
            fig_path.add_trace(go.Scatter(
                x=alphas, y=lasso_coefs_array[:, i],
                name=f'Lasso Feature {i+1}', mode='lines'
            ), row=1, col=2)
        
        # Ridge scores
        fig_path.add_trace(go.Scatter(
            x=alphas, y=ridge_scores,
            name='Ridge R²', mode='lines+markers', line=dict(color='red', width=3)
        ), row=2, col=1)
        
        # Lasso scores
        fig_path.add_trace(go.Scatter(
            x=alphas, y=lasso_scores,
            name='Lasso R²', mode='lines+markers', line=dict(color='blue', width=3)
        ), row=2, col=2)
        
        fig_path.update_xaxes(type='log', title_text="Regularization Parameter (α)")
        fig_path.update_yaxes(title_text="Coefficient Value", row=1, col=1)
        fig_path.update_yaxes(title_text="Coefficient Value", row=1, col=2)
        fig_path.update_yaxes(title_text="R² Score", row=2, col=1)
        fig_path.update_yaxes(title_text="R² Score", row=2, col=2)
        
        fig_path.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig_path, use_container_width=True)
        
        # Insights
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown("**Regularization Effects:**")
        st.markdown("• **L1 (Lasso):** Promotes sparsity by setting coefficients to zero")
        st.markdown("• **L2 (Ridge):** Shrinks coefficients but keeps all features")
        st.markdown("• **ElasticNet:** Combines L1 and L2 penalties for balanced regularization")
        st.markdown("• **Feature Selection:** Lasso naturally performs feature selection")
        st.markdown('</div>', unsafe_allow_html=True)

# PAGE 5: Performance Benchmark
elif page == "Performance Benchmark":
    st.markdown('<h2 class="section-header">Performance Benchmark Analysis</h2>', unsafe_allow_html=True)
    
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.markdown("**Application:** Benchmarking computational performance of different vector norm calculations")
    st.markdown('</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Benchmark Configuration")
        
        # Benchmark parameters
        vector_size = st.slider("Vector Size", 100, 100000, 10000, 100)
        num_iterations = st.slider("Number of Iterations", 10, 1000, 100, 10)
        
        # Generate test data
        np.random.seed(42)
        test_vectors = np.random.randn(num_iterations, vector_size)
        
        if st.button("Run Benchmark"):
            # Benchmark different norm calculations
            results = {}
            
            # L1 Norm benchmark
            with st.spinner("Benchmarking L1 Norm..."):
                start_time = time.time()
                for vector in test_vectors:
                    np.linalg.norm(vector, ord=1)
                l1_time = time.time() - start_time
                results['L1 Norm'] = l1_time
            
            # L2 Norm benchmark
            with st.spinner("Benchmarking L2 Norm..."):
                start_time = time.time()
                for vector in test_vectors:
                    np.linalg.norm(vector, ord=2)
                l2_time = time.time() - start_time
                results['L2 Norm'] = l2_time
            
            # L∞ Norm benchmark
            with st.spinner("Benchmarking L∞ Norm..."):
                start_time = time.time()
                for vector in test_vectors:
                    np.linalg.norm(vector, ord=np.inf)
                linf_time = time.time() - start_time
                results['L∞ Norm'] = linf_time
            
            # Manual L1 implementation
            with st.spinner("Benchmarking Manual L1..."):
                start_time = time.time()
                for vector in test_vectors:
                    np.sum(np.abs(vector))
                manual_l1_time = time.time() - start_time
                results['Manual L1'] = manual_l1_time
            
            # Manual L2 implementation
            with st.spinner("Benchmarking Manual L2..."):
                start_time = time.time()
                for vector in test_vectors:
                    np.sqrt(np.sum(vector**2))
                manual_l2_time = time.time() - start_time
                results['Manual L2'] = manual_l2_time
            
            # Manual L∞ implementation
            with st.spinner("Benchmarking Manual L∞..."):
                start_time = time.time()
                for vector in test_vectors:
                    np.max(np.abs(vector))
                manual_linf_time = time.time() - start_time
                results['Manual L∞'] = manual_linf_time
            
            # Store results in session state
            st.session_state['benchmark_results'] = results
            st.session_state['benchmark_config'] = {
                'vector_size': vector_size,
                'num_iterations': num_iterations
            }
        
        # Display results if available
        if 'benchmark_results' in st.session_state:
            results = st.session_state['benchmark_results']
            config = st.session_state['benchmark_config']
            
            st.markdown("**Benchmark Results:**")
            for method, time_taken in results.items():
                avg_time = (time_taken / config['num_iterations']) * 1000  # Convert to milliseconds
                st.markdown(f'<div class="metric-card"><h4>{method}</h4><p>{time_taken:.4f}s total<br>{avg_time:.4f}ms per vector</p></div>', unsafe_allow_html=True)
    
    with col2:
        st.subheader("Performance Analysis")
        
        if 'benchmark_results' in st.session_state:
            results = st.session_state['benchmark_results']
            config = st.session_state['benchmark_config']
            
            # Create performance comparison chart
            methods = list(results.keys())
            times = list(results.values())
            avg_times = [(t / config['num_iterations']) * 1000 for t in times]  # milliseconds per vector
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=methods, y=avg_times,
                marker_color=['red', 'blue', 'green', 'orange', 'purple', 'brown'],
                text=[f'{t:.4f}ms' for t in avg_times],
                textposition='auto'
            ))
            
            fig.update_layout(
                title=f"Average Computation Time per Vector (Size: {config['vector_size']})",
                xaxis_title="Method",
                yaxis_title="Time (milliseconds)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Scalability analysis
            st.subheader("Scalability Analysis")
            
            # Generate scalability data
            sizes = [100, 500, 1000, 5000, 10000, 50000]
            l1_times = []
            l2_times = []
            linf_times = []
            
            for size in sizes:
                test_vector = np.random.randn(size)
                
                # L1 timing
                start = time.time()
                for _ in range(10):
                    np.linalg.norm(test_vector, ord=1)
                l1_times.append((time.time() - start) / 10 * 1000)
                
                # L2 timing
                start = time.time()
                for _ in range(10):
                    np.linalg.norm(test_vector, ord=2)
                l2_times.append((time.time() - start) / 10 * 1000)
                
                # L∞ timing
                start = time.time()
                for _ in range(10):
                    np.linalg.norm(test_vector, ord=np.inf)
                linf_times.append((time.time() - start) / 10 * 1000)
            
            fig_scale = go.Figure()
            
            fig_scale.add_trace(go.Scatter(
                x=sizes, y=l1_times, mode='lines+markers',
                name='L1 Norm', line=dict(color='red', width=3)
            ))
            
            fig_scale.add_trace(go.Scatter(
                x=sizes, y=l2_times, mode='lines+markers',
                name='L2 Norm', line=dict(color='blue', width=3)
            ))
            
            fig_scale.add_trace(go.Scatter(
                x=sizes, y=linf_times, mode='lines+markers',
                name='L∞ Norm', line=dict(color='green', width=3)
            ))
            
            fig_scale.update_layout(
                title="Scalability Analysis: Time vs Vector Size",
                xaxis_title="Vector Size",
                yaxis_title="Time (milliseconds)",
                height=400
            )
            st.plotly_chart(fig_scale, use_container_width=True)
            
            # Memory usage analysis
            st.subheader("Memory Usage Analysis")
            
            memory_data = {
                'Method': ['L1 Norm', 'L2 Norm', 'L∞ Norm', 'Manual L1', 'Manual L2', 'Manual L∞'],
                'Memory Complexity': ['O(n)', 'O(n)', 'O(n)', 'O(n)', 'O(n)', 'O(n)'],
                'Temporary Arrays': ['1', '1', '1', '1', '1', '1'],
                'Relative Efficiency': ['High', 'High', 'Highest', 'Medium', 'Medium', 'High']
            }
            
            memory_df = pd.DataFrame(memory_data)
            st.dataframe(memory_df, use_container_width=True)
            
            # Performance insights
            st.markdown('<div class="insight-box">', unsafe_allow_html=True)
            st.markdown("**Performance Insights:**")
            st.markdown("• **L∞ Norm:** Fastest computation (simple max operation)")
            st.markdown("• **L1 Norm:** Moderate speed (sum of absolute values)")
            st.markdown("• **L2 Norm:** Slowest due to square root operation")
            st.markdown("• **NumPy implementations:** Generally faster than manual implementations")
            st.markdown("• **Scalability:** All norms scale linearly with vector size O(n)")
            st.markdown("• **Memory:** All methods have similar memory requirements")
            st.markdown('</div>', unsafe_allow_html=True)
        
        else:
            st.info("Click 'Run Benchmark' to see performance analysis")
            
            # Show theoretical complexity
            st.subheader("Theoretical Complexity Analysis")
            
            complexity_data = {
                'Norm Type': ['L1 (Manhattan)', 'L2 (Euclidean)', 'L∞ (Chebyshev)'],
                'Time Complexity': ['O(n)', 'O(n)', 'O(n)'],
                'Space Complexity': ['O(1)', 'O(1)', 'O(1)'],
                'Operations': ['n additions, n absolute values', 'n multiplications, n additions, 1 sqrt', 'n comparisons, n absolute values'],
                'Relative Speed': ['Medium', 'Slow', 'Fast']
            }
            
            complexity_df = pd.DataFrame(complexity_data)
            st.dataframe(complexity_df, use_container_width=True)

# Footer
st.markdown("---")
st.markdown('<p style="text-align: center; color: #7f8c8d;">Vector Norms Analysis Dashboard - Comprehensive Analysis Tool</p>', unsafe_allow_html=True)