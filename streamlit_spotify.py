import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ---------- Utility Functions ----------
@st.cache_data(show_spinner=False)
def load_data():
    """Load and preprocess Spotify data"""
    try:
        # Load data
        df = pd.read_csv('spotify_churn_tidy.csv')
        
        return df
        
    except Exception as e:
        st.error(f"Error occurred while loading data: {str(e)}")
        return None

def kpi_card(label: str, value, help_txt: str = ""):
    """Create a KPI metric card"""
    st.metric(label, value if value is not None else "—", help=help_txt)

def try_mean(df, col):
    """Safely calculate mean of a column"""
    if hasattr(df, 'columns') and col in df.columns and df[col].notna().any():
        return float(df[col].mean())
    elif hasattr(df, 'name') and df.name == col and df.notna().any():
        return float(df.mean())
    else:
        return None

def try_sum(df, col):
    """Safely calculate sum of a column"""
    if hasattr(df, 'columns') and col in df.columns and df[col].notna().any():
        return float(df[col].sum())
    elif hasattr(df, 'name') and df.name == col and df.notna().any():
        return float(df.sum())
    else:
        return None

def try_unique(df, col):
    """Safely count unique values in a column"""
    if hasattr(df, 'columns') and col in df.columns:
        return int(df[col].nunique())
    elif hasattr(df, 'name') and df.name == col:
        return int(df.nunique())
    else:
        return None

def format_number(value, decimals=1):
    """Format large numbers with appropriate units (K, M, B)"""
    if value is None:
        return "—"
    
    abs_value = abs(value)
    if abs_value >= 1e9:
        return f"{value/1e9:.{decimals}f}B"
    elif abs_value >= 1e6:
        return f"{value/1e6:.{decimals}f}M"
    elif abs_value >= 1e3:
        return f"{value/1e3:.{decimals}f}K"
    else:
        return f"{value:.{decimals}f}"

def optional_chart(title, chart):
    """Display a plotly chart with title if chart is not None"""
    if chart is not None:
        st.plotly_chart(chart, use_container_width=True)

# ---------- Chart Creation Functions ----------
def create_line_chart(df, x_col, y_col, title, color="#1DB954"):
    """Create a line chart using plotly express"""
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        return None
    
    fig = px.line(
        df, 
        x=x_col, 
        y=y_col,
        title=title,
        color_discrete_sequence=[color],
        markers=True
    )
    fig.update_layout(
        height=400,
        xaxis_title=x_col.replace("_", " ").title(),
        yaxis_title=y_col.replace("_", " ").title(),
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12)
    )
    fig.update_traces(line=dict(width=3))
    return fig

def create_bar_chart(df, x_col, y_col, title, color="#1DB954", orientation='v'):
    """Create a bar chart using plotly express"""
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        return None
    
    if orientation == 'v':
        fig = px.bar(df, x=x_col, y=y_col, title=title, color_discrete_sequence=[color])
        x_title = x_col.replace("_", " ").title()
        y_title = y_col.replace("_", " ").title()
    else:
        fig = px.bar(df, x=x_col, y=y_col, title=title, color_discrete_sequence=[color], orientation='h')
        x_title = y_col.replace("_", " ").title()
        y_title = x_col.replace("_", " ").title()
    
    fig.update_layout(
        height=400,
        xaxis_title=x_title,
        yaxis_title=y_title,
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12)
    )
    return fig

def create_pie_chart(df, names_col, values_col, title, color_palette=None):
    """Create a pie chart using plotly express"""
    if df.empty or names_col not in df.columns or values_col not in df.columns:
        return None
    
    if color_palette is None:
        color_palette = ['#1DB954', '#1ed760', '#1aa34a', '#168f3a', '#137b2a']
    
    fig = px.pie(
        df, 
        names=names_col, 
        values=values_col,
        title=title,
        color_discrete_sequence=color_palette
    )
    fig.update_layout(
        height=400,
        showlegend=True,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12)
    )
    return fig

def create_histogram(df, column, title, color="#1DB954", nbins=30):
    """Create a histogram using plotly express"""
    if df.empty or column not in df.columns:
        return None
    
    fig = px.histogram(
        df, 
        x=column,
        title=title,
        color_discrete_sequence=[color],
        nbins=nbins
    )
    fig.update_layout(
        height=400,
        xaxis_title=column.replace("_", " ").title(),
        yaxis_title="Count",
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12)
    )
    return fig

def create_scatter_plot(df, x_col, y_col, title, color="#1DB954"):
    """Create a scatter plot using plotly express"""
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        return None
    
    fig = px.scatter(
        df, 
        x=x_col, 
        y=y_col,
        title=title,
        color_discrete_sequence=[color],
        opacity=0.6
    )
    fig.update_layout(
        height=400,
        xaxis_title=x_col.replace("_", " ").title(),
        yaxis_title=y_col.replace("_", " ").title(),
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12)
    )
    return fig

def create_box_plot(df, x_col, y_col, title, color="#1DB954"):
    """Create a box plot using plotly express"""
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        return None
    
    fig = px.box(
        df, 
        x=x_col, 
        y=y_col,
        title=title,
        color_discrete_sequence=[color]
    )
    fig.update_layout(
        height=400,
        xaxis_title=x_col.replace("_", " ").title(),
        yaxis_title=y_col.replace("_", " ").title(),
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12)
    )
    return fig

def create_correlation_heatmap(df, title):
    """Create a correlation heatmap using plotly"""
    if df.empty:
        return None
    
    # Select numeric columns for correlation
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Filter for specific columns we want to include
    desired_columns = [
        'age', 'listening_time_min', 'songs_per_day', 'skip_rate', 'ads_per_week', 
        'listens_per_minute', 'avg_session_like', 'ads_exposure_ratio'
    ]
    
    # Only include columns that exist in the data
    correlation_columns = [col for col in desired_columns if col in numeric_columns]
    
    if len(correlation_columns) < 2:
        return None
    
    # Calculate correlation matrix
    correlation_data = df[correlation_columns].corr()
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=correlation_data.values,
        x=correlation_data.columns,
        y=correlation_data.columns,
        colorscale='RdYlGn',
        zmid=0,
        text=np.round(correlation_data.values, 2),
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title=title,
        height=500,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12),
        xaxis=dict(tickangle=45),
        yaxis=dict(tickangle=0)
    )
    
    return fig

# ---------- Analysis Functions ----------
def create_overview_metrics(df):
    """Create overview metrics"""
    total_users = df['user_id'].nunique()
    churned_users = df['is_churned'].sum()
    active_users = total_users - churned_users
    churn_rate = churned_users / total_users if total_users > 0 else 0
    avg_listening_time = df['listening_time_min'].mean()
    
    return {
        'total_users': total_users,
        'churned_users': churned_users,
        'active_users': active_users,
        'churn_rate': churn_rate,
        'avg_listening_time': avg_listening_time
    }

def create_user_analysis(df):
    """User Analysis"""
    # Gender distribution
    gender_dist = df.groupby('gender').size().reset_index(name='count')
    
    # Age group analysis
    age_dist = df.groupby('age_group').size().reset_index(name='count')
    
    # Country analysis
    country_dist = df.groupby('country').size().sort_values(ascending=False).reset_index()
    country_dist.columns = ['country', 'user_count']
    
    # Subscription type analysis
    subscription_dist = df.groupby('subscription_type').size().reset_index(name='count')
    
    return gender_dist, age_dist, country_dist, subscription_dist

def create_engagement_analysis(df):
    """Engagement Analysis"""
    # Listening behavior by subscription type
    listening_by_sub = df.groupby('subscription_type').agg({
        'listening_time_min': 'mean',
        'songs_per_day': 'mean',
        'skip_rate': 'mean'
    }).reset_index()
    
    # Device usage analysis
    device_usage = df.groupby('device_type').size().reset_index(name='count')
    
    # Offline listening analysis
    offline_usage = df.groupby('offline_listening_flag').size().reset_index(name='count')
    
    return listening_by_sub, device_usage, offline_usage

def create_churn_analysis(df):
    """Churn Analysis"""
    # Churn rate by demographics
    churn_by_gender = df.groupby('gender')['is_churned'].agg(['count', 'sum', 'mean']).reset_index()
    churn_by_gender.columns = ['gender', 'total_users', 'churned_users', 'churn_rate']
    
    # Churn rate by subscription type
    churn_by_subscription = df.groupby('subscription_type')['is_churned'].agg(['count', 'sum', 'mean']).reset_index()
    churn_by_subscription.columns = ['subscription_type', 'total_users', 'churned_users', 'churn_rate']
    
    # Churn rate by age group
    churn_by_age = df.groupby('age_group')['is_churned'].agg(['count', 'sum', 'mean']).reset_index()
    churn_by_age.columns = ['age_group', 'total_users', 'churned_users', 'churn_rate']
    
    # Churn rate by country
    churn_by_country = df.groupby('country')['is_churned'].agg(['count', 'sum', 'mean']).reset_index()
    churn_by_country.columns = ['country', 'total_users', 'churned_users', 'churn_rate']
    churn_by_country = churn_by_country.sort_values('churn_rate', ascending=False)
    
    return churn_by_gender, churn_by_subscription, churn_by_age, churn_by_country

def create_behavior_analysis(df):
    """User Behavior Analysis"""
    # Listening time distribution
    listening_dist = df.groupby(pd.cut(df['listening_time_min'], bins=10))['user_id'].count().reset_index()
    listening_dist['listening_time_min'] = listening_dist['listening_time_min'].astype(str)
    
    # Songs per day distribution
    songs_dist = df.groupby(pd.cut(df['songs_per_day'], bins=10))['user_id'].count().reset_index()
    songs_dist['songs_per_day'] = songs_dist['songs_per_day'].astype(str)
    
    # Skip rate analysis
    skip_rate_dist = df.groupby(pd.cut(df['skip_rate'], bins=10))['user_id'].count().reset_index()
    skip_rate_dist['skip_rate'] = skip_rate_dist['skip_rate'].astype(str)
    
    return listening_dist, songs_dist, skip_rate_dist

# ---------- App Configuration ----------
st.set_page_config(
    page_title="Spotify User Analytics Dashboard", 
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- CSS Styles ----------
st.markdown("""
<style>
/* Spotify Brand Colors */
:root {
    --spotify-green: #1DB954;
    --spotify-light-green: #1ed760;
    --spotify-dark-green: #1aa34a;
    --spotify-black: #191414;
    --spotify-white: #FFFFFF;
}

/* More specific CSS selectors for expander hover */
[data-testid="stExpander"] details:hover summary,
[data-testid="stExpander"] details[open] summary:hover,
.stExpander details:hover summary,
.stExpander details[open] summary:hover {
    color: #1DB954 !important;
}

/* Alternative approach - target all possible expander elements */
div[data-testid="stExpander"] > div > details > summary:hover,
div[data-testid="stExpander"] > div > details[open] > summary:hover {
    color: #1DB954 !important;
}

/* Tab styling - font color and highlight */
.stTabs [data-baseweb="tab-list"] button {
    color: #666 !important;
}

.stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
    color: #1DB954 !important;
}

.stTabs [data-baseweb="tab-list"] button:hover {
    color: #1DB954 !important;
}

/* Tab highlight (underline) color */
[data-baseweb="tab-highlight"] {
    background-color: #1DB954 !important;
}

/* Remove tab border completely */
[data-baseweb="tab-border"] {
    display: none !important;
}

/* Expander toggle button (arrow) color */
[data-testid="stExpander"] details summary svg,
[data-testid="stExpander"] details[open] summary svg,
.stExpander details summary svg,
.stExpander details[open] summary svg {
    color: #1DB954 !important;
    fill: #1DB954 !important;
}

/* Alternative approach for expander arrow */
[data-testid="stExpander"] svg,
.stExpander svg {
    color: #1DB954 !important;
    fill: #1DB954 !important;
}

/* Metric cards styling */
[data-testid="metric-container"] {
    background-color: #f8f9fa;
    border: 1px solid #e9ecef;
    border-radius: 0.5rem;
    padding: 1rem;
}

[data-testid="metric-container"] > div {
    color: #1DB954;
}

/* Custom header styling */
.custom-header {
    background: linear-gradient(90deg, #1DB954 0%, #1ed760 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    color: #1DB954;
}
</style>
""", unsafe_allow_html=True)

# ---------- Main App ----------
def main():
    # Spotify Header
    st.markdown("""
    <div style="padding: 2rem; margin-bottom: 2rem;">
        <h1 style="text-align: center; font-size: 3rem; margin: 0;">
            <span style="font-size: 3rem;">🎵</span>
            <span style="color: #1DB954; display: inline-block; background: linear-gradient(90deg, #1DB954 0%, #1ed760 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; color: #1DB954;"> Spotify User Analytics Dashboard</span>
        </h1>
        <p style="color: #1DB954; background: linear-gradient(90deg, #1DB954 0%, #1ed760 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; color: #1DB954; text-align: center; margin: 0.5rem 0 0 0; font-size: 1.1rem;">
            Comprehensive Analysis of Spotify User Behavior and Churn Patterns
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Project Info
    st.markdown("### 📋 Project Overview")
    with st.expander("**Project Goal:** Analyze Spotify user behavior patterns and predict churn risk using user engagement data", expanded=False):
        st.markdown("""
        **Data Sources:** 
        - Spotify User Dataset with churn information
        - User demographics, listening behavior, subscription types
        - Device usage and engagement metrics
        
        **Technologies Used:**
        - Python (Pandas, Streamlit, Plotly)
        - Data Visualization and Interactive Dashboards
        - User Behavior Analysis and Churn Prediction
        
        **Key Insights:**
        - User engagement patterns
        - Churn risk factors
        - Subscription type preferences
        - Device usage trends
        """)
    
    st.markdown("")

    # Data loading
    with st.spinner("Loading Spotify data..."):
        df = load_data()
    
    if df is None:
        st.error("Unable to load data. Please check the file paths.")
        return

    # ---------- Sidebar Configuration ----------
    st.sidebar.markdown("### 🎛️ Dashboard Controls")
    
    # Sidebar Info Section
    st.sidebar.metric("Total Users", f"{df['user_id'].nunique():,}", "Unique Spotify Users")
    st.sidebar.metric("Churned Users", f"{df['is_churned'].sum():,}", f"{df['is_churned'].mean():.1%} Churn Rate")
    st.sidebar.metric("Active Users", f"{df[df['is_churned']==0]['user_id'].nunique():,}", "Currently Active")
    
    st.sidebar.markdown("---")
    
    # Filters Section
    st.sidebar.markdown("#### 🔍 Filters")
    
    # Gender filter
    genders = ['All'] + list(df['gender'].unique())
    selected_gender = st.sidebar.selectbox("Gender", genders)
    
    if selected_gender != 'All':
        df_filtered = df[df['gender'] == selected_gender]
    else:
        df_filtered = df
    
    # Age group filter
    age_groups = ['All'] + list(df['age_group'].unique())
    selected_age = st.sidebar.selectbox("Age Group", age_groups)
    
    if selected_age != 'All':
        df_filtered = df_filtered[df_filtered['age_group'] == selected_age]
    
    # Subscription type filter
    subscriptions = ['All'] + list(df['subscription_type'].unique())
    selected_subscription = st.sidebar.selectbox("Subscription Type", subscriptions)
    
    if selected_subscription != 'All':
        df_filtered = df_filtered[df_filtered['subscription_type'] == selected_subscription]
    
    # Country filter
    countries = ['All'] + sorted(list(df['country'].unique()))
    selected_country = st.sidebar.selectbox("Country", countries)
    
    if selected_country != 'All':
        df_filtered = df_filtered[df_filtered['country'] == selected_country]

    # ---------- KPIs ----------
    st.markdown("### 📊 Key Performance Indicators")
    
    # Main KPI Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_users = try_unique(df_filtered, 'user_id')
        kpi_card("Total Users", f"{total_users:,}", "Total users in current filter")
    
    with col2:
        churned_users = try_sum(df_filtered, 'is_churned')
        kpi_card("Churned Users", f"{churned_users:,}", "Users who have churned")
    
    with col3:
        churn_rate = try_mean(df_filtered, 'is_churned')
        kpi_card("Churn Rate", f"{churn_rate:.1%}", "Percentage of users who churned")
    
    with col4:
        avg_listening_time = try_mean(df_filtered, 'listening_time_min')
        kpi_card("Avg Listening Time", f"{avg_listening_time:.0f} min", "Average daily listening time")
    
    # Secondary metrics
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        avg_songs_per_day = try_mean(df_filtered, 'songs_per_day')
        kpi_card("Avg Songs/Day", f"{avg_songs_per_day:.0f}", "Average songs played per day")
    
    with col6:
        avg_skip_rate = try_mean(df_filtered, 'skip_rate')
        kpi_card("Avg Skip Rate", f"{avg_skip_rate:.1%}", "Average song skip rate")
    
    with col7:
        premium_users = len(df_filtered[df_filtered['subscription_type'] == 'Premium'])
        premium_rate = premium_users / len(df_filtered) if len(df_filtered) > 0 else 0
        kpi_card("Premium Users", f"{premium_users:,}", f"{premium_rate:.1%} of total users")
    
    with col8:
        offline_users = len(df_filtered[df_filtered['offline_listening_flag'] == 'Yes'])
        offline_rate = offline_users / len(df_filtered) if len(df_filtered) > 0 else 0
        kpi_card("Offline Users", f"{offline_users:,}", f"{offline_rate:.1%} use offline mode")
    
    st.markdown("")

    # ---------- Charts ----------
    st.markdown("### 📈 Data Visualizations")
    
    # Create tabs for different chart categories
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["◽ Overview", "◽ User Demographics", "◽ Engagement Analysis", "◽ Churn Analysis", "◽ Advanced Analytics"])
    
    with tab1:
        st.markdown("##### (1) Business Overview")
        
        # User distribution by subscription type
        subscription_dist = df_filtered.groupby('subscription_type').size().reset_index(name='count')
        subscription_pie = create_pie_chart(subscription_dist, 'subscription_type', 'count', 'User Distribution by Subscription Type')
        optional_chart("", subscription_pie)
        
        # Listening behavior comparison
        col1, col2 = st.columns(2)
        
        with col1:
            # Average listening time by subscription
            listening_by_sub = df_filtered.groupby('subscription_type')['listening_time_min'].mean().reset_index()
            listening_bar = create_bar_chart(listening_by_sub, 'subscription_type', 'listening_time_min', 'Avg Listening Time by Subscription')
            optional_chart("", listening_bar)
        
        with col2:
            # Average songs per day by subscription
            songs_by_sub = df_filtered.groupby('subscription_type')['songs_per_day'].mean().reset_index()
            songs_bar = create_bar_chart(songs_by_sub, 'subscription_type', 'songs_per_day', 'Avg Songs per Day by Subscription')
            optional_chart("", songs_bar)
        
        # Device usage analysis
        device_usage = df_filtered.groupby('device_type').size().reset_index(name='count')
        device_pie = create_pie_chart(device_usage, 'device_type', 'count', 'Device Usage Distribution')
        optional_chart("", device_pie)
    
    with tab2:
        st.markdown("##### (2) User Demographics")
        
        # Gender and age analysis
        col1, col2 = st.columns(2)
        
        with col1:
            gender_dist = df_filtered.groupby('gender').size().reset_index(name='count')
            gender_pie = create_pie_chart(gender_dist, 'gender', 'count', 'Gender Distribution')
            optional_chart("", gender_pie)
        
        with col2:
            age_dist = df_filtered.groupby('age_group').size().reset_index(name='count')
            age_bar = create_bar_chart(age_dist, 'age_group', 'count', 'Age Group Distribution')
            optional_chart("", age_bar)
        
        # Country analysis
        st.markdown("**Top 15 Countries by User Count**")
        country_dist = df_filtered.groupby('country').size().sort_values(ascending=False).head(15).reset_index()
        country_dist.columns = ['country', 'user_count']
        country_bar = create_bar_chart(country_dist, 'user_count', 'country', 'Top Countries by User Count', orientation='h')
        optional_chart("", country_bar)
        
        # Demographics vs churn
        col1, col2 = st.columns(2)
        
        with col1:
            churn_by_gender = df_filtered.groupby('gender')['is_churned'].mean().reset_index()
            churn_by_gender.columns = ['gender', 'churn_rate']
            gender_churn_bar = create_bar_chart(churn_by_gender, 'gender', 'churn_rate', 'Churn Rate by Gender')
            optional_chart("", gender_churn_bar)
        
        with col2:
            churn_by_age = df_filtered.groupby('age_group')['is_churned'].mean().reset_index()
            churn_by_age.columns = ['age_group', 'churn_rate']
            age_churn_bar = create_bar_chart(churn_by_age, 'age_group', 'churn_rate', 'Churn Rate by Age Group')
            optional_chart("", age_churn_bar)
    
    with tab3:
        st.markdown("##### (3) Engagement Analysis")
        
        # Listening behavior distributions
        col1, col2 = st.columns(2)
        
        with col1:
            listening_hist = create_histogram(df_filtered, 'listening_time_min', 'Listening Time Distribution (minutes)')
            optional_chart("", listening_hist)
        
        with col2:
            songs_hist = create_histogram(df_filtered, 'songs_per_day', 'Songs per Day Distribution')
            optional_chart("", songs_hist)
        
        # Skip rate and engagement correlation
        col1, col2 = st.columns(2)
        
        with col1:
            skip_hist = create_histogram(df_filtered, 'skip_rate', 'Skip Rate Distribution')
            optional_chart("", skip_hist)
        
        with col2:
            # Listening time vs songs per day correlation
            engagement_scatter = create_scatter_plot(df_filtered, 'listening_time_min', 'songs_per_day', 'Listening Time vs Songs per Day')
            optional_chart("", engagement_scatter)
        
        # Ads exposure analysis
        ads_analysis = df_filtered.groupby('subscription_type').agg({
            'ads_per_week': 'mean',
            'ads_exposure_ratio': 'mean'
        }).reset_index()
        
        col1, col2 = st.columns(2)
        
        with col1:
            ads_bar = create_bar_chart(ads_analysis, 'subscription_type', 'ads_per_week', 'Average Ads per Week by Subscription')
            optional_chart("", ads_bar)
        
        with col2:
            ads_ratio_bar = create_bar_chart(ads_analysis, 'subscription_type', 'ads_exposure_ratio', 'Ads Exposure Ratio by Subscription')
            optional_chart("", ads_ratio_bar)
    
    with tab4:
        st.markdown("##### (4) Churn Analysis")
        
        # Churn rate by different segments
        col1, col2 = st.columns(2)
        
        with col1:
            churn_by_subscription = df_filtered.groupby('subscription_type')['is_churned'].mean().reset_index()
            churn_by_subscription.columns = ['subscription_type', 'churn_rate']
            sub_churn_bar = create_bar_chart(churn_by_subscription, 'subscription_type', 'churn_rate', 'Churn Rate by Subscription Type')
            optional_chart("", sub_churn_bar)
        
        with col2:
            churn_by_device = df_filtered.groupby('device_type')['is_churned'].mean().reset_index()
            churn_by_device.columns = ['device_type', 'churn_rate']
            device_churn_bar = create_bar_chart(churn_by_device, 'device_type', 'churn_rate', 'Churn Rate by Device Type')
            optional_chart("", device_churn_bar)
        
        # Churn vs engagement metrics
        st.markdown("**Churn Risk vs Engagement Metrics**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Churned vs Active users listening time
            listening_box = create_box_plot(df_filtered, 'churn_label', 'listening_time_min', 'Listening Time: Churned vs Active')
            optional_chart("", listening_box)
        
        with col2:
            # Churned vs Active users skip rate
            skip_box = create_box_plot(df_filtered, 'churn_label', 'skip_rate', 'Skip Rate: Churned vs Active')
            optional_chart("", skip_box)
        
        # Top countries with highest churn rates
        st.markdown("**Top 15 Countries with Highest Churn Rates**")
        churn_by_country = df_filtered.groupby('country')['is_churned'].agg(['count', 'mean']).reset_index()
        churn_by_country.columns = ['country', 'user_count', 'churn_rate']
        churn_by_country = churn_by_country[churn_by_country['user_count'] >= 10]  # Filter for countries with at least 10 users
        churn_by_country = churn_by_country.sort_values('churn_rate', ascending=False).head(15)
        
        country_churn_bar = create_bar_chart(churn_by_country, 'churn_rate', 'country', 'Churn Rate by Country', orientation='h')
        optional_chart("", country_churn_bar)
    
    with tab5:
        st.markdown("##### (5) Advanced Analytics")
        
        # Correlation analysis
        st.markdown("**User Behavior Correlation Matrix**")
        correlation_heatmap = create_correlation_heatmap(df_filtered, 'Behavior Metrics Correlation')
        optional_chart("", correlation_heatmap)
        
        # Advanced metrics analysis
        col1, col2 = st.columns(2)
        
        with col1:
            # Listens per minute distribution
            listens_per_min_hist = create_histogram(df_filtered, 'listens_per_minute', 'Listens per Minute Distribution')
            optional_chart("", listens_per_min_hist)
        
        with col2:
            # Average session like distribution
            avg_session_hist = create_histogram(df_filtered, 'avg_session_like', 'Average Session Like Distribution')
            optional_chart("", avg_session_hist)
        
        # Advanced scatter plots
        col1, col2 = st.columns(2)
        
        with col1:
            # Skip rate vs listening time
            skip_vs_listening = create_scatter_plot(df_filtered, 'skip_rate', 'listening_time_min', 'Skip Rate vs Listening Time')
            optional_chart("", skip_vs_listening)
        
        with col2:
            # Songs per day vs avg session like
            songs_vs_like = create_scatter_plot(df_filtered, 'songs_per_day', 'avg_session_like', 'Songs per Day vs Average Session Like')
            optional_chart("", songs_vs_like)
        
        # Churn prediction insights
        st.markdown("**Churn Risk Factors Analysis**")
        
        # Create a summary of key churn factors
        churn_factors = df_filtered.groupby('is_churned').agg({
            'listening_time_min': 'mean',
            'songs_per_day': 'mean',
            'skip_rate': 'mean',
            'ads_per_week': 'mean'
        }).reset_index()
        
        churn_factors['churn_label'] = churn_factors['is_churned'].map({0: 'Active', 1: 'Churned'})
        
        # Display as a comparison table
        st.dataframe(
            churn_factors[['churn_label', 'listening_time_min', 'songs_per_day', 'skip_rate', 'ads_per_week']].round(2),
            use_container_width=True
        )
    
    st.markdown("---")
    
    # ---------- Key Insights Section ----------
    st.markdown("### 💡 Key Insights & Analysis")
    
    # Generate insights based on filtered data
    insights_col1, insights_col2 = st.columns(2)
    
    with insights_col1:
        st.markdown("##### (1) User Engagement Insights")
        
        # Engagement insights
        avg_listening = try_mean(df_filtered, 'listening_time_min')
        avg_songs = try_mean(df_filtered, 'songs_per_day')
        avg_skip = try_mean(df_filtered, 'skip_rate')
        
        st.markdown(f"""
        <div style="background-color: #f0f8f0; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem; border-left: 4px solid #1DB954;">
        <strong>🎵 Listening Behavior</strong><br>
        Average listening time: {avg_listening:.0f} minutes per day<br>
        Average songs played: {avg_songs:.0f} songs per day<br>
        Average skip rate: {avg_skip:.1%} (lower is better)
        </div>
        """, unsafe_allow_html=True)
        
        # Subscription insights
        premium_count = len(df_filtered[df_filtered['subscription_type'] == 'Premium'])
        free_count = len(df_filtered[df_filtered['subscription_type'] == 'Free'])
        total_count = len(df_filtered)
        
        premium_rate = premium_count / total_count if total_count > 0 else 0
        free_rate = free_count / total_count if total_count > 0 else 0
        
        st.markdown(f"""
        <div style="background-color: #f0f8f0; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem; border-left: 4px solid #1DB954;">
        <strong>💳 Subscription Mix</strong><br>
        Premium users: {premium_count:,} ({premium_rate:.1%})<br>
        Free users: {free_count:,} ({free_rate:.1%})<br>
        Premium conversion opportunity exists
        </div>
        """, unsafe_allow_html=True)
    
    with insights_col2:
        st.markdown("##### (2) Churn Prevention Recommendations")
        
        # Churn insights
        current_churn_rate = try_mean(df_filtered, 'is_churned')
        total_users = try_unique(df_filtered, 'user_id')
        
        st.markdown(f"""
        <div style="background-color: #fff0f0; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem; border-left: 4px solid #ff6b6b;">
        <strong>⚠️ Churn Risk</strong><br>
        Current churn rate: {current_churn_rate:.1%} ({df_filtered['is_churned'].sum():,} out of {total_users:,} users)<br>
        High-priority segment for retention campaigns
        </div>
        """, unsafe_allow_html=True)
        
        # Device and engagement insights
        mobile_users = len(df_filtered[df_filtered['device_type'] == 'Mobile'])
        mobile_rate = mobile_users / total_count if total_count > 0 else 0
        
        offline_users = len(df_filtered[df_filtered['offline_listening_flag'] == 'Yes'])
        offline_rate = offline_users / total_count if total_count > 0 else 0
        
        st.markdown(f"""
        <div style="background-color: #f0f8f0; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #1DB954;">
        <strong>📱 Usage Patterns</strong><br>
        Mobile users: {mobile_users:,} ({mobile_rate:.1%}) - focus on mobile experience<br>
        Offline users: {offline_users:,} ({offline_rate:.1%}) - premium feature users
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p><strong>🎵 Spotify User Analytics Dashboard</strong> | Built with Python, Streamlit & Plotly</p>
        <p>Data-driven insights for user engagement optimization and churn prevention</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
