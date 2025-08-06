import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
from typing import Dict, List, Any, Optional

# Import our modules
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.api.client import F1AnalyticsClient
from src.database.connection import db_pool
from src.analytics.performance import PerformanceAnalyzer
from src.data_pipeline.quality import DataQualityChecker
from src.streaming.realtime_processor import LiveDataSimulator
from config.settings import Config

# Page configuration
st.set_page_config(
    page_title="F1 Analytics Platform",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #FF1801;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #FF1801;
    }
    .sidebar-section {
        margin-bottom: 2rem;
    }
    .status-good {
        color: #28a745;
        font-weight: bold;
    }
    .status-warning {
        color: #ffc107;
        font-weight: bold;
    }
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'api_client' not in st.session_state:
    st.session_state.api_client = F1AnalyticsClient()

if 'selected_session' not in st.session_state:
    st.session_state.selected_session = None

if 'realtime_active' not in st.session_state:
    st.session_state.realtime_active = False

# Helper functions
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_sessions():
    """Load available sessions"""
    try:
        response = st.session_state.api_client.get_sessions()
        if response.success:
            return response.data
    except Exception as e:
        st.error(f"Failed to load sessions: {e}")
    return []

@st.cache_data(ttl=300)
def load_session_summary(session_id: str):
    """Load session summary"""
    try:
        response = st.session_state.api_client.get_session_summary(session_id)
        if response.success:
            return response.data
    except Exception as e:
        st.error(f"Failed to load session summary: {e}")
    return None

@st.cache_data(ttl=300)
def load_performance_data(session_id: str):
    """Load performance analysis data"""
    try:
        response = st.session_state.api_client.get_performance_analysis(session_id)
        if response.success:
            return response.data
    except Exception as e:
        st.error(f"Failed to load performance data: {e}")
    return []

@st.cache_data(ttl=300)
def load_lap_times(session_id: str, driver_code: str = None):
    """Load lap times data"""
    try:
        response = st.session_state.api_client.get_lap_times(session_id, driver_code=driver_code)
        if response.success:
            return pd.DataFrame(response.data['lap_times'])
    except Exception as e:
        st.error(f"Failed to load lap times: {e}")
    return pd.DataFrame()

def create_lap_time_chart(lap_times_df: pd.DataFrame, title: str = "Lap Times"):
    """Create lap time visualization"""
    if lap_times_df.empty:
        return None
    
    fig = px.line(
        lap_times_df,
        x='lap_number',
        y='lap_time',
        color='driver_code',
        title=title,
        labels={'lap_time': 'Lap Time (s)', 'lap_number': 'Lap Number'},
        hover_data=['tire_compound', 'tire_age'] if 'tire_compound' in lap_times_df.columns else None
    )
    
    fig.update_layout(
        height=500,
        hovermode='x unified'
    )
    
    return fig

def create_performance_comparison(performance_data: List[Dict]):
    """Create driver performance comparison chart"""
    if not performance_data:
        return None
    
    df = pd.DataFrame(performance_data)
    
    # Create subplot with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Pace score
    fig.add_trace(
        go.Bar(
            x=df['driver_code'],
            y=df['pace_score'],
            name='Pace Score (%)',
            marker_color='lightblue'
        ),
        secondary_y=False,
    )
    
    # Consistency score
    fig.add_trace(
        go.Scatter(
            x=df['driver_code'],
            y=df['consistency_score'],
            mode='markers+lines',
            name='Consistency (s)',
            marker=dict(size=10, color='orange'),
            line=dict(color='orange')
        ),
        secondary_y=True,
    )
    
    fig.update_xaxes(title_text="Driver")
    fig.update_yaxes(title_text="Pace Score (%)", secondary_y=False)
    fig.update_yaxes(title_text="Consistency Score (s)", secondary_y=True)
    fig.update_layout(title_text="Driver Performance Comparison", height=500)
    
    return fig

def create_tire_strategy_chart(lap_times_df: pd.DataFrame):
    """Create tire strategy visualization"""
    if lap_times_df.empty or 'tire_compound' not in lap_times_df.columns:
        return None
    
    # Calculate stint lengths by compound
    stint_data = []
    for driver in lap_times_df['driver_code'].unique():
        driver_laps = lap_times_df[lap_times_df['driver_code'] == driver].sort_values('lap_number')
        current_compound = None
        stint_start = None
        
        for _, lap in driver_laps.iterrows():
            if lap['tire_compound'] != current_compound:
                if current_compound is not None and stint_start is not None:
                    # End previous stint
                    stint_data.append({
                        'driver_code': driver,
                        'compound': current_compound,
                        'start_lap': stint_start,
                        'end_lap': lap['lap_number'] - 1,
                        'length': lap['lap_number'] - stint_start
                    })
                
                # Start new stint
                current_compound = lap['tire_compound']
                stint_start = lap['lap_number']
        
        # End final stint
        if current_compound is not None and stint_start is not None:
            stint_data.append({
                'driver_code': driver,
                'compound': current_compound,
                'start_lap': stint_start,
                'end_lap': driver_laps['lap_number'].max(),
                'length': driver_laps['lap_number'].max() - stint_start + 1
            })
    
    if not stint_data:
        return None
    
    stint_df = pd.DataFrame(stint_data)
    
    # Create timeline chart
    color_map = {
        'SOFT': 'red',
        'MEDIUM': 'yellow',
        'HARD': 'white',
        'INTERMEDIATE': 'green',
        'WET': 'blue'
    }
    
    fig = go.Figure()
    
    for i, driver in enumerate(stint_df['driver_code'].unique()):
        driver_stints = stint_df[stint_df['driver_code'] == driver]
        
        for _, stint in driver_stints.iterrows():
            fig.add_trace(go.Scatter(
                x=[stint['start_lap'], stint['end_lap']],
                y=[i, i],
                mode='lines',
                line=dict(
                    color=color_map.get(stint['compound'], 'gray'),
                    width=8
                ),
                name=f"{driver} - {stint['compound']}",
                showlegend=False,
                hovertemplate=f"<b>{driver}</b><br>" +
                            f"Compound: {stint['compound']}<br>" +
                            f"Laps: {stint['start_lap']}-{stint['end_lap']}<br>" +
                            f"Length: {stint['length']} laps<extra></extra>"
            ))
    
    fig.update_layout(
        title="Tire Strategy Timeline",
        xaxis_title="Lap Number",
        yaxis_title="Driver",
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(len(stint_df['driver_code'].unique()))),
            ticktext=list(stint_df['driver_code'].unique())
        ),
        height=500
    )
    
    return fig

def main():
    """Main dashboard application"""
    
    # Header
    st.markdown('<h1 class="main-header">🏎️ F1 Analytics Platform</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.markdown("## 🎛️ Control Panel")
    
    # Session selection
    sessions = load_sessions()
    if sessions:
        selected_session = st.sidebar.selectbox(
            "Select Session",
            options=sessions,
            index=0 if not st.session_state.selected_session else sessions.index(st.session_state.selected_session) if st.session_state.selected_session in sessions else 0
        )
        st.session_state.selected_session = selected_session
    else:
        st.sidebar.error("No sessions available")
        st.stop()
    
    # Page selection
    page = st.sidebar.radio(
        "Navigate to",
        ["📊 Overview", "🏁 Performance Analysis", "📈 Lap Times Analysis", "🔧 Data Quality", "🚀 Real-time Monitoring", "🤖 ML Predictions"]
    )
    
    # Main content based on page selection
    if page == "📊 Overview":
        show_overview_page()
    elif page == "🏁 Performance Analysis":
        show_performance_page()
    elif page == "📈 Lap Times Analysis":
        show_lap_times_page()
    elif page == "🔧 Data Quality":
        show_data_quality_page()
    elif page == "🚀 Real-time Monitoring":
        show_realtime_page()
    elif page == "🤖 ML Predictions":
        show_ml_predictions_page()

def show_overview_page():
    """Show overview dashboard"""
    st.header("📊 Session Overview")
    
    session_id = st.session_state.selected_session
    if not session_id:
        st.warning("Please select a session")
        return
    
    # Load session summary
    summary = load_session_summary(session_id)
    if not summary:
        st.error("Failed to load session summary")
        return
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Laps", summary.get('total_laps', 'N/A'))
    
    with col2:
        st.metric("Total Drivers", summary.get('total_drivers', 'N/A'))
    
    with col3:
        fastest_lap = summary.get('fastest_lap')
        if fastest_lap:
            st.metric("Fastest Lap", f"{fastest_lap:.3f}s")
        else:
            st.metric("Fastest Lap", "N/A")
    
    with col4:
        avg_lap_time = summary.get('avg_lap_time')
        if avg_lap_time:
            st.metric("Average Lap Time", f"{avg_lap_time:.3f}s")
        else:
            st.metric("Average Lap Time", "N/A")
    
    # Load performance data for charts
    performance_data = load_performance_data(session_id)
    lap_times_df = load_lap_times(session_id)
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        if performance_data:
            fig = create_performance_comparison(performance_data)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No performance data available")
    
    with col2:
        if not lap_times_df.empty:
            # Top 5 drivers by best lap time
            best_laps = lap_times_df.groupby('driver_code')['lap_time'].min().sort_values().head(5)
            fig = px.bar(
                x=best_laps.values,
                y=best_laps.index,
                orientation='h',
                title="Top 5 Fastest Drivers",
                labels={'x': 'Best Lap Time (s)', 'y': 'Driver'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No lap times data available")
    
    # Tire strategy overview
    if not lap_times_df.empty:
        st.subheader("🛞 Tire Strategy Overview")
        tire_fig = create_tire_strategy_chart(lap_times_df)
        if tire_fig:
            st.plotly_chart(tire_fig, use_container_width=True)
        else:
            st.info("No tire strategy data available")

def show_performance_page():
    """Show performance analysis page"""
    st.header("🏁 Performance Analysis")
    
    session_id = st.session_state.selected_session
    performance_data = load_performance_data(session_id)
    
    if not performance_data:
        st.warning("No performance data available for this session")
        return
    
    # Performance table
    df = pd.DataFrame(performance_data)
    df = df.sort_values('pace_score')  # Sort by pace (lower is better)
    
    st.subheader("Driver Performance Rankings")
    
    # Format the dataframe for display
    display_df = df.copy()
    
    # Safe formatting with null checks
    display_df['avg_lap_time'] = display_df['avg_lap_time'].apply(
        lambda x: f"{x:.3f}s" if pd.notna(x) else "N/A"
    )
    display_df['pace_score'] = display_df['pace_score'].apply(
        lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A"
    )
    display_df['consistency_score'] = display_df['consistency_score'].apply(
        lambda x: f"{x:.3f}s" if pd.notna(x) else "N/A"
    )
    display_df['tire_management'] = display_df['tire_management'].apply(
        lambda x: f"{x:.3f}" if pd.notna(x) else "N/A"
    )
    
    display_df.columns = ['Driver', 'Avg Lap Time', 'Pace Score', 'Consistency', 'Tire Management']
    st.dataframe(display_df, use_container_width=True)
    
    # Detailed analysis
    st.subheader("Detailed Performance Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Pace vs Consistency scatter plot with fixed size parameter
        # Convert tire_management to absolute values and normalize for size
        tire_mgmt_normalized = np.abs(df['tire_management'].fillna(0))
        tire_mgmt_size = (tire_mgmt_normalized - tire_mgmt_normalized.min()) / (tire_mgmt_normalized.max() - tire_mgmt_normalized.min() + 1e-6)
        tire_mgmt_size = tire_mgmt_size * 30 + 5  # Scale to reasonable size range (5-35)
        
        fig = px.scatter(
            df,
            x='pace_score',
            y='consistency_score',
            color='driver_code',
            title="Pace vs Consistency",
            labels={
                'pace_score': 'Pace Score (% off fastest)',
                'consistency_score': 'Consistency (lap time std dev)',
            },
            hover_data=['avg_lap_time', 'tire_management']
        )
        
        # Update marker sizes manually to avoid negative values
        fig.update_traces(marker=dict(size=tire_mgmt_size.tolist()))
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Top performers in each category
        st.markdown("**🏆 Category Leaders**")
        
        try:
            fastest = df.loc[df['pace_score'].idxmin()]
            most_consistent = df.loc[df['consistency_score'].idxmin()]
            best_tire_mgmt = df.loc[df['tire_management'].idxmax()]
            
            st.success(f"**Fastest Pace:** {fastest['driver_code']} ({fastest['pace_score']:.2f}% off)")
            st.success(f"**Most Consistent:** {most_consistent['driver_code']} ({most_consistent['consistency_score']:.3f}s std)")
            st.success(f"**Best Tire Management:** {best_tire_mgmt['driver_code']} ({best_tire_mgmt['tire_management']:.3f})")
        except Exception as e:
            st.error(f"Error calculating category leaders: {e}")
            st.info("Performance metrics may contain invalid data")

def show_lap_times_page():
    """Show lap times analysis page"""
    st.header("📈 Lap Times Analysis")
    
    session_id = st.session_state.selected_session
    
    # Driver filter
    lap_times_df = load_lap_times(session_id)
    if lap_times_df.empty:
        st.warning("No lap times data available")
        return
    
    drivers = ['All'] + sorted(lap_times_df['driver_code'].unique().tolist())
    selected_driver = st.selectbox("Select Driver", drivers)
    
    # Filter data
    if selected_driver != 'All':
        filtered_df = lap_times_df[lap_times_df['driver_code'] == selected_driver]
    else:
        filtered_df = lap_times_df
    
    # Lap times chart
    st.subheader("Lap Times Progression")
    lap_chart = create_lap_time_chart(filtered_df, f"Lap Times - {selected_driver}")
    if lap_chart:
        st.plotly_chart(lap_chart, use_container_width=True)
    
    # Statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Laps", len(filtered_df))
    
    with col2:
        if not filtered_df.empty:
            st.metric("Fastest Lap", f"{filtered_df['lap_time'].min():.3f}s")
    
    with col3:
        if not filtered_df.empty:
            st.metric("Average Lap Time", f"{filtered_df['lap_time'].mean():.3f}s")
    
    # Sector analysis
    if selected_driver != 'All' and not filtered_df.empty:
        st.subheader("Sector Time Analysis")
        
        sector_cols = ['sector_1_time', 'sector_2_time', 'sector_3_time']
        available_sectors = [col for col in sector_cols if col in filtered_df.columns and filtered_df[col].notna().any()]
        
        if available_sectors:
            sector_data = filtered_df[['lap_number'] + available_sectors].dropna()
            
            fig = go.Figure()
            colors = ['red', 'yellow', 'green']
            
            for i, sector in enumerate(available_sectors):
                fig.add_trace(go.Scatter(
                    x=sector_data['lap_number'],
                    y=sector_data[sector],
                    mode='lines+markers',
                    name=f"Sector {i+1}",
                    line=dict(color=colors[i])
                ))
            
            fig.update_layout(
                title="Sector Times Progression",
                xaxis_title="Lap Number",
                yaxis_title="Sector Time (s)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No sector time data available")

def show_data_quality_page():
    """Show data quality dashboard"""
    st.header("🔧 Data Quality Report")
    
    session_id = st.session_state.selected_session
    
    try:
        # Load quality report
        response = st.session_state.api_client.get_quality_report(session_id)
        
        if not response.success:
            st.error(f"Failed to load quality report: {response.message}")
            return
        
        quality_data = response.data
        
        # Overall status
        status = quality_data.get('overall_status', 'UNKNOWN')
        score = quality_data.get('quality_score', 0)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if status == "GOOD":
                st.success(f"**Status:** {status}")
            elif status == "FAIR":
                st.warning(f"**Status:** {status}")
            else:
                st.error(f"**Status:** {status}")
        
        with col2:
            st.metric("Quality Score", f"{score:.1f}%")
        
        with col3:
            passed = quality_data.get('passed_checks', 0)
            total = quality_data.get('total_checks', 0)
            st.metric("Checks Passed", f"{passed}/{total}")
        
        # Display detailed quality issues if available
        if 'issues' in quality_data:
            st.subheader("Quality Issues")
            for issue in quality_data['issues']:
                if issue.get('severity') == 'critical':
                    st.error(f"**Critical:** {issue.get('message', 'Unknown issue')}")
                elif issue.get('severity') == 'warning':
                    st.warning(f"**Warning:** {issue.get('message', 'Unknown issue')}")
                else:
                    st.info(f"**Info:** {issue.get('message', 'Unknown issue')}")
        
    except Exception as e:
        st.error(f"Error loading quality report: {e}")

def show_realtime_page():
    """Show real-time monitoring page"""
    st.header("🚀 Real-time Monitoring")
    
    session_id = st.session_state.selected_session
    
    # Real-time controls
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Start Real-time Processing"):
            try:
                response = st.session_state.api_client.start_realtime_processing(session_id)
                if response.success:
                    st.session_state.realtime_active = True
                    st.success("Real-time processing started!")
                else:
                    st.error(f"Failed to start: {response.message}")
            except Exception as e:
                st.error(f"Error starting real-time processing: {e}")
    
    with col2:
        if st.button("Stop Real-time Processing"):
            try:
                response = st.session_state.api_client.stop_realtime_processing(session_id)
                if response.success:
                    st.session_state.realtime_active = False
                    st.success("Real-time processing stopped!")
                else:
                    st.error(f"Failed to stop: {response.message}")
            except Exception as e:
                st.error(f"Error stopping real-time processing: {e}")
    
    if st.session_state.realtime_active:
        # Auto-refresh placeholder
        placeholder = st.empty()
        
        try:
            # Live leaderboard
            response = st.session_state.api_client.get_live_leaderboard(session_id)
            if response.success:
                leaderboard = response.data.get('leaderboard', [])
                
                with placeholder.container():
                    st.subheader("🏁 Live Leaderboard")
                    
                    if leaderboard:
                        df = pd.DataFrame(leaderboard)
                        st.dataframe(df, use_container_width=True)
                    else:
                        st.info("No live data available yet...")
            else:
                st.error(f"Failed to load live data: {response.message}")
        except Exception as e:
            st.error(f"Error loading live data: {e}")
        
        # Auto-refresh (simplified - in production, use WebSockets)
        time.sleep(2)
        st.rerun()

def show_ml_predictions_page():
    """Show ML predictions page"""
    st.header("🤖 Machine Learning Predictions")
    
    session_id = st.session_state.selected_session
    
    # Prediction form
    st.subheader("Lap Time Prediction")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        driver_code = st.text_input("Driver Code", value="HAM")
        lap_number = st.number_input("Lap Number", min_value=1, max_value=100, value=10)
    
    with col2:
        tire_compound = st.selectbox("Tire Compound", ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"])
        tire_age = st.number_input("Tire Age", min_value=1, max_value=50, value=5)
    
    with col3:
        if st.button("Predict Lap Time"):
            try:
                response = st.session_state.api_client.predict_lap_time(
                    session_id, driver_code, lap_number, tire_compound, tire_age
                )
                
                if response.success:
                    prediction = response.data
                    st.success(f"Predicted Lap Time: {prediction['predicted_lap_time']:.3f}s")
                    confidence = prediction.get('confidence_interval', [0, 0])
                    st.info(f"Confidence Interval: {confidence[0]:.3f}s - {confidence[1]:.3f}s")
                else:
                    st.error(f"Prediction failed: {response.message}")
            except Exception as e:
                st.error(f"Error making prediction: {e}")
    
    # Model training section
    st.subheader("Model Training")
    
    sessions = load_sessions()
    if sessions:
        selected_sessions = st.multiselect("Select Sessions for Training", sessions)
        
        if st.button("Train Model") and selected_sessions:
            with st.spinner("Training model..."):
                try:
                    response = st.session_state.api_client.train_lap_time_model(selected_sessions)
                    
                    if response.success:
                        st.success("Model training completed!")
                        
                        # Show training results
                        performances = response.data.get('model_performances', {})
                        
                        st.subheader("Training Results")
                        for model_name, metrics in performances.items():
                            st.write(f"**{model_name}:**")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("MAE", f"{metrics.get('mae', 0):.3f}s")
                            with col2:
                                st.metric("RMSE", f"{metrics.get('rmse', 0):.3f}s")
                            with col3:
                                st.metric("R² Score", f"{metrics.get('r2_score', 0):.3f}")
                    else:
                        st.error(f"Training failed: {response.message}")
                except Exception as e:
                    st.error(f"Error training model: {e}")

if __name__ == "__main__":
    main()