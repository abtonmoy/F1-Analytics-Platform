import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional

def create_gauge_chart(value: float, title: str, min_val: float = 0, max_val: float = 100, 
                      color_ranges: Optional[List[Dict]] = None) -> go.Figure:
    """Create a gauge chart for metrics"""
    
    if color_ranges is None:
        color_ranges = [
            {"range": [0, 50], "color": "red"},
            {"range": [50, 80], "color": "yellow"},
            {"range": [80, 100], "color": "green"}
        ]
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title},
        delta = {'reference': (min_val + max_val) / 2},
        gauge = {
            'axis': {'range': [min_val, max_val]},
            'bar': {'color': "darkblue"},
            'steps': [{'range': [min_val, max_val], 'color': "lightgray"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': max_val * 0.9
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def create_heatmap(data: pd.DataFrame, x_col: str, y_col: str, z_col: str, 
                  title: str = "Heatmap") -> go.Figure:
    """Create a heatmap visualization"""
    
    pivot_data = data.pivot(index=y_col, columns=x_col, values=z_col)
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot_data.values,
        x=pivot_data.columns,
        y=pivot_data.index,
        colorscale='RdYlBu_r',
        hoverongaps=False
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title=x_col,
        yaxis_title=y_col
    )
    
    return fig

def create_race_position_chart(lap_times_df: pd.DataFrame) -> go.Figure:
    """Create race position evolution chart"""
    
    if 'position' not in lap_times_df.columns:
        return None
    
    fig = go.Figure()
    
    # Get unique drivers and sort by final position
    drivers = lap_times_df.groupby('driver_code')['position'].last().sort_values().index
    
    colors = px.colors.qualitative.Set3[:len(drivers)]
    
    for i, driver in enumerate(drivers):
        driver_data = lap_times_df[lap_times_df['driver_code'] == driver].sort_values('lap_number')
        
        if not driver_data.empty:
            fig.add_trace(go.Scatter(
                x=driver_data['lap_number'],
                y=driver_data['position'],
                mode='lines+markers',
                name=driver,
                line=dict(color=colors[i % len(colors)], width=2),
                marker=dict(size=4)
            ))
    
    fig.update_layout(
        title="Race Position Evolution",
        xaxis_title="Lap Number",
        yaxis_title="Position",
        yaxis=dict(autorange='reversed'),  # Position 1 at top
        height=500,
        hovermode='x unified'
    )
    
    return fig

def create_sector_comparison(lap_times_df: pd.DataFrame, drivers: List[str]) -> go.Figure:
    """Create sector time comparison chart"""
    
    sector_cols = ['sector_1_time', 'sector_2_time', 'sector_3_time']
    available_sectors = [col for col in sector_cols if col in lap_times_df.columns]
    
    if not available_sectors:
        return None
    
    fig = go.Figure()
    
    colors = ['red', 'yellow', 'green']
    
    for driver in drivers:
        driver_data = lap_times_df[lap_times_df['driver_code'] == driver]
        
        if driver_data.empty:
            continue
        
        # Calculate average sector times
        avg_sectors = []
        sector_names = []
        
        for i, sector in enumerate(available_sectors):
            avg_time = driver_data[sector].mean()
            if not pd.isna(avg_time):
                avg_sectors.append(avg_time)
                sector_names.append(f"Sector {i+1}")
        
        if avg_sectors:
            fig.add_trace(go.Bar(
                x=sector_names,
                y=avg_sectors,
                name=driver,
                text=[f"{t:.3f}s" for t in avg_sectors],
                textposition='auto'
            ))
    
    fig.update_layout(
        title="Average Sector Times Comparison",
        xaxis_title="Sector",
        yaxis_title="Time (seconds)",
        barmode='group',
        height=400
    )
    
    return fig

def create_tire_performance_analysis(lap_times_df: pd.DataFrame) -> go.Figure:
    """Create tire compound performance analysis"""
    
    if 'tire_compound' not in lap_times_df.columns:
        return None
    
    # Calculate performance by compound
    compound_stats = lap_times_df.groupby('tire_compound')['lap_time'].agg([
        'mean', 'min', 'max', 'std', 'count'
    ]).reset_index()
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Average Lap Time', 'Best Lap Time', 'Consistency (Std Dev)', 'Usage Count'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    compounds = compound_stats['tire_compound'].tolist()
    colors = {'SOFT': 'red', 'MEDIUM': 'yellow', 'HARD': 'white', 'INTERMEDIATE': 'green', 'WET': 'blue'}
    
    # Average lap time
    fig.add_trace(
        go.Bar(x=compounds, y=compound_stats['mean'], 
               marker_color=[colors.get(c, 'gray') for c in compounds],
               name='Avg Time', showlegend=False),
        row=1, col=1
    )
    
    # Best lap time
    fig.add_trace(
        go.Bar(x=compounds, y=compound_stats['min'],
               marker_color=[colors.get(c, 'gray') for c in compounds],
               name='Best Time', showlegend=False),
        row=1, col=2
    )
    
    # Consistency
    fig.add_trace(
        go.Bar(x=compounds, y=compound_stats['std'],
               marker_color=[colors.get(c, 'gray') for c in compounds],
               name='Consistency', showlegend=False),
        row=2, col=1
    )
    
    # Usage count
    fig.add_trace(
        go.Bar(x=compounds, y=compound_stats['count'],
               marker_color=[colors.get(c, 'gray') for c in compounds],
               name='Usage', showlegend=False),
        row=2, col=2
    )
    
    fig.update_layout(
        title_text="Tire Compound Performance Analysis",
        height=600
    )
    
    return fig

def display_metric_cards(metrics: Dict[str, Any]):
    """Display metrics in card format"""
    
    cols = st.columns(len(metrics))
    
    for i, (key, value) in enumerate(metrics.items()):
        with cols[i]:
            if isinstance(value, dict):
                st.markdown(f"""
                <div class="metric-card">
                    <h4>{key}</h4>
                    <h2>{value.get('value', 'N/A')}</h2>
                    <p>{value.get('description', '')}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>{key}</h4>
                    <h2>{value}</h2>
                </div>
                """, unsafe_allow_html=True)

def create_correlation_matrix(lap_times_df: pd.DataFrame) -> go.Figure:
    """Create correlation matrix for numeric columns"""
    
    numeric_cols = lap_times_df.select_dtypes(include=[np.number]).columns
    correlation_matrix = lap_times_df[numeric_cols].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=correlation_matrix.round(2).values,
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title="Feature Correlation Matrix",
        width=600,
        height=600
    )
    
    return fig

class DashboardState:
    """Manage dashboard state and caching"""
    
    @staticmethod
    def initialize_session_state():
        """Initialize session state variables"""
        if 'selected_drivers' not in st.session_state:
            st.session_state.selected_drivers = []
        
        if 'comparison_mode' not in st.session_state:
            st.session_state.comparison_mode = False
        
        if 'auto_refresh' not in st.session_state:
            st.session_state.auto_refresh = False
    
    @staticmethod
    def clear_cache():
        """Clear Streamlit cache"""
        st.cache_data.clear()
        st.success("Cache cleared!")
    
    @staticmethod
    def export_data(data: pd.DataFrame, filename: str):
        """Export data to CSV"""
        csv = data.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=filename,
            mime='text/csv'
        )