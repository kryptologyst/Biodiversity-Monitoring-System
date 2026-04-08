"""Streamlit demo for biodiversity monitoring system."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
import os
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Biodiversity Monitoring System",
    page_icon="🦎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2E86AB;
    }
    .species-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def load_data():
    """Load demo data."""
    try:
        demo_data = pd.read_csv("assets/demo_data.csv")
        return demo_data
    except FileNotFoundError:
        st.error("Demo data not found. Please run the training script first.")
        return None

def create_species_map(data):
    """Create interactive map showing species detections."""
    # Create base map
    m = folium.Map(
        location=[data['latitude'].mean(), data['longitude'].mean()],
        zoom_start=10,
        tiles='OpenStreetMap'
    )
    
    # Species colors
    species_colors = {
        'bird_present': 'blue',
        'monkey_present': 'green', 
        'insect_present': 'orange',
        'reptile_present': 'red',
        'amphibian_present': 'purple'
    }
    
    # Add markers for each species detection
    for idx, row in data.iterrows():
        # Check which species are present
        present_species = []
        for species in ['bird', 'monkey', 'insect', 'reptile', 'amphibian']:
            if row.get(f'{species}_present', 0) == 1:
                present_species.append(species)
        
        if present_species:
            # Create popup text
            popup_text = f"""
            <b>Location {idx}</b><br>
            <b>Species Detected:</b> {', '.join(present_species)}<br>
            <b>Temperature:</b> {row.get('temperature', 'N/A'):.1f}°C<br>
            <b>Humidity:</b> {row.get('humidity', 'N/A'):.2f}<br>
            <b>Time:</b> {row.get('time_of_day', 'N/A')}:00
            """
            
            # Choose color based on number of species
            if len(present_species) >= 3:
                color = 'red'
            elif len(present_species) == 2:
                color = 'orange'
            else:
                color = 'green'
            
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=8,
                popup=popup_text,
                color=color,
                fill=True,
                fillOpacity=0.7
            ).add_to(m)
    
    return m

def create_species_distribution_chart(data):
    """Create species distribution chart."""
    species_counts = {}
    for species in ['bird', 'monkey', 'insect', 'reptile', 'amphibian']:
        species_counts[species] = data[f'{species}_present'].sum()
    
    fig = px.bar(
        x=list(species_counts.keys()),
        y=list(species_counts.values()),
        title="Species Detection Counts",
        labels={'x': 'Species', 'y': 'Detection Count'},
        color=list(species_counts.values()),
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        showlegend=False,
        height=400
    )
    
    return fig

def create_environmental_conditions_chart(data):
    """Create environmental conditions chart."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Temperature Distribution', 'Humidity Distribution', 
                       'Sound Activity Distribution', 'Vegetation Index Distribution'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Temperature
    fig.add_trace(
        go.Histogram(x=data['temperature'], name='Temperature', nbinsx=20),
        row=1, col=1
    )
    
    # Humidity
    fig.add_trace(
        go.Histogram(x=data['humidity'], name='Humidity', nbinsx=20),
        row=1, col=2
    )
    
    # Sound Activity
    fig.add_trace(
        go.Histogram(x=data['sound_activity'], name='Sound Activity', nbinsx=20),
        row=2, col=1
    )
    
    # Vegetation Index
    fig.add_trace(
        go.Histogram(x=data['vegetation_index'], name='Vegetation Index', nbinsx=20),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=False)
    
    return fig

def create_time_series_chart(data):
    """Create time series chart of species activity."""
    # Group by hour
    hourly_data = data.groupby('time_of_day').agg({
        'bird_present': 'sum',
        'monkey_present': 'sum', 
        'insect_present': 'sum',
        'reptile_present': 'sum',
        'amphibian_present': 'sum'
    }).reset_index()
    
    fig = go.Figure()
    
    species_colors = {
        'bird_present': '#1f77b4',
        'monkey_present': '#ff7f0e',
        'insect_present': '#2ca02c',
        'reptile_present': '#d62728',
        'amphibian_present': '#9467bd'
    }
    
    for species in ['bird', 'monkey', 'insect', 'reptile', 'amphibian']:
        fig.add_trace(go.Scatter(
            x=hourly_data['time_of_day'],
            y=hourly_data[f'{species}_present'],
            mode='lines+markers',
            name=species.title(),
            line=dict(color=species_colors[f'{species}_present'])
        ))
    
    fig.update_layout(
        title="Species Activity by Hour of Day",
        xaxis_title="Hour of Day",
        yaxis_title="Detection Count",
        height=400
    )
    
    return fig

def main():
    """Main demo application."""
    # Header
    st.markdown('<h1 class="main-header">🦎 Biodiversity Monitoring System</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    This interactive demo showcases a biodiversity monitoring system that detects 
    species presence using environmental sensor data. The system uses machine learning 
    models to predict the presence of different species based on environmental conditions.
    """)
    
    # Load data
    data = load_data()
    if data is None:
        return
    
    # Sidebar
    st.sidebar.title("🎛️ Controls")
    
    # Filter options
    st.sidebar.subheader("Filters")
    
    # Temperature filter
    temp_range = st.sidebar.slider(
        "Temperature Range (°C)",
        min_value=float(data['temperature'].min()),
        max_value=float(data['temperature'].max()),
        value=(float(data['temperature'].min()), float(data['temperature'].max()))
    )
    
    # Time filter
    time_range = st.sidebar.slider(
        "Time of Day",
        min_value=int(data['time_of_day'].min()),
        max_value=int(data['time_of_day'].max()),
        value=(int(data['time_of_day'].min()), int(data['time_of_day'].max()))
    )
    
    # Species filter
    selected_species = st.sidebar.multiselect(
        "Species to Display",
        options=['bird', 'monkey', 'insect', 'reptile', 'amphibian'],
        default=['bird', 'monkey', 'insect', 'reptile', 'amphibian']
    )
    
    # Apply filters
    filtered_data = data[
        (data['temperature'] >= temp_range[0]) & 
        (data['temperature'] <= temp_range[1]) &
        (data['time_of_day'] >= time_range[0]) &
        (data['time_of_day'] <= time_range[1])
    ]
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🗺️ Species Detection Map")
        
        # Create and display map
        species_map = create_species_map(filtered_data)
        st_folium(species_map, width=700, height=500)
        
        # Map legend
        st.markdown("""
        **Map Legend:**
        - 🟢 Green: 1 species detected
        - 🟠 Orange: 2 species detected  
        - 🔴 Red: 3+ species detected
        """)
    
    with col2:
        st.subheader("📊 Quick Stats")
        
        # Display metrics
        total_detections = filtered_data[[f'{s}_present' for s in selected_species]].sum().sum()
        avg_temp = filtered_data['temperature'].mean()
        avg_humidity = filtered_data['humidity'].mean()
        
        st.metric("Total Detections", f"{total_detections:,}")
        st.metric("Avg Temperature", f"{avg_temp:.1f}°C")
        st.metric("Avg Humidity", f"{avg_humidity:.2f}")
        
        # Species breakdown
        st.subheader("🦎 Species Breakdown")
        for species in selected_species:
            count = filtered_data[f'{species}_present'].sum()
            percentage = (count / len(filtered_data)) * 100
            st.markdown(f"**{species.title()}:** {count} ({percentage:.1f}%)")
    
    # Charts section
    st.subheader("📈 Analysis Charts")
    
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["Species Distribution", "Environmental Conditions", "Temporal Patterns"])
    
    with tab1:
        species_chart = create_species_distribution_chart(filtered_data)
        st.plotly_chart(species_chart, use_container_width=True)
    
    with tab2:
        env_chart = create_environmental_conditions_chart(filtered_data)
        st.plotly_chart(env_chart, use_container_width=True)
    
    with tab3:
        time_chart = create_time_series_chart(filtered_data)
        st.plotly_chart(time_chart, use_container_width=True)
    
    # Model performance section
    st.subheader("🤖 Model Performance")
    
    # Try to load leaderboard
    try:
        leaderboard = pd.read_csv("assets/model_leaderboard.csv", index_col=0)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("**Model Leaderboard:**")
            st.dataframe(
                leaderboard[['rank', 'accuracy', 'f1_macro', 'jaccard_score']].head(),
                use_container_width=True
            )
        
        with col2:
            # Create performance chart
            perf_fig = px.bar(
                leaderboard.head(),
                x=leaderboard.head().index,
                y='accuracy',
                title="Model Accuracy Comparison",
                labels={'x': 'Model', 'y': 'Accuracy'}
            )
            st.plotly_chart(perf_fig, use_container_width=True)
    
    except FileNotFoundError:
        st.info("Model performance data not available. Run the training script to generate results.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **About this Demo:**
    
    This biodiversity monitoring system demonstrates how environmental sensor data can be used 
    to detect species presence using machine learning. The system uses multiple models including 
    Random Forest, XGBoost, LightGBM, and Neural Networks to predict species presence based on 
    environmental conditions like temperature, humidity, sound activity, and vegetation index.
    
    **Author:** kryptologyst - [GitHub](https://github.com/kryptologyst)
    
    **Disclaimer:** This is a research demonstration using synthetic data. 
    For operational use, please ensure proper validation and compliance with local regulations.
    """)

if __name__ == "__main__":
    main()
