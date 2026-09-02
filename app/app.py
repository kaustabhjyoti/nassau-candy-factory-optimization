import streamlit as st
import pandas as pd 
import numpy as np 
import joblib
import os

st.set_page_config(page_title = "Nassau Candy Factory Optimizer", layout = "wide")

st.markdown("""
<style>
    .stMetric {
        background-color: rgba(255, 99, 71, 0.1);
        padding: 10px;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🍬 Nassau Candy Factory Reallocation & Shipping Optimizer")
st.markdown("Decision intelligence for factory-product reassignment and shipping efficiency.")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "processed_data.csv"))
    route_clusters = pd.read_csv(os.path.join(DATA_DIR, "route_clusters.csv"))
    recommendations = pd.read_csv(os.path.join(DATA_DIR, "recommendations_final.csv"))
    simulation_results = pd.read_csv(os.path.join(DATA_DIR, "simulation_results.csv"))
    kpi_summary = pd.read_csv(os.path.join(DATA_DIR, "kpi_summary.csv"))
    return df, route_clusters, recommendations, simulation_results, kpi_summary

df, route_clusters, recommendations, simulation_results, kpi_summary = load_data()


factories = pd.DataFrame({
    'Factory': ["Lot's O' Nuts", "Wicked Choccy's", "Sugar Shack", "Secret Factory", "The Other Factory"],
    'Factory Lat': [32.881893, 32.076176, 48.119140, 41.446333, 35.117500],
    'Factory Lon': [-111.768036, -81.088371, -96.181150, -90.565487, -89.971107]
})

def haversine_distance(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371
    return c * r

def simulate_factory_options(row):
    lr_model_local = joblib.load(os.path.join(BASE_DIR, "..", "models", "lead_time_model.pkl"))
    model_features_local = joblib.load(os.path.join(BASE_DIR, "..", "models", "model_features.pkl"))
    results = []
    for _, factory_row in factories.iterrows():
        distance = haversine_distance(
            factory_row['Factory Lat'], factory_row['Factory Lon'],
            row['Customer Lat'], row['Customer Lon']
        )
        input_dict = {col: 0 for col in model_features_local}
        input_dict['Distance (km)'] = distance
        for col_prefix, value in [('Ship Mode_', row['Ship Mode']), ('Region_', row['Region']), ('Division_', row['Division'])]:
            target_col = col_prefix + value
            if target_col in input_dict:
                input_dict[target_col] = 1
        input_df = pd.DataFrame([input_dict])[model_features_local]
        predicted_lead_time = lr_model_local.predict(input_df)[0]
        results.append({'Factory': factory_row['Factory'], 'Distance (km)': round(distance, 1), 'Predicted Lead Time': predicted_lead_time})
    return pd.DataFrame(results)

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["📊 Overview & KPIs", "🏭 Factory Optimization Simulator", "🔄 What-If Scenario Analysis", "📋 Recommendation Dashboard", "⚠️ Risk & Impact Panel"]
)

if page == "📊 Overview & KPIs":
    st.header("Project Overview")
    st.markdown("""
    Nassau Candy currently assigns products to factories using static rules, leading to suboptimal 
    shipping distances and lead times. This dashboard simulates factory-product reassignment scenarios 
    to recommend optimal configurations.
    """)
    
    st.subheader("Key Performance Indicators")
    
    # Display KPIs as metric cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Lead Time Reduction", kpi_summary.iloc[0]['Value'])
    with col2:
        st.metric("Profit Impact", "Stable")
        st.caption(f"Distance-profit correlation: 0.007")
    with col3:
        st.metric("Confidence Score", "0.894")
        st.caption("Avg, adjusted for low-volume scenarios")
    with col4:
        st.metric("Recommendation Coverage", kpi_summary.iloc[3]['Value'])
    
    st.subheader("Factory Performance Summary")
    factory_summary = df.groupby('Origin Factory').agg(
        Avg_Lead_Time=('Synthetic Lead Time', 'mean'),
        Order_Count=('Order ID', 'count'),
        Avg_Gross_Profit=('Gross Profit', 'mean')
    ).reset_index()
    st.dataframe(factory_summary, use_container_width=True)
    st.subheader("Factory Locations")
    map_data = factories.rename(columns={'Factory Lat': 'lat', 'Factory Lon': 'lon'})
    st.map(map_data, size=50000, color='#FF6347')

elif page == "🏭 Factory Optimization Simulator":
    st.header("Factory Optimization Simulator")
    st.markdown("Select a product to see predicted shipping performance across all 5 factories.")
    
    # Load the model and features (needed for live prediction)
    lr_model = joblib.load(os.path.join(BASE_DIR, "..", "models", "lead_time_model.pkl"))
    model_features = joblib.load(os.path.join(BASE_DIR, "..", "models", "model_features.pkl"))
    
    # User selects a product
    product_list = sorted(df['Product Name'].unique())
    selected_product = st.selectbox("Select Product", product_list)
    
    # User selects a destination region
    region_list = sorted(df['Region'].unique())
    selected_region = st.selectbox("Select Destination Region", region_list)
    
    # User selects ship mode
    ship_mode_list = sorted(df['Ship Mode'].unique())
    selected_ship_mode = st.selectbox("Select Ship Mode", ship_mode_list)
    
    if st.button("Run Simulation"):
        # Get the division for this product (needed for the model input)
        division = df[df['Product Name'] == selected_product]['Division'].iloc[0]
        
        # Build a representative row matching the user's selections
        sim_row = pd.Series({
            'Ship Mode': selected_ship_mode,
            'Region': selected_region,
            'Division': division,
            'Customer Lat': df[df['Region'] == selected_region]['Customer Lat'].mean(),
            'Customer Lon': df[df['Region'] == selected_region]['Customer Lon'].mean()
        })
        
        results = simulate_factory_options(sim_row)
        results_sorted = results.sort_values('Predicted Lead Time')
        
        st.subheader(f"Predicted Performance for {selected_product}")
        st.dataframe(results_sorted, use_container_width=True)
        
        best = results_sorted.iloc[0]
        st.success(f"✅ Fastest option: **{best['Factory']}** — {best['Predicted Lead Time']:.2f} days")

elif page == "🔄 What-If Scenario Analysis":
    st.header("What-If Scenario Analysis")
    st.markdown("Compare current factory assignments against optimized recommendations.")
    
    # Let user filter by original factory
    factory_filter = st.selectbox("Filter by Current Factory", ["All"] + sorted(recommendations['Original Factory'].unique().tolist()))
    
    filtered = recommendations.copy()
    if factory_filter != "All":
        filtered = filtered[filtered['Original Factory'] == factory_filter]
    
    # Only show scenarios where reassignment actually helps
    show_only_improvements = st.checkbox("Show only scenarios with improvement", value=True)
    if show_only_improvements:
        filtered = filtered[filtered['Improvement (days)'] > 0.01]
    
    filtered_sorted = filtered.sort_values('Improvement (days)', ascending=False)
    
    st.subheader(f"Showing {len(filtered_sorted)} scenarios")
    
    # Show as a comparison table
    display_cols = ['Original Factory', 'Factory', 'Region', 'Ship Mode', 'Division', 
                     'Current Lead Time', 'Predicted Lead Time', 'Improvement (days)', 'Improvement (%)']
    st.dataframe(
        filtered_sorted[display_cols].rename(columns={'Factory': 'Recommended Factory'}),
        use_container_width=True
    )
    
    # Visual comparison chart
    if len(filtered_sorted) > 0:
        st.subheader("Lead Time Comparison (Top 10)")
        top10 = filtered_sorted.head(10).copy()
        
        # Build a unique label per row so bars don't collapse into each other
        top10['Scenario'] = top10['Original Factory'] + " → " + top10['Region'] + " (" + top10['Ship Mode'] + ")"
        
        chart_data = top10.set_index('Scenario')[['Current Lead Time', 'Predicted Lead Time']]
        import plotly.express as px

        fig = px.bar(
            top10, x='Scenario', y=['Current Lead Time', 'Predicted Lead Time'],
            barmode='group', title="Current vs Predicted Lead Time",
            labels={'value': 'Lead Time (days)', 'variable': 'Metric'}
        )
        st.plotly_chart(fig, use_container_width=True)

elif page == "📋 Recommendation Dashboard":
    st.header("Recommendation Dashboard")
    st.markdown("Ranked factory reassignment recommendations by impact.")
    
    # Sort by biggest improvement, show only genuine improvements
    top_recs = recommendations[recommendations['Improvement (days)'] > 0.01].sort_values('Improvement (days)', ascending=False)
    
    st.subheader(f"Top Reassignment Opportunities ({len(top_recs)} total)")
    
    # Let user choose how many to see
    n_show = st.slider("Number of recommendations to show", 5, len(top_recs), 10)
    
    for idx, rec in top_recs.head(n_show).iterrows():
        with st.container():
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.markdown(f"**{rec['Original Factory']}** → **{rec['Factory']}**")
                st.caption(f"{rec['Region']} | {rec['Ship Mode']} | {rec['Division']}")
            with col2:
                st.metric("Improvement", f"{rec['Improvement (days)']:.2f} days")
            with col3:
                st.metric("Reduction", f"{rec['Improvement (%)']:.1f}%")
            st.divider()

elif page == "⚠️ Risk & Impact Panel":
    st.header("Risk & Impact Panel")
    st.markdown("Flagging reassignments that carry higher uncertainty, and confirming profit safety.")
    
    # Reload the confidence-scored recommendations
    recs_confidence = pd.read_csv(os.path.join(DATA_DIR, "recommendations_final.csv"))
    
    st.subheader("Profit Impact Check")
    profit_corr = df['Distance (km)'].corr(df['Gross Profit'])
    if abs(profit_corr) < 0.1:
        st.success(f"✅ Profit Impact: Stable. Correlation between distance and gross profit is {profit_corr:.3f} (near-zero), confirming factory reassignment does not meaningfully affect profitability.")
    else:
        st.warning(f"⚠️ Profit Impact: Distance shows a correlation of {profit_corr:.3f} with gross profit — review before acting on recommendations.")
    
    st.subheader("High-Risk Recommendations")
    st.markdown("Recommendations backed by fewer than 20 historical orders carry lower confidence.")
    
    high_risk = recs_confidence[recs_confidence['Supporting Orders'] < 20].sort_values('Improvement (days)', ascending=False)
    low_risk = recs_confidence[recs_confidence['Supporting Orders'] >= 20]
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("High-Risk Scenarios", len(high_risk))
    with col2:
        st.metric("High-Confidence Scenarios", len(low_risk))
    
    if len(high_risk) > 0:
        st.dataframe(
            high_risk[['Original Factory', 'Factory', 'Region', 'Ship Mode', 'Division', 'Supporting Orders', 'Confidence Score', 'Improvement (days)']].rename(
                columns={'Factory': 'Recommended Factory'}
            ),
            use_container_width=True
        )
    
    st.subheader("Model Reliability")
    st.info(f"The underlying prediction model explains {kpi_summary.iloc[2]['Value']} of lead-time variation (R² = 0.894). Recommendations for high-volume scenarios (≥20 historical orders) carry the full model confidence; low-volume scenarios are flagged above for manual review before acting.")