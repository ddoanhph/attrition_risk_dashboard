import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import GradientBoostingClassifier
from typing import Dict, List
from collections import Counter

# [Previous color palette and configuration remain the same]

@st.cache_data
def load_advanced_data() -> pd.DataFrame:
    """
    Create comprehensive synthetic data for individual-level attrition risk analysis
    """
    np.random.seed(42)
    n_samples = 1200

    divisions = [
        'Technology', 'Operations', 'Marketing', 
        'Finance', 'HR', 'Sales', 'Engineering', 
        'Product Development', 'Customer Support'
    ]
    
    bands = ['Junior', 'Mid-Level', 'Senior', 'Expert', 'Leadership']
    age_groups = ['22-30', '31-40', '41-50', '51+']

    data = {
        'Employee_ID': [f'EMP{i:04d}' for i in range(1, n_samples + 1)],
        'Division': np.random.choice(divisions, size=n_samples),
        'Band': np.random.choice(bands, size=n_samples),
        'Age_Group': np.random.choice(age_groups, size=n_samples),
        'Career_Velocity': np.random.uniform(0.1, 2.0, size=n_samples),
        'Role_Stability': np.random.uniform(0, 1, size=n_samples),
        'Career_Growth_Potential': np.random.uniform(0.1, 1.0, size=n_samples),
        'External_Opportunities': np.random.uniform(0.1, 1.0, size=n_samples),
        'Work_Satisfaction': np.random.uniform(0.1, 1.0, size=n_samples),
        'Team_Size': np.random.randint(5, 75, size=n_samples)
    }

    df = pd.DataFrame(data)

    # Simulate attrition risk calculation
    df['Attrition_Risk'] = (
        0.25 * (1 - df['Role_Stability']) + 
        0.2 * (1 - df['Career_Growth_Potential']) + 
        0.15 * df['External_Opportunities'] + 
        0.1 * (1 - df['Work_Satisfaction']) + 
        0.3 * np.random.uniform(0, 1, size=n_samples)
    )

    # Risk categorization
    df['Risk_Category'] = pd.cut(
        df['Attrition_Risk'],
        bins=[0, 0.3, 0.6, 1],
        labels=['Low', 'Medium', 'High']
    )

    return df

def generate_strategic_insights(df: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    """
    Generate comprehensive strategic insights for different risk levels and divisions
    """
    risk_insights = {
        'High': {
            'Technology': "🚨 Immediate Intervention Required: Design retention programs, conduct stay interviews, review compensation and career paths.",
            'Operations': "🔍 Critical Risk Zone: Enhance role clarity, provide skill development, and create internal mobility opportunities.",
            'Marketing': "💡 Talent Preservation Strategy: Focus on creative autonomy, innovation platforms, and leadership development.",
        },
        'Medium': {
            'Finance': "📊 Proactive Engagement: Implement mentorship programs, review performance recognition, and provide challenging projects.",
            'Sales': "💼 Career Momentum Initiatives: Create clear progression paths, introduce performance-based incentives.",
            'HR': "👥 Talent Development Focus: Strengthen internal growth opportunities, design targeted training programs.",
        },
        'Low': {
            'Engineering': "🛠️ Maintenance Mode: Continue current engagement strategies, periodic pulse surveys, maintain positive work environment.",
            'Product Development': "🚀 Continuous Improvement: Regular skill upgrades, innovation challenges, cross-functional exposure.",
        }
    }
    return risk_insights

def analyze_divisional_risks(df: pd.DataFrame):
    """
    Comprehensive divisional risk analysis
    """
    division_risk = df.groupby('Division').agg({
        'Attrition_Risk': ['mean', 'count'],
        'Risk_Category': lambda x: x.value_counts(normalize=True).get('High', 0) * 100
    }).reset_index()
    
    division_risk.columns = ['Division', 'Avg_Risk', 'Total_Employees', 'High_Risk_Percentage']
    division_risk = division_risk.sort_values('High_Risk_Percentage', ascending=False)
    
    return division_risk

def main():
    # Load and analyze data
    df = load_advanced_data()
    division_risk = analyze_divisional_risks(df)
    strategic_insights = generate_strategic_insights(df)

    # Streamlit App Configuration
    st.set_page_config(
        page_title="Talent Risk Management Dashboard",
        page_icon="🛡️",
        layout="wide"
    )

    # Sidebar Filters
    st.sidebar.header("🔍 Analysis Filters")
    selected_divisions = st.sidebar.multiselect(
        "Select Divisions", 
        options=df['Division'].unique(), 
        default=df['Division'].unique()
    )
    
    risk_filter = st.sidebar.multiselect(
        "Filter by Risk Category", 
        options=['Low', 'Medium', 'High'], 
        default=['High', 'Medium', 'Low']
    )

    # Filter DataFrame
    filtered_df = df[
        (df['Division'].isin(selected_divisions)) & 
        (df['Risk_Category'].isin(risk_filter))
    ]

    # Main Dashboard
    st.title('🏆 Talent Risk Management Dashboard')
    
    # Key Metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_employees = len(filtered_df)
        high_risk_count = len(filtered_df[filtered_df['Risk_Category'] == 'High'])
        st.metric("Total Employees", total_employees, "")
    
    with col2:
        high_risk_percentage = (high_risk_count / total_employees * 100) if total_employees > 0 else 0
        st.metric("High-Risk Employees", f"{high_risk_count} ({high_risk_percentage:.1f}%)", 
                  f"Risk Level: {'🔴 Critical' if high_risk_percentage > 15 else '🟠 Moderate'}")
    
    with col3:
        medium_risk_count = len(filtered_df[filtered_df['Risk_Category'] == 'Medium'])
        medium_risk_percentage = (medium_risk_count / total_employees * 100) if total_employees > 0 else 0
        st.metric("Medium-Risk Employees", f"{medium_risk_count} ({medium_risk_percentage:.1f}%)", 
                  f"Potential Intervention: {'🟡 Recommended' if medium_risk_percentage > 20 else '🟢 Stable'}")

    # Tabs for Detailed Analysis
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔥 Risk Overview", 
        "📊 Divisional Analysis", 
        "🧩 Risk Factors", 
        "🚀 Strategic Recommendations"
    ])

    with tab1:
        st.subheader('📈 Risk Distribution')
        
        # Risk Category Pie Chart
        risk_distribution = filtered_df['Risk_Category'].value_counts()
        fig = px.pie(
            names=risk_distribution.index, 
            values=risk_distribution.values,
            title='Employee Risk Category Distribution',
            color_discrete_sequence=['red', 'orange', 'green']
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader('🔬 Divisional Risk Breakdown')
        
        # Divisional Risk Table
        st.dataframe(
            division_risk[division_risk['Division'].isin(selected_divisions)].style
            .format({
                'Avg_Risk': '{:.2f}',
                'High_Risk_Percentage': '{:.1f}%'
            })
            .background_gradient(cmap='Reds')
        )

    with tab3:
        st.subheader('🎯 Key Risk Factors')
        
        # Risk Factor Analysis
        factor_columns = [
            'Career_Velocity', 'Role_Stability', 
            'Career_Growth_Potential', 'External_Opportunities'
        ]
        correlation_matrix = filtered_df[factor_columns + ['Attrition_Risk']].corr()
        
        fig = px.imshow(
            correlation_matrix, 
            labels=dict(x="Factors", y="Factors", color="Correlation"),
            title='Correlation of Risk Factors',
            color_continuous_scale='RdBu_r'
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader('🎯 Strategic Talent Management')
        
        # Top Risk Division Recommendations
        if not division_risk.empty:
            top_risk_division = division_risk.iloc[0]['Division']
            top_risk_percentage = division_risk.iloc[0]['High_Risk_Percentage']
            
            st.warning(f"**Top Risk Division: {top_risk_division}**")
            st.markdown(f"🚨 High-Risk Percentage: {top_risk_percentage:.1f}%")
            
            risk_level = 'High' if top_risk_percentage > 15 else 'Medium'
            st.info(strategic_insights.get(risk_level, {}).get(top_risk_division, "No specific insights available"))

    # High-Risk Employees Table
    st.subheader('🚨 High-Risk Employees Detailed View')
    high_risk_employees = filtered_df[filtered_df['Risk_Category'] == 'High'].sort_values('Attrition_Risk', ascending=False).head(10)
    
    if not high_risk_employees.empty:
        st.dataframe(high_risk_employees, use_container_width=True)
    else:
        st.info("No high-risk employees found with current filters.")

if __name__ == "__main__":
    main()
