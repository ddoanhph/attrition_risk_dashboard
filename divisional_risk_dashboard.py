import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import GradientBoostingClassifier
from typing import Dict, List

# Airbus Color Palette
AIRBUS_COLORS = {
    'primary_blue': '#00205B',   # Dark Airbus Blue
    'secondary_blue': '#5BA3D4', # Lighter Airbus Blue
    'grey': '#6D6E71',           # Airbus Grey
    'light_blue': '#A4D4E9',     # Light Blue
    'accent_blue': '#00A3E1',    # Bright Accent Blue
}

# Set page configuration
st.set_page_config(
    page_title="Divisional Attrition Risk Analysis",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Airbus theme
st.markdown(f"""
<style>
    .stApp {{
        background-color: white;
    }}
    .stTabs [data-baseweb="tab-list"] {{
        gap: 10px;
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: {AIRBUS_COLORS['light_blue']};
        color: {AIRBUS_COLORS['primary_blue']};
        border-radius: 4px;
    }}
    .stTabs [data-baseweb="tab"][data-selected="true"] {{
        background-color: {AIRBUS_COLORS['primary_blue']};
        color: white;
    }}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_division_data() -> pd.DataFrame:
    """
    Create comprehensive synthetic data for divisional-level attrition risk analysis
    
    Enhanced data generation with more nuanced risk factors:
    - Expanded division complexity
    - More granular strategic importance
    - Additional contextual factors
    """
    np.random.seed(42)
    n_samples = 1000

    divisions = [
        'Technology', 'Operations', 'Marketing', 
        'Finance', 'HR', 'Sales', 'Engineering', 
        'Product Development', 'Customer Support'
    ]
    
    # More nuanced strategic importance mapping
    strategic_importance = {
        'Technology': 0.9,
        'Product Development': 0.85,
        'Engineering': 0.8,
        'Operations': 0.75,
        'Marketing': 0.7,
        'Sales': 0.7,
        'Finance': 0.6,
        'Customer Support': 0.5,
        'HR': 0.5
    }

    # Enhanced data generation with more contextual factors
    data = {
        'Division': np.random.choice(divisions, size=n_samples),
        'Strategic_Importance': [strategic_importance[div] for div in np.random.choice(divisions, size=n_samples)],
        'Project_Complexity': np.random.uniform(0.1, 1.0, size=n_samples),
        'Team_Size': np.random.randint(5, 75, size=n_samples),
        'Innovation_Score': np.random.uniform(0.1, 1.0, size=n_samples),
        'External_Opportunities': np.random.uniform(0.1, 1.0, size=n_samples),
        'Career_Growth_Potential': np.random.uniform(0.1, 1.0, size=n_samples),
        'Work_Life_Balance': np.random.uniform(0.1, 1.0, size=n_samples),
        'Compensation_Satisfaction': np.random.uniform(0.1, 1.0, size=n_samples)
    }

    df = pd.DataFrame(data)

    # More sophisticated attrition risk calculation
    df['Attrition_Risk'] = (
        0.25 * df['Strategic_Importance'] + 
        0.2 * df['Project_Complexity'] + 
        0.1 * (1 / df['Team_Size']) + 
        0.15 * (1 - df['Innovation_Score']) + 
        0.1 * df['External_Opportunities'] +
        0.1 * (1 - df['Career_Growth_Potential']) +
        0.05 * (1 - df['Work_Life_Balance']) +
        0.05 * (1 - df['Compensation_Satisfaction'])
    )

    # Refined risk categorization
    df['Risk_Category'] = pd.cut(
        df['Attrition_Risk'],
        bins=[0, 0.3, 0.6, 1],
        labels=['Low', 'Medium', 'High']
    )

    return df

def analyze_divisional_risks(df: pd.DataFrame) -> tuple:
    """
    Comprehensive divisional risk analysis with multiple perspectives
    """
    # Divisional risk summary with expanded metrics
    division_risk = df.groupby('Division').agg({
        'Attrition_Risk': ['mean', 'std'],
        'Division': 'count',
        'Strategic_Importance': 'mean',
        'Project_Complexity': 'mean',
        'Innovation_Score': 'mean'
    }).reset_index()
    
    division_risk.columns = [
        'Division', 'Avg_Risk', 'Risk_Volatility', 
        'Total_Teams', 'Avg_Strategic_Importance', 
        'Avg_Project_Complexity', 'Avg_Innovation_Score'
    ]
    division_risk = division_risk.sort_values('Avg_Risk', ascending=False)

    # Risk category distribution per division
    risk_distribution = df.groupby(['Division', 'Risk_Category']).size().unstack(fill_value=0)
    risk_distribution_pct = risk_distribution.div(risk_distribution.sum(axis=1), axis=0) * 100

    return division_risk, risk_distribution_pct

def generate_strategic_insights() -> Dict[str, str]:
    """
    Generate comprehensive strategic insights for each division
    """
    return {
        'Technology': """
        🚨 High Risk Division
        - Implement advanced retention strategies
        - Create specialized career development programs
        - Enhance innovation and autonomy opportunities
        - Competitive compensation and cutting-edge project assignments
        """,
        'Product Development': """
        🔍 Strategic Focus Needed
        - Develop clear career progression paths
        - Encourage cross-functional skill development
        - Create mentorship and innovation incubation programs
        - Balance project complexity and team resources
        """,
        'Engineering': """
        ⚙️ Proactive Risk Management
        - Invest in continuous learning and certification programs
        - Implement flexible work arrangements
        - Create technical leadership tracks
        - Regular skill assessment and growth opportunities
        """,
        'Operations': """
        🔄 Operational Stability Required
        - Improve team dynamics and communication
        - Provide cross-training and skill diversification
        - Enhance process efficiency and autonomy
        - Create performance recognition mechanisms
        """,
        'Marketing': """
        🎨 Creative Retention Strategies
        - Foster creative freedom and idea sharing
        - Implement project diversity initiatives
        - Provide marketing technology training
        - Create collaborative and inspiring work environments
        """,
        'Sales': """
        💼 Performance-Driven Approach
        - Design competitive incentive structures
        - Provide sales skill enhancement workshops
        - Create clear career progression paths
        - Implement performance and achievement recognition
        """,
        'Finance': """
        📊 Strategic Talent Preservation
        - Develop financial career development programs
        - Offer competitive compensation packages
        - Create analytical and strategic role opportunities
        - Implement financial skill enhancement initiatives
        """,
        'Customer Support': """
        🤝 Support Team Engagement
        - Design empowerment and skill development programs
        - Create clear career progression in support roles
        - Implement technology and communication training
        - Develop recognition and growth opportunities
        """,
        'HR': """
        👥 Internal Talent Management
        - Strengthen internal mobility programs
        - Create HR professional development framework
        - Implement strategic talent acquisition skills
        - Design comprehensive HR career pathways
        """
    }

def main():
    # Load and analyze data
    df = load_division_data()
    division_risk, risk_distribution_pct = analyze_divisional_risks(df)
    strategic_insights = generate_strategic_insights()

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
        default=['Low', 'Medium', 'High']
    )

    # Filter DataFrame
    filtered_df = df[
        (df['Division'].isin(selected_divisions)) & 
        (df['Risk_Category'].isin(risk_filter))
    ]

    # Main title
    st.title('🛫 Divisional Attrition Risk Analysis')
    st.markdown("**Comprehensive Talent Risk Management Framework**")

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔥 Risk Overview", 
        "📊 Divisional Comparisons", 
        "🧩 Risk Factors", 
        "🚀 Strategic Insights"
    ])

    with tab1:
        st.subheader('📈 Divisional Risk Distribution')
        
        # Explanation
        st.info("""
        **Tab Focus**: Visualize overall risk distribution across divisions.
        - Heatmap shows risk category percentages
        - Pie chart illustrates overall organizational risk
        """)
        
        # Heatmap of Risk Levels
        fig = px.imshow(
            risk_distribution_pct, 
            labels=dict(x="Risk Category", y="Division", color="Percentage"),
            color_continuous_scale='Blues',
            title='Risk Category Distribution Across Divisions'
        )
        st.plotly_chart(fig, use_container_width=True)

        # Pie Chart of Overall Risk Categories
        overall_risk_dist = filtered_df['Risk_Category'].value_counts(normalize=True) * 100
        fig = px.pie(
            values=overall_risk_dist.values, 
            names=overall_risk_dist.index,
            title='Overall Risk Category Distribution',
            color_discrete_sequence=[
                AIRBUS_COLORS['primary_blue'], 
                AIRBUS_COLORS['secondary_blue'], 
                AIRBUS_COLORS['light_blue']
            ]
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader('🔬 Comparative Divisional Risk Analysis')
        
        # Explanation
        st.info("""
        **Tab Focus**: Deep dive into divisional risk variations.
        - Bar chart shows average attrition risks
        - Scatter plot correlates team size with risk volatility
        """)
        
        # Filter division risk data
        filtered_division_risk = division_risk[division_risk['Division'].isin(selected_divisions)]
        
        # Bar chart of average risks
        fig = px.bar(
            filtered_division_risk, 
            x='Division', 
            y='Avg_Risk',
            color='Division',
            title='Average Attrition Risk by Division',
            color_discrete_sequence=[AIRBUS_COLORS['primary_blue'], 
                                     AIRBUS_COLORS['secondary_blue'], 
                                     AIRBUS_COLORS['accent_blue']]
        )
        st.plotly_chart(fig, use_container_width=True)

        # Scatter plot of risk vs team size
        fig = px.scatter(
            filtered_division_risk, 
            x='Total_Teams', 
            y='Avg_Risk', 
            size='Risk_Volatility',
            color='Division',
            hover_data=['Division'],
            title='Risk Correlation with Team Size',
            labels={'Total_Teams': 'Number of Teams', 'Avg_Risk': 'Average Risk'},
            color_discrete_sequence=[AIRBUS_COLORS['primary_blue'], 
                                     AIRBUS_COLORS['secondary_blue'], 
                                     AIRBUS_COLORS['accent_blue']]
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader('🔍 Key Risk Factors Analysis')
        
        # Explanation
        st.info("""
        **Tab Focus**: Understand complex risk factor interactions.
        - Correlation matrix reveals factor relationships
        - Box plots show factor distributions across divisions
        """)
        
        # Correlation of risk factors
        correlation_factors = [
            'Strategic_Importance', 'Project_Complexity', 
            'Team_Size', 'Innovation_Score', 
            'External_Opportunities', 'Career_Growth_Potential'
        ]
        correlation_matrix = filtered_df[correlation_factors + ['Attrition_Risk']].corr()
        
        fig = px.imshow(
            correlation_matrix, 
            labels=dict(x="Factors", y="Factors", color="Correlation"),
            title='Correlation of Risk Factors',
            color_continuous_scale='RdBu_r'
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader('🎯 Strategic Talent Management Insights')
        
        # Explanation
        st.info("""
        **Tab Focus**: Actionable recommendations for HR and Leadership.
        - Detailed divisional risk metrics
        - Targeted strategic insights for each division
        """)
        
        # Top risk divisions with recommendations
        st.dataframe(
            division_risk[division_risk['Division'].isin(selected_divisions)].style
            .format({
                'Avg_Risk': '{:.2f}',
                'Risk_Volatility': '{:.2f}',
                'Avg_Strategic_Importance': '{:.2f}',
                'Avg_Project_Complexity': '{:.2f}',
                'Avg_Innovation_Score': '{:.2f}'
            })
            .background_gradient(cmap='Blues')
        )

        # Strategic insights section
        st.subheader('🔮 Divisional Risk Mitigation Strategies')
        for division in selected_divisions:
            if division in strategic_insights:
                st.warning(f"**{division} Division Insights:**")
                st.markdown(strategic_insights[division])

if __name__ == "__main__":
    main()
