import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import GradientBoostingClassifier

# Airbus Color Palette
AIRBUS_COLORS = {
    'primary_blue': '#00205B',  # Dark Airbus Blue
    'secondary_blue': '#5BA3D4',  # Lighter Airbus Blue
    'grey': '#6D6E71',  # Airbus Grey
    'light_blue': '#A4D4E9',  # Light Blue
    'accent_blue': '#00A3E1',  # Bright Accent Blue
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
def load_division_data():
    """
    Create synthetic data focusing on divisional-level metrics
    """
    np.random.seed(42)
    n_samples = 800

    divisions = ['Technology', 'Operations', 'Marketing', 'Finance', 'HR', 'Sales']
    strategic_importance = {
        'Technology': 0.9,
        'Operations': 0.8,
        'Marketing': 0.7,
        'Finance': 0.6,
        'HR': 0.5,
        'Sales': 0.7
    }

    data = {
        'Division': np.random.choice(divisions, size=n_samples),
        'Strategic_Importance': [strategic_importance[div] for div in np.random.choice(divisions, size=n_samples)],
        'Project_Complexity': np.random.uniform(0.1, 1.0, size=n_samples),
        'Team_Size': np.random.randint(5, 50, size=n_samples),
        'Innovation_Score': np.random.uniform(0.1, 1.0, size=n_samples),
        'External_Opportunities': np.random.uniform(0.1, 1.0, size=n_samples)
    }

    df = pd.DataFrame(data)

    # Simulate attrition risk based on various factors
    df['Attrition_Risk'] = (
            0.3 * df['Strategic_Importance'] +
            0.2 * df['Project_Complexity'] +
            0.1 * (1 / df['Team_Size']) +
            0.2 * (1 - df['Innovation_Score']) +
            0.2 * df['External_Opportunities']
    )

    # Categorize risk
    df['Risk_Category'] = pd.cut(
        df['Attrition_Risk'],
        bins=[0, 0.3, 0.6, 1],
        labels=['Low', 'Medium', 'High']
    )

    return df


def analyze_divisional_risks(df):
    """
    Perform divisional-level risk analysis
    """
    # Divisional risk summary
    division_risk = df.groupby('Division').agg({
        'Attrition_Risk': ['mean', 'std'],
        'Division': 'count'
    }).reset_index()
    division_risk.columns = ['Division', 'Avg_Risk', 'Risk_Volatility', 'Total_Teams']
    division_risk = division_risk.sort_values('Avg_Risk', ascending=False)

    # Risk category distribution per division
    risk_distribution = df.groupby(['Division', 'Risk_Category']).size().unstack(fill_value=0)
    risk_distribution_pct = risk_distribution.div(risk_distribution.sum(axis=1), axis=0) * 100

    return division_risk, risk_distribution_pct


def main():
    # Load and analyze data
    df = load_division_data()
    division_risk, risk_distribution_pct = analyze_divisional_risks(df)

    # Main title
    st.title('Divisional Attrition Risk Analysis')

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Risk Overview",
        "Divisional Comparisons",
        "Risk Factors",
        "Strategic Insights"
    ])

    with tab1:
        # Risk Overview Section
        st.subheader('Divisional Risk Distribution')

        # Heatmap of Risk Levels
        fig = px.imshow(
            risk_distribution_pct,
            labels=dict(x="Risk Category", y="Division", color="Percentage"),
            color_continuous_scale='Blues',
            title='Risk Category Distribution Across Divisions'
        )
        st.plotly_chart(fig, use_container_width=True)

        # Pie Chart of Overall Risk Categories
        overall_risk_dist = df['Risk_Category'].value_counts(normalize=True) * 100
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
        # Divisional Comparisons
        st.subheader('Comparative Divisional Risk Analysis')

        # Bar chart of average risks
        fig = px.bar(
            division_risk,
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
        st.subheader('Risk Volatility and Team Size')
        fig = px.scatter(
            division_risk,
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
        # Risk Factors Analysis
        st.subheader('Key Risk Factors Across Divisions')

        # Correlation of risk factors
        correlation_factors = ['Strategic_Importance', 'Project_Complexity', 'Team_Size', 'Innovation_Score',
                               'External_Opportunities']
        correlation_matrix = df[correlation_factors + ['Attrition_Risk']].corr()

        fig = px.imshow(
            correlation_matrix,
            labels=dict(x="Factors", y="Factors", color="Correlation"),
            title='Correlation of Risk Factors',
            color_continuous_scale='RdBu_r'
        )
        st.plotly_chart(fig, use_container_width=True)

        # Box plot of risk factors by division
        fig = go.Figure()
        for factor in correlation_factors:
            fig.add_trace(go.Box(
                y=df[df['Division'] == factor],
                name=factor,
                boxpoints='outliers'
            ))
        fig.update_layout(title='Risk Factor Distribution by Division')
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        # Strategic Insights
        st.subheader('Strategic Recommendations')

        # Top risk divisions with recommendations
        st.dataframe(
            division_risk.style.format({
                'Avg_Risk': '{:.2f}',
                'Risk_Volatility': '{:.2f}'
            }).background_gradient(cmap='Blues')
        )

        # Generate strategic insights
        st.subheader('Divisional Risk Mitigation Strategies')
        insights = {
            'Technology': "Focus on innovation retention and career development programs.",
            'Operations': "Improve team dynamics and cross-functional training.",
            'Marketing': "Enhance creative freedom and project diversity.",
            'Finance': "Develop clear career progression paths and competitive compensation.",
            'HR': "Strengthen internal mobility and professional growth opportunities.",
            'Sales': "Implement performance-based incentives and skill enhancement workshops."
        }

        for division, insight in insights.items():
            st.info(f"**{division}**: {insight}")


if __name__ == "__main__":
    main()