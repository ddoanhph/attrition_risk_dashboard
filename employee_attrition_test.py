import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Set page configuration
st.set_page_config(
    page_title="Employee Attrition Risk Dashboard",
    page_icon="🧑‍💼",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Function to load and prepare data - in a real scenario, load actual data
@st.cache_data
def load_data():
    # For demo purposes, we'll create a synthetic dataset that mimics your structure
    # In a real scenario, you would load your actual data

    # Create synthetic data for demonstration
    np.random.seed(42)
    n_samples = 800

    # Using the actual Job Family Groups from your dataset
    job_family_groups = [
        'Manuf., Assembly, Integr. & Test <FU-MA>',
        'Management <FU-FA>',
        'Customer Services and Support <FU-CS>',
        'Engineering <FU-EN>',
        'Quality <FU-QU>',
        'Supply Management <FU-SM>',
        'P&PM and Configuration Mgmt <FU-PP>',
        'Marketing, Sales & Commercial Contracts <FU-MC>',
        'Corporate Governance <FU-CG>',
        'Finance <FU-FI>',
        'Digital <FU-IM>',
        'Human Resources <FU-HR>',
        'Business Improvement <FU-BI>',
        'Health & Safety <FU-HS>'
    ]

    # Distribution weights based on the counts you provided
    job_family_group_weights = [
        0.28, 0.13, 0.13, 0.10, 0.09, 0.09, 0.05,
        0.03, 0.03, 0.02, 0.02, 0.02, 0.01, 0.00
    ]

    bands = ['BZ', 'BII', 'BIII', 'BIV', 'BV']
    regions = ['USA']
    age_groups = ['20-30', '31-40', '41-50', '51+']

    data = {
        'Corp_ID': [f'E{i:04d}' for i in range(1, n_samples + 1)],
        'Attrition': np.random.choice([0, 1], size=n_samples, p=[0.89, 0.11]),
        'Job_Family_Group': np.random.choice(job_family_groups, size=n_samples, p=job_family_group_weights),
        'Band': np.random.choice(bands, size=n_samples, p=[0.3, 0.3, 0.2, 0.15, 0.05]),
        'Location_Region': np.random.choice(regions, size=n_samples),
        'Age_Group': np.random.choice(age_groups, size=n_samples),
        'Career_Velocity': np.random.uniform(0.1, 2.0, size=n_samples),
        'Role_Stability': np.random.uniform(0, 1, size=n_samples),
        'Career_Growth_Score': np.random.uniform(1, 10, size=n_samples),
        'Employment_Complexity': np.random.randint(0, 5, size=n_samples),
        'Division_Transfer_Rate': np.random.uniform(0, 0.5, size=n_samples)
    }

    df = pd.DataFrame(data)

    # Ensure higher attrition rates for certain conditions to make the data more realistic
    # For example, lower Role_Stability often correlates with higher attrition
    mask = (df['Role_Stability'] < 0.3) & (df['Attrition'] == 0)
    flip_indices = np.random.choice(df[mask].index, size=int(len(df[mask]) * 0.3), replace=False)
    df.loc[flip_indices, 'Attrition'] = 1

    # Make certain job family groups more prone to attrition
    high_risk_groups = ['Digital <FU-IM>', 'Engineering <FU-EN>']
    mask = (df['Job_Family_Group'].isin(high_risk_groups)) & (df['Attrition'] == 0)
    flip_indices = np.random.choice(df[mask].index, size=int(len(df[mask]) * 0.2), replace=False)
    df.loc[flip_indices, 'Attrition'] = 1

    return df


# Function to train model and get predictions
@st.cache_resource
def get_predictions(df):
    # In a real scenario, you would use your trained model
    # For demo purposes, we'll train a simple model on our synthetic data

    # Features to use for prediction
    feature_cols = [
        'Career_Velocity', 'Role_Stability', 'Career_Growth_Score',
        'Employment_Complexity', 'Division_Transfer_Rate'
    ]

    # Prepare data for modeling
    X = df[feature_cols]
    y = df['Attrition']

    # Train a model (in practice, you'd load your pre-trained model)
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X, y)

    # Get predictions and probabilities
    df['Predicted_Attrition'] = model.predict(X)
    df['Attrition_Risk'] = model.predict_proba(X)[:, 1]

    # Add risk category
    df['Risk_Category'] = pd.cut(
        df['Attrition_Risk'],
        bins=[0, 0.3, 0.6, 1],
        labels=['Low', 'Medium', 'High']
    )

    # Get feature importance
    feature_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)

    # For demonstration purposes, we'll simulate SHAP values
    # In practice, you would use actual SHAP values
    def get_key_factors(row):
        factors = []
        if row['Role_Stability'] < 0.3:
            factors.append('Low Role Stability')
        if row['Career_Velocity'] > 1.5:
            factors.append('High Career Velocity')
        if row['Career_Growth_Score'] < 4:
            factors.append('Low Career Growth Score')
        if row['Employment_Complexity'] > 3:
            factors.append('High Employment Complexity')
        if row['Division_Transfer_Rate'] < 0.1:
            factors.append('Low Division Transfer Rate')

        return ', '.join(factors[:2]) if factors else 'No significant factors'

    df['Key_Factors'] = df.apply(get_key_factors, axis=1)

    return df, model, feature_importance


# Load data and get predictions
df = load_data()
df_with_predictions, model, feature_importance = get_predictions(df)

# UI Components

# Sidebar filters
st.sidebar.header('Filters')

# Job Family Group filter
selected_job_family_groups = st.sidebar.multiselect(
    'Select Job Family Groups',
    options=sorted(df['Job_Family_Group'].unique()),
    default=sorted(df['Job_Family_Group'].unique())
)

# Band filter
selected_bands = st.sidebar.multiselect(
    'Select Bands',
    options=sorted(df['Band'].unique()),
    default=sorted(df['Band'].unique())
)

# Region filter
selected_regions = st.sidebar.multiselect(
    'Select Regions',
    options=sorted(df['Location_Region'].unique()),
    default=sorted(df['Location_Region'].unique())
)

# Risk category filter
selected_risk = st.sidebar.multiselect(
    'Risk Category',
    options=['High', 'Medium', 'Low'],
    default=['High', 'Medium', 'Low']
)

# Apply filters
filtered_df = df_with_predictions[
    (df_with_predictions['Job_Family_Group'].isin(selected_job_family_groups)) &
    (df_with_predictions['Band'].isin(selected_bands)) &
    (df_with_predictions['Location_Region'].isin(selected_regions)) &
    (df_with_predictions['Risk_Category'].isin(selected_risk))
    ]

# Main dashboard layout
st.title('Employee Attrition Risk Dashboard')

# Top metrics row
col1, col2, col3 = st.columns(3)

with col1:
    total_employees = len(filtered_df)
    predicted_attrition = filtered_df['Predicted_Attrition'].sum()
    attrition_rate = (predicted_attrition / total_employees * 100) if total_employees > 0 else 0

    st.metric(
        label="Predicted Attrition",
        value=f"{int(predicted_attrition)}",
        delta=f"{attrition_rate:.1f}% of {total_employees} employees"
    )

with col2:
    risk_counts = filtered_df['Risk_Category'].value_counts().to_dict()
    high_risk = risk_counts.get('High', 0)
    medium_risk = risk_counts.get('Medium', 0)
    low_risk = risk_counts.get('Low', 0)

    st.metric(
        label="High Risk Employees",
        value=f"{high_risk}",
        delta=f"{high_risk / total_employees * 100:.1f}% of total" if total_employees > 0 else "0%"
    )

with col3:
    # Calculate average attrition risk
    avg_risk = filtered_df['Attrition_Risk'].mean() * 100 if not filtered_df.empty else 0
    st.metric(
        label="Average Attrition Risk",
        value=f"{avg_risk:.1f}%",
        delta=None
    )

# Attrition by category section
st.subheader('Attrition by Category')

# Category selector
category_options = ['Job_Family_Group', 'Band', 'Location_Region', 'Age_Group']
category_names = ['Job Family Group', 'Band', 'Region', 'Age Group']
selected_category_index = st.radio(
    "Select Category",
    options=range(len(category_options)),
    format_func=lambda i: category_names[i],
    horizontal=True
)
selected_category = category_options[selected_category_index]

# Calculate attrition metrics by selected category
category_attrition = filtered_df.groupby(selected_category).agg(
    Total=('Corp_ID', 'count'),
    Attrition=('Predicted_Attrition', 'sum')
)
category_attrition['Rate'] = (category_attrition['Attrition'] / category_attrition['Total'] * 100).round(1)
category_attrition = category_attrition.reset_index().sort_values('Rate', ascending=False)

# Create visualization
fig = make_subplots(specs=[[{"secondary_y": True}]])

fig.add_trace(
    go.Bar(
        x=category_attrition[selected_category],
        y=category_attrition['Attrition'],
        name="Predicted Attrition",
        marker_color='rgba(55, 83, 109, 0.7)'
    )
)

fig.add_trace(
    go.Scatter(
        x=category_attrition[selected_category],
        y=category_attrition['Rate'],
        name="Attrition Rate (%)",
        mode='lines+markers',
        marker_color='rgba(255, 79, 38, 0.7)',
        line=dict(width=2)
    ),
    secondary_y=True
)

fig.update_layout(
    title_text=f'Attrition by {category_names[selected_category_index]}',
    xaxis_title=category_names[selected_category_index],
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
    height=400
)

fig.update_yaxes(title_text="Number of Employees", secondary_y=False)
fig.update_yaxes(title_text="Attrition Rate (%)", secondary_y=True)

st.plotly_chart(fig, use_container_width=True)

# Feature importance and high risk analysis sections
col1, col2 = st.columns(2)

with col1:
    st.subheader('Top Attrition Risk Factors')

    # Create horizontal bar chart for feature importance
    fig = px.bar(
        feature_importance.head(5),
        x='Importance',
        y='Feature',
        orientation='h',
        labels={'Importance': 'Relative Importance', 'Feature': ''},
        color='Importance',
        color_continuous_scale=px.colors.sequential.Blues
    )

    fig.update_layout(
        height=300,
        yaxis=dict(autorange="reversed"),
        coloraxis_showscale=False
    )

    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader('Risk Distribution')

    fig = px.pie(
        filtered_df,
        names='Risk_Category',
        values='Corp_ID',
        color='Risk_Category',
        color_discrete_map={'High': 'red', 'Medium': 'orange', 'Low': 'green'},
        hole=0.4
    )

    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

# High risk job family groups section - replacing individual employees section
st.subheader('Top High-Risk Job Family Groups')

# Calculate risk metrics by job family group
job_family_risk = filtered_df.groupby('Job_Family_Group').agg(
    Total_Employees=('Corp_ID', 'count'),
    High_Risk_Count=(filtered_df['Risk_Category'] == 'High', 'sum'),
    Average_Risk=('Attrition_Risk', 'mean'),
    Predicted_Attrition=('Predicted_Attrition', 'sum')
)

job_family_risk['High_Risk_Percentage'] = (
            job_family_risk['High_Risk_Count'] / job_family_risk['Total_Employees'] * 100).round(1)
job_family_risk['Attrition_Rate'] = (
            job_family_risk['Predicted_Attrition'] / job_family_risk['Total_Employees'] * 100).round(1)
job_family_risk = job_family_risk.reset_index().sort_values('High_Risk_Percentage', ascending=False)

if not job_family_risk.empty:
    # Get most common risk factors for each job family group
    def get_top_factors_for_group(group):
        group_df = filtered_df[filtered_df['Job_Family_Group'] == group]
        factors = []
        for factor in group_df['Key_Factors'].str.split(', '):
            if isinstance(factor, list):
                factors.extend(factor)

        from collections import Counter
        if factors:
            common = Counter(factors).most_common(2)
            return ', '.join([f"{item[0]}" for item in common])
        return "No significant factors"


    job_family_risk['Key_Risk_Factors'] = job_family_risk['Job_Family_Group'].apply(get_top_factors_for_group)

    # Prepare display dataframe
    high_risk_display = job_family_risk.head(10)
    high_risk_display = high_risk_display.rename(columns={
        'Job_Family_Group': 'Job Family Group',
        'Total_Employees': 'Total Employees',
        'High_Risk_Count': 'High Risk Count',
        'High_Risk_Percentage': 'High Risk %',
        'Average_Risk': 'Avg Risk Score',
        'Attrition_Rate': 'Attrition Rate %',
        'Key_Risk_Factors': 'Key Risk Factors'
    })

    # Format risk score as percentage
    high_risk_display['Avg Risk Score'] = high_risk_display['Avg Risk Score'].apply(lambda x: f"{x:.1%}")

    # Show only relevant columns
    display_cols = ['Job Family Group', 'Total Employees', 'High Risk Count', 'High Risk %',
                    'Avg Risk Score', 'Attrition Rate %', 'Key Risk Factors']

    st.dataframe(high_risk_display[display_cols], use_container_width=True)

    # Create a heatmap of job family groups by risk
    st.subheader('Job Family Group Risk Heatmap')

    # Pivot the data to create a heatmap
    if len(job_family_risk) > 1:  # Only show heatmap if we have more than one job family
        # Get top 8 job families by risk percentage for better visualization
        top_job_families = job_family_risk.head(8)['Job_Family_Group'].tolist()

        # Filter to these top families
        heatmap_data = filtered_df[filtered_df['Job_Family_Group'].isin(top_job_families)]

        # Create a pivot table showing risk distribution
        pivot = pd.crosstab(
            index=heatmap_data['Job_Family_Group'],
            columns=heatmap_data['Risk_Category'],
            normalize='index'
        ) * 100

        # Ensure all risk categories are present
        for cat in ['Low', 'Medium', 'High']:
            if cat not in pivot.columns:
                pivot[cat] = 0

        # Sort pivot table by high risk percentage
        pivot = pivot.sort_values('High', ascending=False)

        # Create heatmap
        fig = px.imshow(
            pivot,
            text_auto='.1f',
            aspect="auto",
            color_continuous_scale='RdYlGn_r',
            labels=dict(x='Risk Category', y='Job Family Group', color='Percentage'),
            x=['High', 'Medium', 'Low']  # Force this order
        )

        fig.update_layout(
            height=400,
            coloraxis_colorbar=dict(title='Percentage')
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough job family groups to create a heatmap with the current filters.")
else:
    st.info("No job family groups match the current filters.")

# Add action recommendations section
st.subheader('Recommended Actions')

col1, col2, col3 = st.columns(3)

# Get top job family with highest attrition
top_job_family = job_family_risk.iloc[0]['Job_Family_Group'] if not job_family_risk.empty else "N/A"
top_job_family_risk = job_family_risk.iloc[0]['High_Risk_Percentage'] if not job_family_risk.empty else 0
top_risk_factors = job_family_risk.iloc[0]['Key_Risk_Factors'] if not job_family_risk.empty else "N/A"

# Get most common band in high risk employees
if not filtered_df.empty:
    high_risk_df = filtered_df[filtered_df['Risk_Category'] == 'High']
    most_common_band = high_risk_df['Band'].mode()[0] if not high_risk_df.empty else "N/A"
else:
    most_common_band = "N/A"

with col1:
    st.info(
        f"**Focus on {top_job_family}**\n\n"
        f"Highest risk at {top_job_family_risk:.1f}%. "
        f"Consider conducting targeted retention interviews and implementing career development programs."
    )

with col2:
    st.info(
        f"**Attention to {most_common_band} Band**\n\n"
        f"This band shows elevated risk across job families. "
        f"Review compensation competitiveness and growth opportunities."
    )

with col3:
    st.info(
        f"**Address: {top_risk_factors}**\n\n"
        f"These are the most common risk factors in the highest risk job family. "
        f"Develop targeted initiatives to address these specific challenges."
    )

# Footer
st.markdown("---")
st.markdown(
    "**Note:** This dashboard uses predictive modeling to identify attrition risks by job family group. "
    "All predictions should be verified with qualitative assessment before taking action."
)
