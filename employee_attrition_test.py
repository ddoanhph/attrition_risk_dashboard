import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import GradientBoostingClassifier
from collections import Counter

# Set page configuration
st.set_page_config(page_title="Employee Attrition Risk Dashboard", layout="wide")

# Function to load and prepare data (unchanged from original)
def load_data():
    np.random.seed(42)
    n_samples = 800

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

    job_family_group_weights = [
        0.28, 0.13, 0.13, 0.10, 0.09, 0.09, 0.05,
        0.03, 0.03, 0.02, 0.02, 0.02, 0.01, 0.00
    ]

    divisions = {
        'Manufacturing': ['Manuf., Assembly, Integr. & Test <FU-MA>', 'Quality <FU-QU>'],
        'Management': ['Management <FU-FA>', 'Corporate Governance <FU-CG>'],
        'Customer Operations': ['Customer Services and Support <FU-CS>',
                                'Marketing, Sales & Commercial Contracts <FU-MC>'],
        'Engineering': ['Engineering <FU-EN>', 'Digital <FU-IM>'],
        'Supply Chain': ['Supply Management <FU-SM>', 'P&PM and Configuration Mgmt <FU-PP>'],
        'Support Functions': ['Finance <FU-FI>', 'Human Resources <FU-HR>', 'Business Improvement <FU-BI>',
                              'Health & Safety <FU-HS>']
    }

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

    df['Division'] = df['Job_Family_Group'].apply(
        lambda x: next((div for div, groups in divisions.items() if x in groups), 'Other')
    )

    mask = (df['Role_Stability'] < 0.3) & (df['Attrition'] == 0)
    flip_indices = np.random.choice(df[mask].index, size=int(len(df[mask]) * 0.3), replace=False)
    df.loc[flip_indices, 'Attrition'] = 1

    high_risk_groups = ['Digital <FU-IM>', 'Engineering <FU-EN>']
    mask = (df['Job_Family_Group'].isin(high_risk_groups)) & (df['Attrition'] == 0)
    flip_indices = np.random.choice(df[mask].index, size=int(len(df[mask]) * 0.2), replace=False)
    df.loc[flip_indices, 'Attrition'] = 1

    high_risk_divisions = ['Engineering', 'Supply Chain']
    mask = (df['Division'].isin(high_risk_divisions)) & (df['Attrition'] == 0)
    flip_indices = np.random.choice(df[mask].index, size=int(len(df[mask]) * 0.15), replace=False)
    df.loc[flip_indices, 'Attrition'] = 1

    return df

# Function to train model and get predictions (unchanged from original)
def get_predictions(df):
    feature_cols = [
        'Career_Velocity', 'Role_Stability', 'Career_Growth_Score',
        'Employment_Complexity', 'Division_Transfer_Rate'
    ]

    X = df[feature_cols]
    y = df['Attrition']

    model = GradientBoostingClassifier(random_state=42)
    model.fit(X, y)

    df['Predicted_Attrition'] = model.predict(X)
    df['Attrition_Risk'] = model.predict_proba(X)[:, 1]

    df['Risk_Category'] = pd.cut(
        df['Attrition_Risk'],
        bins=[0, 0.3, 0.6, 1],
        labels=['Low', 'Medium', 'High']
    )

    feature_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)

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

# Sidebar for filters
st.sidebar.header("Filters")

division_options = sorted(df['Division'].unique())
selected_divisions = st.sidebar.multiselect("Select Divisions", division_options, default=division_options)

band_options = sorted(df['Band'].unique())
selected_bands = st.sidebar.multiselect("Select Bands", band_options, default=band_options)

region_options = sorted(df['Location_Region'].unique())
selected_regions = st.sidebar.multiselect("Select Regions", region_options, default=region_options)

risk_options = ['High', 'Medium', 'Low']
selected_risk = st.sidebar.multiselect("Risk Category", risk_options, default=risk_options)

# Filter dataframe based on selections
filtered_df = df_with_predictions[
    (df_with_predictions['Division'].isin(selected_divisions)) &
    (df_with_predictions['Band'].isin(selected_bands)) &
    (df_with_predictions['Location_Region'].isin(selected_regions)) &
    (df_with_predictions['Risk_Category'].isin(selected_risk))
]

# Main dashboard
st.title("Employee Attrition Risk Dashboard")

# Metrics
col1, col2, col3 = st.columns(3)
total_employees = len(filtered_df)
predicted_attrition = filtered_df['Predicted_Attrition'].sum()
attrition_rate = (predicted_attrition / total_employees * 100) if total_employees > 0 else 0
risk_counts = filtered_df['Risk_Category'].value_counts().to_dict()
high_risk = risk_counts.get('High', 0)
high_risk_pct = (high_risk / total_employees * 100) if total_employees > 0 else 0
avg_risk = filtered_df['Attrition_Risk'].mean() * 100 if not filtered_df.empty else 0

with col1:
    st.metric("Predicted Attrition", f"{int(predicted_attrition)}", f"{attrition_rate:.1f}% of {total_employees}")
with col2:
    st.metric("High Risk Employees", f"{high_risk}", f"{high_risk_pct:.1f}%")
with col3:
    st.metric("Average Attrition Risk", f"{avg_risk:.1f}%")

# Attrition by category
st.subheader("Attrition by Category")
category_options = ['Division', 'Job_Family_Group', 'Band', 'Location_Region', 'Age_Group']
selected_category = st.radio("Select Category", category_options, index=0, horizontal=True)

category_attrition = filtered_df.groupby(selected_category).agg(
    Total=('Corp_ID', 'count'),
    Attrition=('Predicted_Attrition', 'sum')
)
category_attrition['Rate'] = (category_attrition['Attrition'] / category_attrition['Total'] * 100).round(1)
category_attrition = category_attrition.reset_index().sort_values('Rate', ascending=False)

fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Bar(x=category_attrition[selected_category], y=category_attrition['Attrition'], name="Predicted Attrition", marker_color='rgba(55, 83, 109, 0.7)'))
fig.add_trace(go.Scatter(x=category_attrition[selected_category], y=category_attrition['Rate'], name="Attrition Rate (%)", mode='lines+markers', marker_color='rgba(255, 79, 38, 0.7)', line=dict(width=2)), secondary_y=True)
fig.update_layout(title_text=f'Attrition by {selected_category.replace("_", " ")}', xaxis_title=selected_category.replace("_", " "), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), height=400)
fig.update_yaxes(title_text="Number of Employees", secondary_y=False)
fig.update_yaxes(title_text="Attrition Rate (%)", secondary_y=True)
st.plotly_chart(fig, use_container_width=True)

# Feature importance and risk distribution
col1, col2 = st.columns(2)

with col1:
    st.subheader("Top Attrition Risk Factors")
    fig = px.bar(feature_importance.head(5), x='Importance', y='Feature', orientation='h', labels={'Importance': 'Relative Importance', 'Feature': ''}, color='Importance', color_continuous_scale=px.colors.sequential.Blues)
    fig.update_layout(low=300, yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Risk Distribution")
    fig = px.pie(filtered_df, names='Risk_Category', values='Corp_ID', color='Risk_Category', color_discrete_map={'High': 'red', 'Medium': 'orange', 'Low': 'green'}, hole=0.4)
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

# Division risk analysis
st.subheader("Division Risk Analysis")
division_risk = filtered_df.groupby('Division').agg(
    Total_Employees=('Corp_ID', 'count'),
    High_Risk_Count=(filtered_df['Risk_Category'] == 'High', 'sum'),
    Average_Risk=('Attrition_Risk', 'mean'),
    Predicted_Attrition=('Predicted_Attrition', 'sum')
)
division_risk['High_Risk_Percentage'] = (division_risk['High_Risk_Count'] / division_risk['Total_Employees'] * 100).round(1)
division_risk['Attrition_Rate'] = (division_risk['Predicted_Attrition'] / division_risk['Total_Employees'] * 100).round(1)
division_risk = division_risk.reset_index().sort_values('High_Risk_Percentage', ascending=False)

def get_top_factors_for_division(division):
    division_df = filtered_df[filtered_df['Division'] == division]
    factors = []
    for factor in division_df['Key_Factors'].str.split(', '):
        if isinstance(factor, list):
            factors.extend(factor)
    if factors:
        common = Counter(factors).most_common(2)
        return ', '.join([f"{item[0]}" for item in common])
    return "No significant factors"

division_risk['Key_Risk_Factors'] = division_risk['Division'].apply(get_top_factors_for_division)
display_df = division_risk.rename(columns={
    'Division': 'Division',
    'Total_Employees': 'Total Employees',
    'High_Risk_Count': 'High Risk Count',
    'High_Risk_Percentage': 'High Risk %',
    'Average_Risk': 'Avg Risk Score',
    'Attrition_Rate': 'Attrition Rate %',
    'Key_Risk_Factors': 'Key Risk Factors'
})
display_df['Avg Risk Score'] = display_df['Avg Risk Score'].apply(lambda x: f"{x:.1%}")
st.dataframe(display_df, use_container_width=True)

# Division risk heatmap
st.subheader("Division Risk Heatmap")
if len(selected_divisions) > 1:
    pivot = pd.crosstab(index=filtered_df['Division'], columns=filtered_df['Risk_Category'], normalize='index') * 100
    for cat in ['Low', 'Medium', 'High']:
        if cat not in pivot.columns:
            pivot[cat] = 0
    pivot = pivot.sort_values('High', ascending=False)
    fig = px.imshow(pivot, text_auto='.1f', aspect="auto", color_continuous_scale='RdYlGn_r', labels=dict(x='Risk Category', y='Division', color='Percentage'), x=['High', 'Medium', 'Low'])
    fig.update_layout(height=400, coloraxis_colorbar=dict(title='Percentage'))
else:
    fig = go.Figure()
    fig.add_annotation(text="Please select multiple divisions to generate a heatmap", showarrow=False, font=dict(size=16))
    fig.update_layout(height=400)
st.plotly_chart(fig, use_container_width=True)

# Recommendations
st.subheader("Recommended Actions")
division_risk = filtered_df.groupby('Division').agg(
    Total_Employees=('Corp_ID', 'count'),
    High_Risk_Count=(filtered_df['Risk_Category'] == 'High', 'sum'),
    Average_Risk=('Attrition_Risk', 'mean'),
    Predicted_Attrition=('Predicted_Attrition', 'sum')
)
division_risk['High_Risk_Percentage'] = (division_risk['High_Risk_Count'] / division_risk['Total_Employees'] * 100).round(1)
division_risk = division_risk.reset_index().sort_values('High_Risk_Percentage', ascending=False)

top_division = division_risk.iloc[0]['Division'] if not division_risk.empty else "N/A"
top_division_risk = division_risk.iloc[0]['High_Risk_Percentage'] if not division_risk.empty else 0
top_division_df = filtered_df[filtered_df['Division'] == top_division]
top_factors = []
for factor in top_division_df['Key_Factors'].str.split(', '):
    if isinstance(factor, list):
        top_factors.extend(factor)
top_factors_text = "No significant factors" if not top_factors else ', '.join([f"{item[0]}" for item in Counter(top_factors).most_common(2)])

high_risk_df = filtered_df[filtered_df['Risk_Category'] == 'High']
most_common_band = high_risk_df['Band'].mode()[0] if not high_risk_df.empty else "N/A"

col1, col2, col3 = st.columns(3)
with col1:
    st.info(f"**Focus on {top_division}**\n\nThis division has {top_division_risk:.1f}% of employees at high risk. Conduct stay interviews and engagement surveys to identify specific concerns. Consider targeted retention strategies for critical roles.")
with col2:
    st.info(f"**Address Band {most_common_band} Concerns**\n\nBand {most_common_band} shows the highest concentration of at-risk employees. Review compensation structures, career progression paths, and workload distribution. Consider implementing mentorship programs or career development workshops.")
with col3:
    st.info(f"**Target Key Risk Factors**\n\nThe most common risk factors are {top_factors_text}. Develop targeted initiatives to address these specific concerns. Consider pulse surveys to track improvement in these areas.")

# Footer
st.markdown("---")
st.markdown("*Note: This dashboard uses predictive modeling to identify attrition risks by division. All predictions should be verified with qualitative assessment before taking action.*", unsafe_allow_html=True)

if __name__ == "__main__":
    # Streamlit runs automatically when the script is executed
    pass
