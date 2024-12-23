import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
import pandas as pd

# Constants
HEALTH_BASE_RATES = {
    'young': 3000,    # Monthly: 250
    'adult': 5000,    # Monthly: 416
    'middle_age': 7000,  # Monthly: 583
    'senior': 10000   # Monthly: 833
}

LIFE_BASE_RATES = {
    'young': 6000,    # Monthly: 500
    'adult': 8400,    # Monthly: 700
    'middle_age': 12000,  # Monthly: 1000
    'senior': 18000   # Monthly: 1500
}

PROFESSIONS = {
    'Healthcare': ['Doctor', 'Nurse', 'Pharmacist', 'Therapist'],
    'Technology': ['IT Professional', 'Software Engineer', 'Data Scientist', 'System Administrator'],
    'Business': ['Accountant', 'Business Analyst', 'Manager', 'Salesperson'],
    'Education': ['Teacher', 'Professor', 'Education Administrator', 'Counselor'],
    'High Risk': ['Construction Worker', 'Firefighter', 'Police Officer', 'Manual Labor'],
    'Creative': ['Artist', 'Designer', 'Writer', 'Actor'],
    'Other': ['Other']
}

COVERAGE_AMOUNTS = {
    '50L': 5000000,
    '1Cr': 10000000,
    '1.5Cr': 15000000,
    '2Cr': 20000000,
    '2.5Cr': 25000000,
    '3Cr': 30000000,
    '3.5Cr': 35000000,
    '4Cr': 40000000,
    '4.5Cr': 45000000,
    '5Cr': 50000000
}

def get_age_category(age):
    if age < 25:
        return 'young'
    elif age < 35:
        return 'adult'
    elif age < 45:
        return 'middle_age'
    return 'senior'

def calculate_base_premium(age, gender, profession, insurance_type, coverage_amount=0):
    age_category = get_age_category(age)
    
    if insurance_type == 'Health Insurance':
        base_rate = HEALTH_BASE_RATES[age_category]
    else:  # Life Insurance
        base_rate = LIFE_BASE_RATES[age_category] * (coverage_amount / 10000000)

    # Gender adjustment
    if gender.lower() == 'female':
        base_rate *= 0.95

    # Profession risk adjustment
    profession_lower = profession.lower()
    if profession in PROFESSIONS['High Risk']:
        base_rate *= 1.3
    elif profession in PROFESSIONS['Technology'] or profession in PROFESSIONS['Education']:
        base_rate *= 1.1
    else:
        base_rate *= 1.2

    return base_rate / 12  # Convert to monthly premium

def calculate_discount(fitness_score):
    if fitness_score >= 90:
        return 30
    elif fitness_score >= 80:
        return 25
    elif fitness_score >= 70:
        return 20
    elif fitness_score >= 60:
        return 15
    elif fitness_score >= 50:
        return 10
    elif fitness_score >= 40:
        return 5
    return 0

def generate_personalized_plan(age, gender, profession, fitness_score, insurance_type, coverage_amount=0):
    base_premium = calculate_base_premium(age, gender, profession, insurance_type, coverage_amount)
    discount_rate = calculate_discount(fitness_score)
    discount_amount = base_premium * (discount_rate / 100)
    discounted_premium = base_premium - discount_amount
    max_discount_amount = base_premium * 0.3  # 30% maximum discount
    max_discounted_premium = base_premium - max_discount_amount
    
    return base_premium, discount_rate, discount_amount, discounted_premium, max_discounted_premium

def plot_premium_trend(fitness_scores, base_premium):
    scores = list(fitness_scores)
    premiums = [base_premium * (1 - calculate_discount(score)/100) for score in scores]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=scores, 
        y=premiums, 
        mode='lines+markers',
        name='Premium Amount',
        hovertemplate='Fitness Score: %{x}<br>Premium: ₹%{y:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Premium Amount vs Fitness Score',
        xaxis_title='Fitness Score',
        yaxis_title='Monthly Premium (INR)',
        hovermode='x',
        showlegend=False,
        yaxis_tickformat='₹,.2f'
    )
    
    fig.update_traces(line=dict(width=3), marker=dict(size=8))
    return fig

def make_your_own_plan_page():
    st.title("Personalized Insurance Plan Generator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        name = st.text_input("Enter your name:")
        age = st.number_input("Age", min_value=18, max_value=100, step=1)
        gender = st.selectbox("Gender", options=["Male", "Female", "Other"])
        profession_category = st.selectbox("Profession Category", options=list(PROFESSIONS.keys()))
        profession = st.selectbox("Specific Profession", options=PROFESSIONS[profession_category])
        
    with col2:
        insurance_type = st.selectbox("Insurance Type", options=["Health Insurance", "Life Insurance"])
        
        if insurance_type == "Life Insurance":
            coverage = st.selectbox("Coverage Amount", options=list(COVERAGE_AMOUNTS.keys()))
            coverage_amount = COVERAGE_AMOUNTS[coverage]
            
            st.info(f"""
            💡 Coverage Calculator:
            - Monthly Income Protection: ~INR {coverage_amount/120:,.2f}
            - Years of Coverage: ~{coverage_amount/(age*50000):,.1f} years
            """)
        else:
            coverage_amount = 0

    fitness_score = st.slider("Enter your fitness score:", 0, 100, 50,
                            help="Higher fitness score leads to better discounts!")

    if st.button("Generate Plan", type="primary"):
        results = generate_personalized_plan(
            age, gender, profession, fitness_score, insurance_type, coverage_amount)
        base_premium, discount_rate, discount_amount, discounted_premium, max_discounted_premium = results
        
        st.success("Plan Generated Successfully!")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Your Monthly Premium", f"₹{discounted_premium:,.2f}", f"-{discount_rate}%")
        with col2:
            st.metric("Base Premium Before Discount", f"₹{base_premium:,.2f}")
        with col3:
            actual_savings = base_premium - discounted_premium
            st.metric("You Saved", f"₹{actual_savings:,.2f}")
        
        st.subheader("Premium Trend Analysis")
        fig = plot_premium_trend(range(0, 101, 10), base_premium)
        st.plotly_chart(fig, use_container_width=True)
        
        # Plan details download
        plan_details = {
            'Insurance Type': insurance_type,
            'Age': age,
            'Gender': gender,
            'Profession': profession,
            'Coverage Amount': f"₹{coverage_amount:,}" if insurance_type == "Life Insurance" else "N/A",
            'Base Premium': f"₹{base_premium:,.2f}",
            'Current Discount': f"{discount_rate}%",
            'Final Premium': f"₹{discounted_premium:,.2f}",
            'Fitness Score': f"{fitness_score:.2f}"
        }
        
        df = pd.DataFrame([plan_details])
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download Plan Details",
            data=csv,
            file_name=f"insurance_plan_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    st.set_page_config(page_title="Insurance Plan Generator", page_icon="🏥", layout="wide")
    make_your_own_plan_page()
