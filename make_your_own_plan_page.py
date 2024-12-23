import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pandas as pd
from fitness_score_page import display_user_metrics, get_fitness_score_and_discount, load_data_and_model

# Move constants to the top for better organization
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

RISK_FACTORS = {
    'high_risk': ['manual labor', 'construction worker', 'police officer', 'firefighter'],
    'medium_risk': ['teacher', 'it professional', 'office worker', 'salesperson'],
    'risk_multipliers': {'high': 1.3, 'medium': 1.1, 'low': 1.2}
}

# Add more profession categories with detailed descriptions
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
    base_rate = 0

    health_base_rates = {
        'young': 3000,  # Monthly: 250
        'adult': 5000,  # Monthly: 416
        'middle_age': 7000,  # Monthly: 583
        'senior': 10000  # Monthly: 833
    }

    life_base_rates = {
        'young': 6000,  # Monthly: 500
        'adult': 8400,  # Monthly: 700
        'middle_age': 12000,  # Monthly: 1000
        'senior': 18000  # Monthly: 1500
    }

    # Age categories
    if age < 25:
        age_category = 'young'
    elif age < 35:
        age_category = 'adult'
    elif age < 45:
        age_category = 'middle_age'
    else:
        age_category = 'senior'

    if insurance_type == 'Health Insurance':
        base_rate = health_base_rates[age_category]
    elif insurance_type == 'Life Insurance':
        base_rate = life_base_rates[age_category] * (coverage_amount / 10000000)

    if gender.lower() == 'female':
        base_rate *= 0.95

    high_risk_professions = ['manual labor', 'construction worker', 'police officer', 'firefighter']
    medium_risk_professions = ['teacher', 'it professional', 'office worker', 'salesperson']

    profession_lower = profession.lower()
    if profession_lower in [p.lower() for p in high_risk_professions]:
        base_rate *= 1.3
    elif profession_lower in [p.lower() for p in medium_risk_professions]:
        base_rate *= 1.1
    else:
        base_rate *= 1.2

    base_rate /= 12
    return base_rate

def calculate_discount(fitness_score):
    if fitness_score >= 90:
        return 30  # 30% discount
    elif fitness_score >= 80:
        return 25  # 25% discount
    elif fitness_score >= 70:
        return 20  # 20% discount
    elif fitness_score >= 60:
        return 15  # 15% discount
    elif fitness_score >= 50:
        return 10  # 10% discount
    elif fitness_score >= 40:
        return 5  # 5% discount
    else:
        return 0  # No discount


def generate_personalized_plan(age, gender, profession, fitness_score, insurance_type, coverage_amount=0):

    base_premium = calculate_base_premium(age, gender, profession, insurance_type, coverage_amount)

    discount_rate = calculate_discount(fitness_score)
    discount_amount = base_premium * (discount_rate / 100)
    discounted_premium = base_premium - discount_amount

    max_discount_rate = 30
    max_discount_amount = base_premium * (max_discount_rate / 100)
    max_discounted_premium = base_premium - max_discount_amount

    return base_premium, discount_rate, discount_amount, discounted_premium, max_discounted_premium

def plot_premium_trend(fitness_scores, base_premium):
    """Generate a plot showing premium trends based on different fitness scores"""
    # Convert range to list and create data points
    scores = list(fitness_scores)
    premiums = []
    for score in scores:
        discount = calculate_discount(score)
        premium = base_premium * (1 - discount/100)
        premiums.append(premium)
    
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
    
    # Add better styling
    fig.update_traces(
        line=dict(width=3),
        marker=dict(size=8)
    )
    
    return fig

def make_your_own_plan_page():
    st.title("Personalized Insurance Plan Generator")
    
    # Load fitness score model and data
    df, rf_regressor, label_encoders, scaler, feature_names, numerical_columns = load_data_and_model()
    
    if df is None or rf_regressor is None:
        st.error("Failed to load required model files. Please check if all model files exist.")
        return
    
    # Create tabs for better organization
    tab1, tab2 = st.tabs(["Create Plan", "Fitness Details"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Enter your name:")
            age = st.number_input("Age", min_value=18, max_value=100, step=1)
            gender = st.selectbox("Gender", options=["Male", "Female", "Other"])
            
            # Improved profession selection with categories
            profession_category = st.selectbox("Profession Category", options=list(PROFESSIONS.keys()))
            profession = st.selectbox("Specific Profession", options=PROFESSIONS[profession_category])
            
        with col2:
            insurance_type = st.selectbox("Insurance Type", options=["Health Insurance", "Life Insurance"])
            
            if insurance_type == "Life Insurance":
                coverage = st.selectbox("Coverage Amount", options=list(COVERAGE_AMOUNTS.keys()))
                coverage_amount = COVERAGE_AMOUNTS[coverage]
                
                # Add coverage calculator
                st.info(f"""
                💡 Coverage Calculator:
                - Monthly Income Protection: ~INR {coverage_amount/120:,.2f}
                - Years of Coverage: ~{coverage_amount/(age*50000):,.1f} years
                """)
            else:
                coverage_amount = 0

        # Get fitness score from the model
        if name and age:
            fitness_score, discount, selected_data = get_fitness_score_and_discount(
                df, rf_regressor, name, age, label_encoders, scaler, feature_names)
            if fitness_score is not None:
                st.success(f"Your fitness score has been calculated: {fitness_score:.2f}")
                
                # Display fitness metrics in a collapsible section
                with st.expander("View Your Fitness Metrics"):
                    display_user_metrics(selected_data, fitness_score, discount)
            else:
                st.warning("Please complete your fitness assessment first.")
                fitness_score = st.slider("Enter your fitness score manually:", 0, 100, 50,
                                        help="Higher fitness score leads to better discounts!")
        else:
            fitness_score = st.slider("Enter your fitness score:", 0, 100, 50,
                                    help="Higher fitness score leads to better discounts!")

        if st.button("Generate Plan", type="primary"):
            results = generate_personalized_plan(
                age, gender, profession, fitness_score, insurance_type, coverage_amount)
            base_premium, discount_rate, discount_amount, discounted_premium, max_discounted_premium = results
            
            # Display results in an organized way
            st.success("Plan Generated Successfully!")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Monthly Premium", f"₹{discounted_premium:,.2f}", 
                         f"-{discount_rate}%")
            with col2:
                st.metric("Base Premium", f"₹{base_premium:,.2f}")
            with col3:
                potential_savings = base_premium - max_discounted_premium
                st.metric("Monthly Savings", f"₹{potential_savings:,.2f}")
            
            # Add premium trend visualization
            st.subheader("Premium Trend Analysis")
            fig = plot_premium_trend(range(0, 101, 10), base_premium)
            st.plotly_chart(fig, use_container_width=True)
            
            # Add download button for detailed plan
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
    
    with tab2:
        if name and age and fitness_score is not None:
            st.header("Your Fitness Profile")
            # Display comprehensive fitness metrics from fitness_score_page
            display_user_metrics(selected_data, fitness_score, discount)
        else:
            st.info("Please enter your name and age in the Create Plan tab to view your fitness metrics.")

if __name__ == "__main__":
    st.set_page_config(page_title="Insurance Plan Generator", page_icon="🏥", layout="wide")
    make_your_own_plan_page()
