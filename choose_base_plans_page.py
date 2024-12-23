import pandas as pd
import numpy as np
import streamlit as st


def choose_base_plans_page():
    st.title('Insurance Policy Recommendation System')
    
    # Add a brief description with markdown
    st.markdown("""
    ### Let us help you find the perfect insurance policy! 
    Our AI-powered system analyzes your profile to recommend the most suitable insurance plan.
    
    ⚡ **Quick Tips**:
    - Lower premiums are better for tight budgets
    - Higher coverage is recommended for high-risk occupations
    - Consider family history in your decision
    """)

    # Load and preprocess data
    @st.cache_data  # Cache the data loading
    def load_and_preprocess_data():
        df = pd.read_csv("base_plans.csv")
        df.dropna(inplace=True)
        df.drop(['perc_premium_paid_by_cash_credit','Count_3-6_months_late','Count_6-12_months_late',
                'Count_more_than_12_months_late','application_underwriting_score','target',
                'sourcing_channel','residence_area_type'], axis='columns', inplace=True)
        
        df['age_in_years'] = df['age_in_days'] // 365
        
        np.random.seed(42)
        occupations = ['Engineer', 'Doctor', 'Teacher', 'Clerk', 'Manager', 'Laborer', 'Mechanic', 'Driver']
        df['occupation'] = np.random.choice(occupations, size=len(df))
        df['job_type'] = df['occupation'].apply(lambda x: 'White-collar' if x in ['Engineer', 'Doctor', 'Teacher', 'Manager'] else 'Blue-collar')
        
        # Enhanced policy assignment with more realistic options
        conditions = [
            (df['age_in_years'] < 18),
            (df['age_in_years'] >= 18) & (df['Income'] > 1500000),
            (df['age_in_years'] >= 18) & (df['Income'] > 1000000) & (df['Income'] <= 1500000),
            (df['age_in_years'] >= 18) & (df['Income'] > 500000) & (df['Income'] <= 1000000),
            (df['age_in_years'] >= 18) & (df['Income'] <= 500000)
        ]
        choices = ['SmartSure Junior Plus', 'Premium Elite Care', 'Comprehensive Shield', 'SmartSure Essential Pro', 'Basic Care Plus']
        df['policy'] = np.select(conditions, choices, default='Basic Care')
        
        return df, occupations

    df, occupations = load_and_preprocess_data()

    # Create tabs for better organization
    tab1, tab2 = st.tabs(["📋 Policy Finder", "ℹ️ Policy Details"])

    with tab1:
        st.subheader("Find Your Ideal Policy")
        
        col1, col2 = st.columns(2)
        
        with col1:
            user_age = st.number_input("Age (years):", min_value=0, max_value=100, step=1)
            user_income = st.number_input("Annual Income:", min_value=0, step=1000, format="%d")
        
        with col2:
            user_occupation = st.selectbox("Occupation:", occupations)
            risk_factor = st.slider("Health Risk Factor (1-10):", 1, 10, 5,
                                  help="1 = Excellent health, 10 = High risk")

        # Enhanced policy prediction
        user_age_in_days = user_age * 365
        if user_age_in_days < 6570:  # Under 18
            recommended_policy = "SmartSure Junior Plus"
            additional_info = "Comprehensive coverage for your child's health and future"
        else:
            if user_income > 1500000:
                recommended_policy = "Premium Elite Care"
                additional_info = "Premium coverage with global benefits and executive health features"
            elif user_income > 1000000:
                recommended_policy = "Comprehensive Shield"
                additional_info = "Extensive coverage with modern healthcare benefits"
            elif user_income > 500000:
                recommended_policy = "SmartSure Essential Pro"
                additional_info = "Balanced coverage with essential benefits"
            else:
                recommended_policy = "Basic Care Plus"
                additional_info = "Affordable coverage with essential health protection"

        # Display recommendation in an attractive box
        st.markdown(f"""
        <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px;'>
            <h3 style='color: #1f77b4;'>Recommended Policy: {recommended_policy}</h3>
            <p>{additional_info}</p>
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        st.subheader("Available Insurance Policies")
        
        # Display policy details in an expandable format
        policies = {
            "SmartSure Junior Plus": {
                "Age Range": "0-18 years",
                "Features": [
                    "Education fund protection",
                    "Child-specific critical illness coverage",
                    "Annual health checkups",
                    "Vaccination coverage",
                    "Dental and vision care",
                    "Parent hospitalization benefit"
                ],
                "Premium Range": "₹8,000 - ₹20,000 annually",
                "Sum Insured": "₹5 Lakhs - ₹25 Lakhs"
            },
            "Premium Elite Care": {
                "Age Range": "18-65 years",
                "Features": [
                    "Global coverage including USA/Canada",
                    "Executive health checkups",
                    "Advanced robotics surgery coverage",
                    "Alternative treatment coverage (AYUSH)",
                    "Personal accident cover up to ₹1 Crore",
                    "Air ambulance coverage",
                    "Recovery benefit",
                    "Zero room rent capping"
                ],
                "Premium Range": "₹35,000 - ₹75,000 annually",
                "Sum Insured": "₹50 Lakhs - ₹2 Crores"
            },
            "Comprehensive Shield": {
                "Age Range": "18-70 years",
                "Features": [
                    "Comprehensive hospitalization coverage",
                    "Pre & post hospitalization expenses",
                    "Critical illness coverage",
                    "No claim bonus up to 100%",
                    "Daily hospital cash allowance",
                    "Organ donor expenses",
                    "Modern treatment methods"
                ],
                "Premium Range": "₹20,000 - ₹45,000 annually",
                "Sum Insured": "₹25 Lakhs - ₹1 Crore"
            },
            "SmartSure Essential Pro": {
                "Age Range": "18-75 years",
                "Features": [
                    "Cashless hospitalization",
                    "Day care procedures",
                    "Annual health checkup",
                    "Pre-existing disease coverage after 2 years",
                    "Ambulance charges",
                    "Tax benefits under Section 80D",
                    "Family floater option"
                ],
                "Premium Range": "₹12,000 - ₹30,000 annually",
                "Sum Insured": "₹10 Lakhs - ₹50 Lakhs"
            },
            "Basic Care Plus": {
                "Age Range": "18-65 years",
                "Features": [
                    "In-patient hospitalization",
                    "Basic critical illness coverage",
                    "Network hospital cashless facility",
                    "Pre & post hospitalization (30/60 days)",
                    "Affordable premiums",
                    "No medical test up to 45 years"
                ],
                "Premium Range": "₹6,000 - ₹15,000 annually",
                "Sum Insured": "₹3 Lakhs - ₹10 Lakhs"
            },
            "Senior Citizen Special": {
                "Age Range": "60-80 years",
                "Features": [
                    "Specialized senior citizen coverage",
                    "Pre-existing disease coverage after 1 year",
                    "Regular health check-ups",
                    "Domiciliary hospitalization",
                    "AYUSH treatment coverage",
                    "Chronic condition management"
                ],
                "Premium Range": "₹15,000 - ₹40,000 annually",
                "Sum Insured": "₹5 Lakhs - ₹25 Lakhs"
            },
            "Family Floater Supreme": {
                "Age Range": "18-65 years (Primary insured)",
                "Features": [
                    "Coverage for entire family under single sum insured",
                    "Maternity benefits",
                    "New born baby coverage",
                    "No claim bonus for family",
                    "Preventive healthcare benefits",
                    "Second medical opinion"
                ],
                "Premium Range": "₹25,000 - ₹60,000 annually",
                "Sum Insured": "₹15 Lakhs - ₹1 Crore"
            },
            "Critical Illness Shield": {
                "Age Range": "18-65 years",
                "Features": [
                    "Coverage for 36 critical illnesses",
                    "Lump sum payout on diagnosis",
                    "Income tax benefit",
                    "No medical test up to 45 years",
                    "Survival period of 30 days",
                    "Premium waiver benefit"
                ],
                "Premium Range": "₹8,000 - ₹30,000 annually",
                "Sum Insured": "₹10 Lakhs - ₹50 Lakhs"
            }
        }

        for policy, details in policies.items():
            with st.expander(f"📄 {policy}"):
                st.write("**Age Range:**", details["Age Range"])
                st.write("**Features:**")
                for feature in details["Features"]:
                    st.write(f"- {feature}")
                st.write("**Premium Range:**", details["Premium Range"])

if __name__ == "__main__":
    choose_base_plans_page()
