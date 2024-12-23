import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import plotly.express as px
import plotly.graph_objects as go
import base64
from fpdf import FPDF
import joblib
from datetime import datetime, timedelta
import random

# Add custom CSS for better UI
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Function to load data and model, and preprocess data
@st.cache_resource
def load_data_and_model():
    with st.spinner('Loading data and model...'):
        try:
            model = joblib.load('models/fitness_model_rf.joblib')
            df = pd.read_csv('models/processed_fitness_claim_dataset.csv')
            label_encoders = joblib.load('models/label_encoders.joblib')
            scaler = joblib.load('models/scaler.joblib')
            feature_names = joblib.load('models/feature_names.joblib')
            numerical_columns = joblib.load('models/numerical_columns.joblib')
            return df, model, label_encoders, scaler, feature_names, numerical_columns
        except Exception as e:
            st.error(f"Error loading model and data: {str(e)}")
            return None, None, None, None, None, None

def create_pdf_certificate(data, fitness_score, discount):
    pdf = FPDF()
    pdf.add_page()
    
    # Add certificate styling
    pdf.set_fill_color(240, 248, 255)
    pdf.rect(0, 0, 210, 297, 'F')
    
    # Add header
    pdf.set_font('Arial', 'B', 24)
    pdf.set_text_color(44, 62, 80)
    pdf.cell(0, 30, 'Health & Fitness Certificate', 0, 1, 'C')
    
    # Add data in a structured format
    pdf.set_font('Arial', '', 12)
    y_position = 60
    
    # Add metrics in a grid layout
    metrics = {
        'Fitness Metrics': ['BMI', 'Heart Beats', 'Blood Pressure', 'SpO2 Levels'],
        'Activity Metrics': ['Steps Taken', 'Active Minutes', 'Calories Burned'],
        'Wellness Metrics': ['Sleep Duration', 'Sleep Quality', 'Stress Levels']
    }
    
    for category, items in metrics.items():
        pdf.set_font('Arial', 'B', 14)
        pdf.set_xy(20, y_position)
        pdf.cell(0, 10, category, 0, 1)
        y_position += 15
        
        pdf.set_font('Arial', '', 12)
        for item in items:
            if item in data:
                pdf.set_xy(30, y_position)
                pdf.cell(0, 10, f"{item}: {data[item]}", 0, 1)
                y_position += 10
        
        y_position += 10

    # Add fitness score and discount
    pdf.set_xy(20, y_position)
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, f"Final Fitness Score: {fitness_score:.2f}", 0, 1)
    pdf.set_xy(20, y_position + 15)
    pdf.cell(0, 10, f"Insurance Discount Earned: {discount}%", 0, 1)
    
    return pdf.output(dest='S').encode('latin-1')

def create_radar_chart(metrics):
    categories = ['Physical Fitness', 'Vital Signs', 'Sleep Quality', 
                 'Activity Level', 'Stress Management']
    
    # Calculate category scores
    scores = [
        (metrics.get('Steps Taken', 0) / 10000 + 
         metrics.get('Calories Burned', 0) / 2500) * 50,  # Physical Fitness
        (metrics.get('SpO2 Levels', 0) / 100 + 
         (100 - metrics.get('Heart Beats', 0)) / 100) * 50,  # Vital Signs
        metrics.get('Sleep Quality', 0) * 10,  # Sleep Quality
        metrics.get('Active Minutes', 0) / 60 * 100,  # Activity Level
        (10 - metrics.get('Stress Levels', 0)) * 10  # Stress Management
    ]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=scores,
        theta=categories,
        fill='toself',
        name='Current Profile'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=False,
        title="Fitness Component Breakdown"
    )
    return fig

def create_historical_trends():
    # Simulate historical data for demonstration
    dates = [(datetime.now() - timedelta(days=x)).strftime('%Y-%m-%d') 
            for x in range(30, -1, -1)]
    
    fig = go.Figure()
    
    # Simulate different metrics
    fig.add_trace(go.Scatter(x=dates, y=[random.randint(60, 90) for _ in dates],
                            name='Sleep Quality', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=dates, y=[random.randint(70, 85) for _ in dates],
                            name='Activity Score', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=dates, y=[random.randint(75, 95) for _ in dates],
                            name='Vital Signs', line=dict(color='red')))
    
    fig.update_layout(
        title="30-Day Health Trends",
        xaxis_title="Date",
        yaxis_title="Score",
        hovermode='x unified'
    )
    return fig

def generate_recommendations(metrics, fitness_score):
    recommendations = {
        'Physical Activity': [],
        'Sleep': [],
        'Stress Management': [],
        'Vital Signs': []
    }
    
    # Physical Activity recommendations
    if metrics.get('Steps Taken', 0) < 8000:
        recommendations['Physical Activity'].append(
            "Try to increase daily steps to 10,000 by taking short walks")
    if metrics.get('Active Minutes', 0) < 30:
        recommendations['Physical Activity'].append(
            "Aim for at least 30 minutes of moderate activity daily")
    
    # Sleep recommendations
    if metrics.get('Sleep Quality', 0) < 7:
        recommendations['Sleep'].append(
            "Improve sleep quality by maintaining a consistent sleep schedule")
    if metrics.get('Sleep Duration', 0) < 7:
        recommendations['Sleep'].append(
            "Aim for 7-9 hours of sleep per night")
    
    # Stress Management
    if metrics.get('Stress Levels', 0) > 7:
        recommendations['Stress Management'].append(
            "Consider incorporating meditation or breathing exercises")
    
    # Vital Signs
    if metrics.get('Heart Beats', 0) > 80:
        recommendations['Vital Signs'].append(
            "Consider cardiovascular exercises to improve heart rate")
    
    return recommendations

def display_user_metrics(selected_data, fitness_score, discount):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Fitness Score", f"{fitness_score:.2f}")
    with col2:
        st.metric("Discount Earned", f"{discount}%")
    with col3:
        st.metric("Health Status", get_health_status(fitness_score))

    # Add radar chart
    radar_fig = create_radar_chart(selected_data)
    st.plotly_chart(radar_fig)
    
    # Add historical trends
    st.subheader("Health Trends")
    trends_fig = create_historical_trends()
    st.plotly_chart(trends_fig)
    
    # Display detailed metrics in expandable sections
    with st.expander("View Detailed Health Metrics"):
        metrics_cols = st.columns(3)
        
        vital_metrics = {
            "Blood Pressure": f"{selected_data.get('Blood Pressure (Systolic)', 'N/A')}/{selected_data.get('Blood Pressure (Diastolic)', 'N/A')}",
            "Heart Rate": f"{selected_data.get('Heart Beats', 'N/A')} bpm",
            "SpO2": f"{selected_data.get('SpO2 Levels', 'N/A')}%"
        }
        
        activity_metrics = {
            "Steps": selected_data.get('Steps Taken', 'N/A'),
            "Active Minutes": selected_data.get('Active Minutes', 'N/A'),
            "Calories": f"{selected_data.get('Calories Burned', 'N/A')} kcal"
        }
        
        wellness_metrics = {
            "Sleep Duration": f"{selected_data.get('Sleep Duration', 'N/A')} hrs",
            "Sleep Quality": f"{selected_data.get('Sleep Quality', 'N/A')}/10",
            "Stress Level": f"{selected_data.get('Stress Levels', 'N/A')}/10"
        }
        
        for i, (title, metrics) in enumerate(zip(
            ["Vital Signs", "Activity Metrics", "Wellness Metrics"],
            [vital_metrics, activity_metrics, wellness_metrics]
        )):
            with metrics_cols[i]:
                st.subheader(title)
                for key, value in metrics.items():
                    st.markdown(f"**{key}:** {value}")
    
    # Add personalized recommendations
    st.subheader("Personalized Recommendations")
    recommendations = generate_recommendations(selected_data, fitness_score)
    
    for category, rec_list in recommendations.items():
        if rec_list:  # Only show categories with recommendations
            with st.expander(f"{category} Recommendations"):
                for rec in rec_list:
                    st.write(f"• {rec}")

def get_health_status(fitness_score):
    if fitness_score >= 90:
        return "Excellent"
    elif fitness_score >= 75:
        return "Very Good"
    elif fitness_score >= 60:
        return "Good"
    elif fitness_score >= 45:
        return "Fair"
    else:
        return "Needs Improvement"

def get_fitness_score_and_discount(df, rf_regressor, name, age, label_encoders, scaler, feature_names):
    row = df[(df['Name'] == name) & (df['Age'] == age)]
    if not row.empty:
        features = row.drop(['Name', 'Fitness Score'], axis=1)
        
        # Apply preprocessing steps
        for column in features.select_dtypes(include=['object']).columns:
            if column in label_encoders:
                features[column] = label_encoders[column].transform(features[column])
        
        numerical_cols = features.select_dtypes(include=[np.number]).columns
        scaled_features = scaler.transform(features[numerical_cols])
        features[numerical_cols] = scaled_features
        
        # Ensure the order of columns matches the training data
        features = features[feature_names]
        
        fitness_score = rf_regressor.predict(features)[0]
        discount = predict_discount(fitness_score)
        selected_data = row.iloc[0].to_dict()
        return fitness_score, discount, selected_data
    else:
        return None, None, None

def predict_discount(fitness_score):
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

def display_model_metrics(df, rf_regressor, label_encoders, scaler, feature_names):
    # Prepare features
    X = df.drop(['Name', 'Fitness Score'], axis=1)
    y = df['Fitness Score']
    
    # Ensure columns are in the correct order
    X = X[feature_names]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Make predictions
    try:
        y_pred_test = rf_regressor.predict(X_test)
        y_pred_train = rf_regressor.predict(X_train)
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        target_variance = np.var(y_test)
        explained_variance_percentage = (1 - (test_rmse ** 2 / target_variance)) * 100
        r_squared = r2_score(y_test, y_pred_test)
        mae = mean_absolute_error(y_test, y_pred_test)
        mse = mean_squared_error(y_test, y_pred_test)

        # Display metrics
        st.write("## Model Performance Metrics")
        metrics_cols = st.columns(2)
        
        with metrics_cols[0]:
            st.metric("Train RMSE", f"{train_rmse:.2f}")
            st.metric("Test RMSE", f"{test_rmse:.2f}")
            st.metric("R-squared", f"{r_squared:.2f}")
        
        with metrics_cols[1]:
            st.metric("MAE", f"{mae:.2f}")
            st.metric("MSE", f"{mse:.2f}")
            st.metric("Explained Variance", f"{explained_variance_percentage:.2f}%")
            
    except Exception as e:
        st.error(f"Error calculating metrics: {str(e)}")
        st.info("Please retrain the model to ensure compatibility with the current scikit-learn version.")

def fitness_score_page():
    st.title('Fitness Score & Insurance Discount Calculator')
    
    # Load data and model with numerical columns
    df, rf_regressor, label_encoders, scaler, feature_names, numerical_columns = load_data_and_model()
    
    if df is None or rf_regressor is None:
        st.error("Failed to load required model files. Please check if all model files exist.")
        return
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Calculate Score", "View Analytics", "Model Performance"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Enter your name:")
        with col2:
            age = st.number_input("Enter your age:", min_value=0, max_value=120)
            
        if st.button('Calculate Fitness Score & Generate Certificate'):
            if name and age:
                fitness_score, discount, selected_data = get_fitness_score_and_discount(df, rf_regressor, name, age, label_encoders, scaler, feature_names)
                if fitness_score is not None:
                    display_user_metrics(selected_data, fitness_score, discount)
                    
                    # Generate and provide certificate download
                    pdf_bytes = create_pdf_certificate(selected_data, fitness_score, discount)
                    st.download_button(
                        label="Download Health Certificate",
                        data=pdf_bytes,
                        file_name="health_certificate.pdf",
                        mime="application/pdf"
                    )
                else:
                    st.error(f"No data found for {name} (age {age}). Please check your input.")
            else:
                st.warning("Please enter both name and age.")
    
    with tab2:
        st.subheader("Population Health Analytics")
        # Display 3D scatter plot
        viz_df = df.copy()
        viz_df['Predicted Discount'] = viz_df['Fitness Score'].apply(predict_discount)
        
        fig = px.scatter_3d(
            viz_df,
            x='Fitness Score',
            y='Predicted Discount',
            z='Age',
            color='Predicted Discount',
            size='Fitness Score',
            hover_data=['Name', 'Age'],
            title='3D Visualization of Fitness Scores and Discounts'
        )
        
        fig.update_layout(
            scene=dict(
                xaxis_title='Fitness Score',
                yaxis_title='Predicted Discount (%)',
                zaxis_title='Age'
            ),
            width=800,
            height=800
        )
        
        st.plotly_chart(fig)
        
        # Add distribution plots
        col1, col2 = st.columns(2)
        with col1:
            fig_dist = px.histogram(df, x='Fitness Score', title='Distribution of Fitness Scores')
            st.plotly_chart(fig_dist)
        with col2:
            fig_age = px.box(df, x='Age', y='Fitness Score', title='Fitness Score by Age')
            st.plotly_chart(fig_age)
    
    with tab3:
        display_model_metrics(df, rf_regressor, label_encoders, scaler, feature_names)

# Main function to control page navigation
def main():
    st.sidebar.title('Navigation')
    page = st.sidebar.radio("Go to", ('Home', 'AI Assistant', 'Fitness Score'))

    if page == "Home":
        st.title('Home Page')
        st.write("Welcome to the Fitness Claim Discosunt Predictor app.")
    elif page == "AI Assistant":
        st.title('AI Assistant Page')
        st.write("This is the AI assistant page content.")
    elif page == "Fitness Score":
        fitness_score_page()

if __name__ == "__main__":
    main()