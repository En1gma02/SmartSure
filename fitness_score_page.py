import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go
import base64
from fpdf import FPDF
import joblib
from datetime import datetime
import random

def load_data_and_model():
    try:
        # Load model and preprocessors
        rf_regressor = joblib.load('models/fitness_model_rf.joblib')
        scaler = joblib.load('models/scaler.joblib')
        feature_names = joblib.load('models/feature_names.joblib')
        float_columns = joblib.load('models/float_columns.joblib')
        integer_columns = joblib.load('models/integer_columns.joblib')
        
        # Load both ML and visualization datasets
        ml_df = pd.read_csv('models/ml_processed_dataset.csv')
        viz_df = pd.read_csv('models/viz_processed_dataset.csv')
        
        return viz_df, ml_df, rf_regressor, scaler, feature_names, float_columns, integer_columns
    except Exception as e:
        st.error(f"Error loading model files: {str(e)}")
        return None, None, None, None, None, None, None

def get_fitness_score_and_discount(viz_df, ml_df, rf_regressor, name, age, scaler, feature_names, float_columns, integer_columns):
    try:
        # Find user in visualization dataset
        user_data = viz_df[viz_df['Name'] == name]
        if not user_data.empty and user_data['Age'].iloc[0] == age:
            # Get corresponding ML data for prediction
            ml_user_data = ml_df[ml_df['Name'] == name]
            
            # Extract features in correct order for prediction
            X = ml_user_data[feature_names]
            
            # Get fitness score
            fitness_score = float(user_data['Fitness Score'].iloc[0])
            
            # Calculate discount based on fitness score
            if fitness_score >= 90:
                discount = 30
            elif fitness_score >= 80:
                discount = 25
            elif fitness_score >= 70:
                discount = 20
            elif fitness_score >= 60:
                discount = 15
            elif fitness_score >= 50:
                discount = 10
            elif fitness_score >= 40:
                discount = 5
            else:
                discount = 0
                
            return fitness_score, discount, user_data.iloc[0]
    except Exception as e:
        st.error(f"Error calculating fitness score: {str(e)}")
    return None, None, None

def display_user_metrics(user_data, fitness_score, discount):
    try:
        # Main metrics display
        st.write("## Your Health Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            fitness_status = ("Good" if fitness_score >= 70 else 
                            "Average" if fitness_score >= 50 else "Needs Improvement")
            st.metric("Fitness Score", f"{fitness_score:.2f}", 
                     delta=fitness_status,
                     delta_color="normal" if fitness_score >= 50 else "inverse")
        with col2:
            bp_normal = 120 <= user_data['Blood Pressure (Systolic)'] <= 129
            bp_status = "Normal" if bp_normal else "High"
            st.metric("Blood Pressure", 
                     f"{int(user_data['Blood Pressure (Systolic)'])}/{int(user_data['Blood Pressure (Diastolic)'])}",
                     delta=bp_status,
                     delta_color="normal" if bp_normal else "inverse")
        with col3:
            bmi_normal = 18.5 <= user_data['BMI'] <= 24.9
            bmi_status = "Normal" if bmi_normal else ("High" if user_data['BMI'] > 24.9 else "Low")
            st.metric("BMI", f"{user_data['BMI']:.1f}", 
                     delta=bmi_status,
                     delta_color="normal" if bmi_normal else "inverse")
        with col4:
            st.metric("Insurance Discount", f"{discount}%", 
                     delta=f"Save ₹{discount*100:,}/month",
                     delta_color="normal")

        # Detailed metrics in expandable section
        with st.expander("View Detailed Metrics"):
            col1, col2, col3 = st.columns(3)
            
            # First row
            with col1:
                heart_normal = 60 <= user_data['Heart Beats'] <= 100
                st.metric("Heart Rate",
                    f"{int(user_data['Heart Beats'])} bpm",
                    "Normal" if heart_normal else "Irregular",
                    delta_color="normal" if heart_normal else "inverse")
            
            with col2:
                steps_good = user_data['Steps Taken'] >= 10000
                st.metric("Steps Taken",
                    f"{int(user_data['Steps Taken']):,}",
                    "Goal Achieved" if steps_good else "Below Target",
                    delta_color="normal" if steps_good else "inverse")
            
            with col3:
                active_good = user_data['Active Minutes'] >= 150
                st.metric("Active Minutes",
                    f"{int(user_data['Active Minutes'])} min",
                    "Goal Achieved" if active_good else "Below Target",
                    delta_color="normal" if active_good else "inverse")
            
            # Second row
            with col1:
                sleep_good = 7 <= user_data['Sleep Duration'] <= 9
                st.metric("Sleep Duration",
                    f"{user_data['Sleep Duration']:.1f} hrs",
                    "Optimal" if sleep_good else "Suboptimal",
                    delta_color="normal" if sleep_good else "inverse")
            
            with col2:
                quality_good = user_data['Sleep Quality'] >= 7
                st.metric("Sleep Quality",
                    f"{int(user_data['Sleep Quality'])}/10",
                    "Good" if quality_good else "Poor",
                    delta_color="normal" if quality_good else "inverse")
            
            with col3:
                stress_good = user_data['Stress Levels'] <= 4
                st.metric("Stress Level",
                    f"{int(user_data['Stress Levels'])}/10",
                    "Low" if stress_good else "High",
                    delta_color="normal" if stress_good else "inverse")

        # Visualizations
        st.write("### Your Health Insights")
        
        # Radar Chart of Key Metrics
        col1, col2 = st.columns(2)
        with col1:
            # Normalize metrics for radar chart
            metrics_radar = {
                'Fitness': fitness_score/100,
                'Sleep': user_data['Sleep Quality']/10,
                'Activity': min(user_data['Steps Taken']/10000, 1),
                'Heart': (100 - abs(75 - user_data['Heart Beats']))/100,
                'Stress': (10 - user_data['Stress Levels'])/10,
                'BMI Health': 1 - abs(22 - user_data['BMI'])/22
            }
            
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=list(metrics_radar.values()),
                theta=list(metrics_radar.keys()),
                fill='toself',
                name='Current Status',
                line_color='rgb(67, 147, 195)',
                fillcolor='rgba(67, 147, 195, 0.3)'
            ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=False,
                title='Health Metrics Overview'
            )
            st.plotly_chart(fig_radar)
            
        with col2:
            # Generate monthly trend data (simulated)
            dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
            base_steps = user_data['Steps Taken']
            base_heart = user_data['Heart Beats']
            
            trend_data = pd.DataFrame({
                'Date': dates,
                'Steps': [max(0, base_steps + random.randint(-1000, 1000)) for _ in range(30)],
                'Heart Rate': [max(60, min(100, base_heart + random.randint(-5, 5))) for _ in range(30)]
            })
            
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                x=trend_data['Date'],
                y=trend_data['Steps'],
                name='Daily Steps',
                line=dict(color='rgb(67, 147, 195)', width=2)
            ))
            
            fig_trend.add_trace(go.Scatter(
                x=trend_data['Date'],
                y=trend_data['Heart Rate'],
                name='Heart Rate',
                yaxis='y2',
                line=dict(color='rgb(195, 67, 67)', width=2)
            ))
            
            fig_trend.update_layout(
                title='30-Day Activity Trend',
                xaxis=dict(title='Date'),
                yaxis=dict(
                    title='Steps',
                    titlefont=dict(color='rgb(67, 147, 195)'),
                    tickfont=dict(color='rgb(67, 147, 195)')
                ),
                yaxis2=dict(
                    title='Heart Rate (bpm)',
                    titlefont=dict(color='rgb(195, 67, 67)'),
                    tickfont=dict(color='rgb(195, 67, 67)'),
                    overlaying='y',
                    side='right'
                ),
                showlegend=True
            )
            st.plotly_chart(fig_trend)
                
    except Exception as e:
        st.error(f"Error displaying metrics: {str(e)}")

def create_pdf_certificate(user_data, fitness_score, discount):
    try:
        pdf = FPDF()
        pdf.add_page()
        
        # Add some color
        pdf.set_fill_color(240, 248, 255)  # Light blue background
        pdf.rect(0, 0, 210, 297, 'F')
        
        # Certificate header with blue header
        pdf.set_fill_color(67, 147, 195)
        pdf.rect(0, 0, 210, 40, 'F')
        
        pdf.set_text_color(255, 255, 255)
        pdf.set_font('Arial', 'B', 28)
        pdf.cell(0, 30, 'Health Assessment Certificate', 0, 1, 'C')
        
        # Reset text color
        pdf.set_text_color(0, 0, 0)
        
        # User details section
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 20, '', 0, 1)  # Add some space
        pdf.cell(0, 10, 'Personal Details', 0, 1)
        
        pdf.set_font('Arial', '', 12)
        pdf.cell(0, 8, f"Name: {user_data['Name']}", 0, 1)
        pdf.cell(0, 8, f"Date: {datetime.now().strftime('%B %d, %Y')}", 0, 1)
        pdf.cell(0, 8, f"Fitness Score: {fitness_score:.2f}/100", 0, 1)
        pdf.cell(0, 8, f"Insurance Discount: {discount}%", 0, 1)
        
        # Health metrics section
        pdf.ln(10)
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'Health Metrics Summary', 0, 1)
        
        # Create two columns for metrics
        pdf.set_font('Arial', '', 12)
        x_left = 20
        x_right = 110
        y_start = pdf.get_y()
        
        # Left column metrics
        pdf.set_xy(x_left, y_start)
        metrics_left = [
            f"BMI: {user_data['BMI']:.1f}",
            f"Blood Pressure: {int(user_data['Blood Pressure (Systolic)'])}/{int(user_data['Blood Pressure (Diastolic)'])}",
            f"Heart Rate: {int(user_data['Heart Beats'])} bpm",
            f"Steps Taken: {int(user_data['Steps Taken']):,}",
            f"Active Minutes: {int(user_data['Active Minutes'])} min"
        ]
        
        # Right column metrics
        metrics_right = [
            f"Sleep Duration: {user_data['Sleep Duration']:.1f} hrs",
            f"Sleep Quality: {int(user_data['Sleep Quality'])}/10",
            f"VO2 Max: {user_data['VO2 Max']:.1f}",
            f"SpO2 Levels: {user_data['SpO2 Levels']:.1f}%",
            f"Stress Level: {int(user_data['Stress Levels'])}/10"
        ]
        
        for left, right in zip(metrics_left, metrics_right):
            pdf.set_xy(x_left, pdf.get_y())
            pdf.cell(80, 8, left, 0, 0)
            pdf.set_xy(x_right, pdf.get_y())
            pdf.cell(80, 8, right, 0, 1)
        
        # Add footer with logo and text
        footer_y = 250  # Adjust this value to position the footer
        
        # Add logo in center of footer
        try:
            logo_width = 40
            logo_x = (210 - logo_width) / 2  # Center the logo
            pdf.image('images/SmartSure_logo.png', logo_x, footer_y, logo_width)
        except:
            pass
        
        # Add footer text below logo
        pdf.set_y(footer_y + 30)  # Position text below logo
        pdf.set_font('Arial', 'I', 8)
        pdf.cell(0, 10, 'This certificate is generated based on your health metrics and is valid for insurance purposes.', 0, 1, 'C')
        pdf.cell(0, 5, f'Generated on {datetime.now().strftime("%B %d, %Y")}', 0, 1, 'C')
            
        return pdf.output(dest='S').encode('latin1')
        
    except Exception as e:
        st.error(f"Error creating certificate: {str(e)}")
        return None

def display_model_metrics(viz_df, ml_df, rf_regressor, feature_names):
    try:
        # Prepare features from ML dataset
        X = ml_df[feature_names]
        y = ml_df['Fitness Score']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Make predictions
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

        # Display metrics in 3x2 grid
        st.write("## Model Performance Metrics")
        col1, col2, col3 = st.columns(3)
        
        # Row 1
        with col1:
            st.metric("Train RMSE", f"{train_rmse:.2f}")
        with col2:
            st.metric("Test RMSE", f"{test_rmse:.2f}")
        with col3:
            st.metric("R-squared", f"{r_squared:.2f}")
        
        # Row 2
        with col1:
            st.metric("MAE", f"{mae:.2f}")
        with col2:
            st.metric("MSE", f"{mse:.2f}")
        with col3:
            st.metric("Explained Variance", f"{explained_variance_percentage:.2f}%")
            
    except Exception as e:
        st.error(f"Error calculating metrics: {str(e)}")
        st.info("Please retrain the model to ensure compatibility.")

def fitness_score_page():
    st.title('Fitness Score & Insurance Discount Calculator')
    
    # Load data and model
    viz_df, ml_df, rf_regressor, scaler, feature_names, float_columns, integer_columns = load_data_and_model()
    
    if viz_df is None or ml_df is None or rf_regressor is None:
        st.error("Failed to load required model files. Please check if all model files exist.")
        return
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Calculate Score", "View Analytics", "Model Performance"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Enter your name: (Try 'Heer Kibe')")
        with col2:
            age = st.number_input("Enter your age: (Try '54')", min_value=0, max_value=120)
            
        if st.button('Calculate Fitness Score & Generate Certificate'):
            if name and age:
                fitness_score, discount, user_data = get_fitness_score_and_discount(
                    viz_df, ml_df, rf_regressor, name, age, scaler, feature_names, float_columns, integer_columns
                )
                if fitness_score is not None:
                    display_user_metrics(user_data, fitness_score, discount)
                    
                    # Generate and provide certificate download
                    pdf_bytes = create_pdf_certificate(user_data, fitness_score, discount)
                    if pdf_bytes:
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
        st.write("## Population Health Analytics")
        
        # 3D scatter plot first
        st.write("### Age, Fitness Score & Discount Relationship")
        
        # Calculate discount for each fitness score
        discounts = viz_df['Fitness Score'].apply(lambda x: 30 if x >= 90 else (
            25 if x >= 80 else (
            20 if x >= 70 else (
            15 if x >= 60 else (
            10 if x >= 50 else (
            5 if x >= 40 else 0))))))
        
        # Create 3D scatter plot with ice blue theme
        fig_3d = go.Figure(data=[go.Scatter3d(
            x=viz_df['Age'],
            y=viz_df['Fitness Score'],
            z=discounts,
            mode='markers',
            marker=dict(
                size=5,
                color=discounts,
                colorscale='Ice',  # Changed to ice blue theme
                opacity=0.8,
                colorbar=dict(title="Discount %")
            ),
            text=viz_df['Name'],  # Add names for hover
            hovertemplate=
            "<b>Name</b>: %{text}<br>" +
            "<b>Age</b>: %{x}<br>" +
            "<b>Fitness Score</b>: %{y:.2f}<br>" +
            "<b>Discount</b>: %{z}%<br>" +
            "<extra></extra>"
        )])
        
        fig_3d.update_layout(
            scene=dict(
                xaxis_title='Age',
                yaxis_title='Fitness Score',
                zaxis_title='Discount %'
            ),
            width=800,
            height=800,
            title='3D Visualization of Age, Fitness Score & Insurance Discount'
        )
        
        st.plotly_chart(fig_3d, use_container_width=True)
        
        # Distribution plots below
        st.write("### Distribution Analysis")
        col1, col2 = st.columns(2)
        with col1:
            fig_dist = px.histogram(viz_df, x='Fitness Score', title='Distribution of Fitness Scores')
            st.plotly_chart(fig_dist)
        with col2:
            fig_age = px.box(viz_df, x='Age', y='Fitness Score', title='Fitness Score by Age')
            st.plotly_chart(fig_age)
    
    with tab3:
        display_model_metrics(viz_df, ml_df, rf_regressor, feature_names)

if __name__ == "__main__":
    fitness_score_page()