import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load the trained model and preprocessing objects
@st.cache_resource
def load_model_components():
    model = joblib.load('churn_model.pkl')
    scaler = joblib.load('scaler.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
    feature_columns = joblib.load('feature_columns.pkl')
    return model, scaler, label_encoders, feature_columns

# Load data for dashboard
@st.cache_data
def load_data():
    return pd.read_csv('PredictionData.csv')

def main():
    st.set_page_config(
        page_title="Customer Churn Prediction Dashboard",
        page_icon="🔮",
        layout="wide"
    )
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Analytics Dashboard", "Churn Prediction"]
    )
    
    if page == "Analytics Dashboard":
        show_powerbi_page()
    elif page == "Churn Prediction":
        show_prediction_page()


def show_powerbi_page():
    st.title("Customer Churn Analytics Dashboard")
    st.subheader("Analytics Dashboard")
    
    # Paste your Power BI URL directly here
    powerbi_url = "https://app.powerbi.com/view?r=eyJrIjoiZTdkY2ExMTYtMmFjOC00YTFhLWI5MmEtMWRmNmY1MTBmMTgyIiwidCI6IjM1MjFjMTZiLTBlYzgtNDBjMi04ZDQ2LTg1OTViZDEyNjM5MyJ9" # Replace with your actual Power BI embed URL
    
    if powerbi_url:
        # Embed Power BI dashboard directly
        st.components.v1.iframe(
            src=powerbi_url,
            width=1200,
            height=740,
            scrolling=True
        )
    else:
        # Show sample dashboard when no URL is provided
        st.info("URL INVALID")

def show_prediction_page():
    st.title("Customer Churn Prediction")
    
    try:
        model, scaler, label_encoders, feature_columns = load_model_components()
        
        st.header("Enter Customer Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No"])
            internet_service = st.selectbox("Internet Service", ["Yes", "No"])
            internet_type = st.selectbox("Internet Type", ["DSL", "Fiber Optic", "Cable", "None"])
            online_security = st.selectbox("Online Security", ["Yes", "No"])
            online_backup = st.selectbox("Online Backup", ["Yes", "No"])
            
            streaming_tv = st.selectbox("Streaming TV", ["Yes", "No"])
            streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No"])
            streaming_music = st.selectbox("Streaming Music", ["Yes", "No"])
        
        with col2:
            unlimited_data = st.selectbox("Unlimited Data", ["Yes", "No"])
            contract = st.selectbox("Contract", ["Month-to-Month", "One Year", "Two Year"])
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment_method = st.selectbox("Payment Method", ["Bank Withdrawal", "Credit Card", "Mailed Check"])
            monthly_charge = st.number_input("Monthly Charge", 0.0, 200.0, 65.0)
            total_charges = st.number_input("Total Charges", 0.0, 10000.0, 1500.0)
            total_refunds = st.number_input("Total Refunds", 0.0, 1000.0, 0.0)
            device_protection = st.selectbox("Device Protection Plan", ["Yes", "No"])
            premium_support = st.selectbox("Premium Support", ["Yes", "No"])
            

        with col3:
            extra_data_charges = st.number_input("Total Extra Data Charges", 0.0, 500.0, 0.0)
            long_distance_charges = st.number_input("Total Long Distance Charges", 0.0, 5000.0, 500.0)
            total_revenue = st.number_input("Total Revenue", 0.0, 15000.0, 2000.0)
            married = st.selectbox("Married", ["Yes", "No"])
            gender = st.selectbox("Gender", ["Male", "Female"])
            age = st.slider("Age", 18, 100, 35)
            referrals = st.slider("Number of Referrals", 0, 15, 2)
            tenure = st.slider("Tenure (Months)", 1, 72, 12)

        
            # Create input dataframe
            input_data = pd.DataFrame({
                'Gender': [gender],
                'Age': [age],
                'Married': [married],
                'Number_of_Referrals': [referrals],
                'Tenure_in_Months': [tenure],
                'Phone_Service': [phone_service],
                'Multiple_Lines': [multiple_lines],
                'Internet_Service': [internet_service],
                'Internet_Type': [internet_type],
                'Online_Security': [online_security],
                'Online_Backup': [online_backup],
                'Device_Protection_Plan': [device_protection],
                'Premium_Support': [premium_support],
                'Streaming_TV': [streaming_tv],
                'Streaming_Movies': [streaming_movies],
                'Streaming_Music': [streaming_music],
                'Unlimited_Data': [unlimited_data],
                'Contract': [contract],
                'Paperless_Billing': [paperless_billing],
                'Payment_Method': [payment_method],
                'Monthly_Charge': [monthly_charge],
                'Total_Charges': [total_charges],
                'Total_Refunds': [total_refunds],
                'Total_Extra_Data_Charges': [extra_data_charges],
                'Total_Long_Distance_Charges': [long_distance_charges],
                'Total_Revenue': [total_revenue]
            })
            
            # Preprocess input data
            for col in input_data.select_dtypes(include=['object']).columns:
                if col in label_encoders:
                    try:
                        input_data[col] = label_encoders[col].transform(input_data[col])
                    except ValueError:
                        input_data[col] = 0  # Handle unseen categories
            
            # Scale the data
            input_scaled = scaler.transform(input_data)
            
            # Make prediction
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0]
            
            # Display results
            st.subheader("Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if prediction == 1:
                    st.error("HIGH CHURN RISK")
                    st.caption("Customer likely to churn")
                else:
                    st.success("LOW CHURN RISK")
                    st.caption("Customer likely to stay")
            
            with col2:
                churn_prob = probability[1] * 100
                st.metric("Churn Probability", f"{churn_prob:.1f}%")
                st.caption("Model confidence")
            
            with col3:
                retention_prob = probability[0] * 100
                st.metric("Retention Probability", f"{retention_prob:.1f}%")
                st.caption("Likelihood to stay")
            
            # Probability gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = churn_prob,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Churn Probability"},
                delta = {'reference': 50},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 25], 'color': "lightgreen"},
                        {'range': [25, 50], 'color': "yellow"},
                        {'range': [50, 75], 'color': "orange"},
                        {'range': [75, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 1,
                        'value': 90
                    }
                }
            ))
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)   
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Please ensure the model files are available in the application directory.")




        
        


if __name__ == "__main__":
    main()
