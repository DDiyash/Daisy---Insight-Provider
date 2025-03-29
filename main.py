import pandas as pd
import streamlit as st
import numpy as np
from data_cleaning import clean_data
from data_cleaning import visualizations

st.title("Daisy")
st.subheader("Your Insight Provider")


st.sidebar.title("Input")
uploaded_file = st.sidebar.file_uploader("Upload Data File (Only CSV File)")

if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'): #Upload file
        df = pd.read_csv(uploaded_file) #Checks extension
    else:
        st.error("Unsupported file format")
        st.stop()

    if df is not None:
        st.write("Raw Data Preview")
        st.dataframe(df)

        cleaned_data,numeric_columns,categorical_columns = clean_data(df)
        
        st.write("Cleaned Data Preview")
        st.dataframe(cleaned_data)#converts to dataframe

        st.sidebar.subheader("Select Columns for Visualization")

        selected_num_col = st.sidebar.selectbox("Choose a Numeric Column", options=["All"]+list(numeric_columns))
        selected_cat_col = st.sidebar.selectbox("Choose a Categorical Column", options=["All"]+list(categorical_columns))
        viz_type = st.sidebar.radio("Select Graph Type", ["Histogram", "Boxplot", "Scatterplot", "Countplot", "Pie Chart","Heat Map"])
        visualizations(cleaned_data,viz_type,selected_num_col,selected_cat_col)
        
else:
    st.warning("Please upload a CSV file to proceed.")


