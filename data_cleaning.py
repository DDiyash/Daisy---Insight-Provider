import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import io



def clean_data(df):
    #eda
    # Capture df.info() output in a string buffer
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()  # Get the string representation

    # Display Data Info in Streamlit
    st.write("### Data Info:")
    st.text(info_str)  # Show `df.info()` properly in Streamlit
    

    # Show descriptive statistics
    st.write("### Summary Statistics:")
    st.write(df.describe())
    #drop duplicates
    df = df.drop_duplicates()
    #handling null values
    df = df.dropna(axis=1,thresh = int(0.6*len(df)))
    numeric_columns = df.select_dtypes(include=['float64','int64']).columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in numeric_columns:
        df[col] = df[col].fillna(df[col].mean())

    for col in categorical_columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    #Converting dtypes
    for col in categorical_columns:
        df[col] = df[col].astype('category')

    #datetime conversion
    for col in categorical_columns:
        sample_values = df[col].dropna().astype(str).sample(min(10, len(df)))  # Sample up to 10 non-null values
    try:
        inferred_dates = pd.to_datetime(sample_values, infer_datetime_format=True, errors='coerce')
        if inferred_dates.notna().sum() > 0.7 * len(inferred_dates):  # If 70%+ of sampled values can be converted
            df[col] = pd.to_datetime(df[col], infer_datetime_format=True, errors='coerce')
            print(f"Converted {col} to datetime using inferred format.")
    except Exception as e:
        print(f"Could not convert {col} to datetime: {e}")

    #handling outliers
    
    for col in numeric_columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    print("Outliers removed")

    return df,numeric_columns,categorical_columns
    
def visualizations(df, viz_type, selected_num_col, selected_cat_col):
    """Function to generate visualizations based on user selection."""
    
    if viz_type == "Heat Map":
        st.write("### Correlation Heat Map")
        plt.figure(figsize=(10, 6))
        sns.heatmap(df.select_dtypes(include=['float64', 'int64']).corr(), annot=True, cmap='coolwarm', fmt='.2f')
        st.pyplot(plt)
        st.markdown("""
            ** Correlation Heat Map**
            - Shows the relationships between numeric features.
            - Values close to **+1** indicate a strong **positive correlation** (features increase together).
            - Values close to **-1** indicate a strong **negative correlation** (one feature increases while the other decreases).
            - Helps identify **multicollinearity**, which can affect model performance.
            """)


    elif viz_type == "Histogram":
        st.write("### Histogram")
        fig, ax = plt.subplots(figsize=(8, 5))
        
        if selected_num_col == "All":
            for col in df.select_dtypes(include=['float64', 'int64']).columns:
                sns.histplot(df[col], kde=True, ax=ax, bins=30, label=col)
            ax.legend()
        else:
            sns.histplot(df[selected_num_col], kde=True, ax=ax, bins=30)
        
        st.pyplot(fig)
        st.markdown("""
            ** Histogram**
            - Shows the **distribution** of numeric data.
            - Helps identify **skewness** (left/right) and **outliers**.
            - The presence of **multiple peaks** might suggest **multiple subgroups** in data.
            - The **KDE curve** shows the estimated probability density.
            """)


    elif viz_type == "Boxplot":
        st.write("### Boxplot")
        fig, ax = plt.subplots(figsize=(8, 6))
        
        if selected_num_col == "All":
            sns.boxplot(data=df.select_dtypes(include=['float64', 'int64']), ax=ax)
        else:
            sns.boxplot(y=df[selected_num_col], ax=ax)
        
        st.pyplot(fig)
        st.markdown("""
            ** Boxplot**
            - Visualizes **data distribution** and **outliers**.
            - The **box** represents the **interquartile range (IQR)** (middle 50% of the data).
            - The **whiskers** show the **minimum and maximum** values within 1.5 × IQR.
            - **Dots outside whiskers** indicate potential **outliers**.
            - Useful for detecting **skewness** in numeric features.
            """)


    elif viz_type == "Scatterplot":
        st.sidebar.subheader("Select Another Numeric Column for Scatterplot")
        scatter_x = st.sidebar.selectbox("X-axis", options=df.select_dtypes(include=['float64', 'int64']).columns)
        scatter_y = st.sidebar.selectbox("Y-axis", options=df.select_dtypes(include=['float64', 'int64']).columns)
        st.write(f"### Scatterplot: {scatter_x} vs {scatter_y}")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(x=df[scatter_x], y=df[scatter_y], ax=ax)
        st.pyplot(fig)
        st.markdown("""
            ** Scatterplot**
            - Displays the **relationship** between two numeric variables.
            - Helps identify:
            - **Trends** (linear, non-linear).
            - **Clusters** (groups in data).
            - **Outliers** (points deviating from the pattern).
            - If points follow an **upward trend**, there is a **positive correlation**.
            - If points follow a **downward trend**, there is a **negative correlation**.
            """)


    elif viz_type == "Countplot":
        st.write("### Countplot")
        fig, ax = plt.subplots(figsize=(8, 6))
        
        if selected_cat_col == "All":
            for col in df.select_dtypes(include=['category']).columns:
                sns.countplot(y=df[col], order=df[col].value_counts().index, ax=ax)
                st.pyplot(fig)
        else:
            sns.countplot(y=df[selected_cat_col], order=df[selected_cat_col].value_counts().index, ax=ax)
            st.pyplot(fig)

            st.markdown("""
                ** Countplot**
                - Shows the **frequency** of categorical values.
                - Helps detect **imbalances** in categorical data.
                - A higher count for a particular category might indicate **dominance**.
                - Useful for analyzing **class distributions** before applying models.
                """)


    elif viz_type == "Pie Chart":
        st.write("### Pie Chart")
        
        if selected_cat_col == "All":
            st.error("Pie chart only works with one categorical column at a time.")
        else:
            fig, ax = plt.subplots(figsize=(6, 6))
            df[selected_cat_col].value_counts().plot.pie(autopct='%1.1f%%', ax=ax, cmap="Pastel1")
            st.pyplot(fig)
            st.markdown("""
                ** Pie Chart**
                - Shows the **proportion** of each category in a dataset.
                - Helps compare **relative sizes** of different groups.
                - Best used for **low-cardinality categorical data** (few unique values).
                - Overuse of pie charts can make it **harder to compare** categories.
                """)


    else:
        st.error("Invalid visualization type selected!")


