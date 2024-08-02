import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans

# Title of the app
st.title('No Code Machine Learning')

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# Initialize dataset variable
if 'data' not in st.session_state:
    st.session_state.data = None

if uploaded_file is not None:
    # Read the CSV file
    data = pd.read_csv(uploaded_file)
    st.session_state.data = data.copy()
    
    # Display the columns
    st.subheader("Columns in the CSV file:")
    st.write(data.columns)
    
    # Display the first few rows of the dataset
    st.subheader("First few rows of the dataset:")
    st.write(data.head())
    
    # Button container for operations
    st.sidebar.header("Pre Processing")
    
    # Button to show number of null values in each column
    if st.sidebar.button('Show Null Values per Column'):
        if st.session_state.data is not None:
            st.subheader("Number of null values in each column:")
            st.write(st.session_state.data.isnull().sum())
        else:
            st.write("Please upload a CSV file first.")
    
    # Button to remove outliers and replace with column mean
    if st.sidebar.button('Remove Outliers'):
        if st.session_state.data is not None:
            data_clean = st.session_state.data.copy()
            
            # Define function to remove outliers and replace with column mean
            def remove_outliers_replace_with_mean(df, col):
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df[col] = np.where((df[col] < lower_bound) | (df[col] > upper_bound), df[col].mean(), df[col])
                return df
            
            # Apply function to all numerical columns
            numerical_cols = data_clean.select_dtypes(include=['float64', 'int64']).columns
            for col in numerical_cols:
                data_clean = remove_outliers_replace_with_mean(data_clean, col)
            
            st.subheader("Dataset after removing outliers and replacing with column mean:")
            st.write(data_clean)
            
            # Update dataset to use cleaned data
            st.session_state.data = data_clean
            
            # Display null values after removing outliers
            st.subheader("Number of null values in each column after removing outliers:")
            st.write(st.session_state.data.isnull().sum())
            
        else:
            st.write("Please upload a CSV file first.")
    
    # Button to remove rows with null values
    if st.sidebar.button('Remove Null Values'):
        if st.session_state.data is not None:
            st.session_state.data = st.session_state.data.dropna()
            if st.session_state.data.empty:
                st.write("All rows were removed. The resulting dataset is empty.")
            else:
                st.subheader("Dataset after removing null values:")
                st.write(st.session_state.data)
                # Update null value display
                st.subheader("Number of null values in each column after removing null values:")
                st.write(st.session_state.data.isnull().sum())
        else:
            st.write("Please upload a CSV file first.")
    
    # Button to replace null values (numerical with mean, categorical with mode)
    if st.sidebar.button('Replace Null Values'):
        if st.session_state.data is not None:
            data_filled = st.session_state.data.copy()
            
            # Replace numerical columns' null values with mean
            for col in data_filled.select_dtypes(include=['float64', 'int64']).columns:
                data_filled[col].fillna(data_filled[col].mean(), inplace=True)
            
            # Replace categorical columns' null values with mode
            for col in data_filled.select_dtypes(include=['object']).columns:
                data_filled[col].fillna(data_filled[col].mode()[0], inplace=True)
            
            st.session_state.data = data_filled
            st.subheader("Dataset after replacing null values:")
            st.write(st.session_state.data)
            # Update null value display
            st.subheader("Number of null values in each column after replacing null values:")
            st.write(st.session_state.data.isnull().sum())
        else:
            st.write("Please upload a CSV file first.")
    if st.session_state.data is not None:
        st.sidebar.subheader("Label Encoder")
    # Label Encode Categorical Values
    def label_encode_data(data):
        label_encoders = {}
        for column in data.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            data[column] = le.fit_transform(data[column].astype(str))
            label_encoders[column] = le
        return data, label_encoders

    if st.sidebar.button('Label Encode Categorical Values'):
        if st.session_state.data is not None:
            st.session_state.data, label_encoders = label_encode_data(st.session_state.data)
            st.subheader('Categorical values have been label encoded.')
            st.write(st.session_state.data.head())
        else:
            st.write("Please upload a CSV file first.")
    
    # Selection of Dependent and Independent Columns
    if st.session_state.data is not None:
        st.sidebar.subheader("Linear Regression")
        dependent_column = st.sidebar.selectbox('Select the dependent column (target)', options=st.session_state.data.columns)
        independent_columns = st.sidebar.multiselect('Select the independent columns (features)', options=st.session_state.data.columns)
    else:
        dependent_column = None
        independent_columns = []
    
    # Specify the Ratio for Train-Test Split
    test_size = st.sidebar.slider('Select the test size ratio', 0.1, 0.5, 0.2)
    
    # Apply Linear Regression
    if st.sidebar.button('Apply Linear Regression'):
        if st.session_state.data is not None and dependent_column and independent_columns:
            X = st.session_state.data[independent_columns]
            y = st.session_state.data[dependent_column]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            
            coefficients = model.coef_
            mse = mean_squared_error(y_test, y_pred)
            
            st.subheader('Linear Regression Results')
            st.write('Coefficient values:', coefficients)
            st.write('Mean Squared Error (MSE):', mse)
        else:
            st.write("Please ensure you have selected both the dependent and independent columns.")
    
    # K-Means Clustering Parameters
    if st.session_state.data is not None:
        st.sidebar.subheader("K-mean Clustering")
        num_clusters = st.sidebar.number_input('Enter number of clusters:', min_value=1, max_value=20, value=3)
        clustering_columns = st.sidebar.multiselect('Select columns for clustering:', options=st.session_state.data.columns)
    
    # Apply K-Means Clustering
    if st.sidebar.button('Apply K-Means Clustering'):
        if st.session_state.data is not None and clustering_columns:
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            st.session_state.data['Cluster'] = kmeans.fit_predict(st.session_state.data[clustering_columns])
            
            st.subheader('Dataset with K-Means Clustering Applied')
            st.write(st.session_state.data)
        else:
            st.write("Please ensure you have selected columns for clustering.")



    if st.session_state.data is not None:
        st.sidebar.subheader("Plotting")
        st.session_state.visualization_type = st.sidebar.selectbox("Select Visualization Type", ["Scatter Plot", "Bar Plot"])
        st.session_state.selected_column = st.sidebar.selectbox("Select Column", st.session_state.data.columns.tolist())

        if st.sidebar.button("Generate Plot"):
            if st.session_state.visualization_type and st.session_state.selected_column:
                if st.session_state.visualization_type == "Bar Plot":
                    fig, ax = plt.subplots(figsize=(10, 6))
                    value_counts = st.session_state.data[st.session_state.selected_column].value_counts()
                    ax.bar(value_counts.index, value_counts.values)

                    plt.xlabel(st.session_state.selected_column)
                    plt.ylabel('Count')
                    plt.title(f'Bar Plot for {st.session_state.selected_column}')

                    st.pyplot(fig)
                elif st.session_state.visualization_type == "Scatter Plot":
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.scatter(st.session_state.data[st.session_state.selected_column], range(len(st.session_state.data)))

                    plt.xlabel(st.session_state.selected_column)
                    plt.ylabel('Index')
                    plt.title(f'Scatter Plot for {st.session_state.selected_column}')

                    st.pyplot(fig)
            else:
                st.write("Please select both a visualization type and a column.")