import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def main():
    st.title("No Code Machine Learning")
   
    # Unique key for file uploader
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"], key="file_uploader")

    # Initialize session state for the DataFrame if not already initialized
    if 'df' not in st.session_state:
        st.session_state.df = None

    if uploaded_file is not None:
        st.session_state.df = pd.read_csv(uploaded_file)
        st.write("### Dataset:")
        st.dataframe(st.session_state.df)
        st.write("### Column Names:")
        st.write(st.session_state.df.columns.tolist())

    # Ensure DataFrame is not None before proceeding
    if st.session_state.df is not None:
        # Sidebar for preprocessing
        st.sidebar.header("Preprocessing")
        
        if st.sidebar.button("Preprocessing", key="preprocessing_button"):
            st.session_state.show_preprocessing = not st.session_state.get('show_preprocessing', False)

        if st.session_state.get('show_preprocessing', False):
            if st.sidebar.button("Show Null Values", key="show_null_values"):
                st.write("### Null Values in Dataset")
                st.write(st.session_state.df.isnull().sum())

            if st.sidebar.button("Remove Null Values", key="remove_null_values"):
                st.session_state.df.dropna(inplace=True)
                st.write("### Dataset after removing null values:")
                st.dataframe(st.session_state.df)

            if st.sidebar.button("Replace Null Values (Categorical with Mode, Numerical with Mean)", key="replace_null_values"):
                for col in st.session_state.df.columns:
                    if st.session_state.df[col].dtype == 'object':
                        st.session_state.df[col].fillna(st.session_state.df[col].mode()[0], inplace=True)
                    else:
                        st.session_state.df[col].fillna(st.session_state.df[col].mean(), inplace=True)
                st.write("### Dataset after replacing null values:")
                st.dataframe(st.session_state.df)

            if st.sidebar.button("Remove Outliers", key="remove_outliers"):
                numerical_cols = st.session_state.df.select_dtypes(include=np.number).columns
                Q1 = st.session_state.df[numerical_cols].quantile(0.25)
                Q3 = st.session_state.df[numerical_cols].quantile(0.75)
                IQR = Q3 - Q1
                st.session_state.df = st.session_state.df[~((st.session_state.df[numerical_cols] < (Q1 - 1.5 * IQR)) | (st.session_state.df[numerical_cols] > (Q3 + 1.5 * IQR))).any(axis=1)]
                st.write("### Dataset after removing outliers:")
                st.dataframe(st.session_state.df)

        # Label Encoder
        st.sidebar.header("Label Encoder")

        if st.sidebar.button("Label Encoder", key="label_encoder"):
            st.session_state.show_label_encoder = not st.session_state.get('show_label_encoder', False)

        if st.session_state.get('show_label_encoder', False):
            categorical_cols = st.session_state.df.select_dtypes(include='object').columns
            if len(categorical_cols) == 0:
                st.write("### No categorical columns to encode.")
            else:
                cols_to_encode = st.sidebar.multiselect("Select Categorical Columns to Encode", categorical_cols, key="categorical_columns")
                if st.sidebar.button("Encode", key="encode_button"):
                    le = LabelEncoder()
                    for col in cols_to_encode:
                        st.session_state.df[col] = le.fit_transform(st.session_state.df[col])
                    st.write("### Dataset after Label Encoding:")
                    st.dataframe(st.session_state.df)

        # Clustering
        st.sidebar.header("Clustering")

        if st.sidebar.button("Clustering", key="clustering_button"):
            st.session_state.show_clustering = not st.session_state.get('show_clustering', False)

        if st.session_state.get('show_clustering', False):
            numerical_cols = st.session_state.df.select_dtypes(include=np.number).columns
            if len(numerical_cols) < 2:
                st.write("### Need at least two numerical columns for clustering.")
            else:
                cols_to_cluster = st.sidebar.multiselect("Select Numerical Columns for Clustering", numerical_cols, key="cluster_columns")
                n_clusters = st.sidebar.slider("Select Number of Clusters", min_value=2, max_value=10, value=3, key="num_clusters")
                if st.sidebar.button("Apply K-Means Clustering", key="apply_clustering"):
                    if len(cols_to_cluster) >= 2:
                        kmeans = KMeans(n_clusters=n_clusters)
                        st.session_state.df['Cluster'] = kmeans.fit_predict(st.session_state.df[cols_to_cluster])
                        st.write(f"### K-Means Clustering with {n_clusters} Clusters:")
                        st.dataframe(st.session_state.df)
                    else:
                        st.write("### Please select at least two numerical columns for clustering.")

        # Visualization
        st.sidebar.header("Visualization")

        if st.sidebar.button("Visualization", key="visualization_button"):
            st.session_state.show_visualization = not st.session_state.get('show_visualization', False)

        if st.session_state.get('show_visualization', False):
            plot_type = st.sidebar.selectbox("Select Plot Type", ["Bar Plot", "Scatter Plot", "Line Graph", "Pie Chart"], key="plot_type")
            numerical_cols = st.session_state.df.select_dtypes(include=np.number).columns
            selected_col = st.sidebar.selectbox("Select Column", numerical_cols, key="plot_column")

            if st.sidebar.button("Generate Plot", key="generate_plot"):
                st.write(f"### {plot_type}")

                fig, ax = plt.subplots()
                if plot_type == "Bar Plot":
                    st.session_state.df[selected_col].value_counts().plot(kind='bar', ax=ax)

                elif plot_type == "Scatter Plot":
                    if len(numerical_cols) > 1:  # Use the first numerical column as x-axis
                        st.session_state.df.plot(kind='scatter', x=numerical_cols[0], y=selected_col, ax=ax)
                    else:
                        st.write("Not enough numerical columns for scatter plot.")

                elif plot_type == "Line Graph":
                    st.session_state.df[selected_col].plot(kind='line', ax=ax)

                elif plot_type == "Pie Chart":
                    st.session_state.df[selected_col].value_counts().plot(kind='pie', ax=ax, autopct='%1.1f%%')

                st.pyplot(fig)

                # Save plot to an in-memory buffer
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                
                # Download plot image
                st.download_button(
                    label="Download Plot Image",
                    data=buf,
                    file_name="plot.png",
                    mime="image/png"
                )

        # Model Section
        st.sidebar.header("Model")

        if st.sidebar.button("Model", key="model_button"):
            st.session_state.show_model = not st.session_state.get('show_model', False)

        if st.session_state.get('show_model', False):
            # Linear Regression
            st.sidebar.header("Linear Regression")

            if st.sidebar.button("Linear Regression", key="linear_regression"):
                st.session_state.show_linear_regression = not st.session_state.get('show_linear_regression', False)

            if st.session_state.get('show_linear_regression', False):
                numerical_cols = st.session_state.df.select_dtypes(include=np.number).columns
                dependent_col = st.sidebar.selectbox("Select Dependent Column", numerical_cols, key="dependent_column")
                independent_cols = st.sidebar.multiselect("Select Independent Columns", numerical_cols, key="independent_columns")
                test_size = st.sidebar.slider("Select Train-Test Split Ratio", min_value=0.1, max_value=0.9, value=0.2, key="test_size")

                if st.sidebar.button("Apply Linear Regression", key="apply_linear_regression"):
                    if dependent_col and independent_cols:
                        X = st.session_state.df[independent_cols]
                        y = st.session_state.df[dependent_col]

                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                        model = LinearRegression()
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)

                        st.write("### Linear Regression Results")
                        st.write("Coefficient Values:", model.coef_)
                        st.write("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
                    else:
                        st.write("Please select both dependent and independent columns.")

    # Download updated dataset
    st.write("## Download Updated Dataset")
    if st.session_state.df is not None:
        updated_file = uploaded_file.name
        csv = st.session_state.df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=updated_file,
            mime='text/csv'
        )
    st.write("NOTE: You can only perform single operation to each upload of Data.")    

if __name__ == "__main__":
    main()
