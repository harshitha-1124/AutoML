import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from pycaret.classification import setup as setup_clf, compare_models as compare_clfs, evaluate_model as evaluate_clfs, pull as pull_clf
from pycaret.regression import setup as setup_reg, compare_models as compare_reg, evaluate_model as evaluate_reg, pull as pull_reg

st.set_page_config(page_title="SwiftML", layout="wide")
st.title("SwiftML - Accelerating ML Journeys")

# Sidebar with blank option
section = st.sidebar.radio("Choose Section", ["-- Select Section --", "Data Preprocessing", "Data Ingestion", "Model Selection", "Clustering (Unsupervised → Supervised)"])

# Blank landing page
if section == "-- Select Section --":
   

    st.markdown("""
    ### 🧠 <span style='font-size:26px; font-weight:bold; color:#ff4b4b;'>ALL-IN-ONE ASSISTANT</span>  
    ### ⚡ <span style='font-size:22px; font-weight:bold; color:#ffa500;'>– NO CODE, JUST RESULTS! –</span>

    - 📌 <span style='font-size:20px; font-weight:bold;'>Auto-select the best ML models in seconds</span>  
    - 🧹 <span style='font-size:20px; font-weight:bold;'>Clean and preprocess your data with ease</span>  
    - 📊 <span style='font-size:20px; font-weight:bold;'>Visualize, interpret, and export insights effortlessly</span>  

    ---

    💡 Use the sidebar to navigate through powerful tools designed to accelerate your ML journey.
    
    🎉 It’s fast, smart, and made just for you!
    """, unsafe_allow_html=True)


# === Preprocessing Section ===
elif section == "Data Preprocessing":
    class DatasetPreprocessor:
        def __init__(self, output_dir="preprocessed_data"):
            self.output_dir = output_dir
            os.makedirs(output_dir, exist_ok=True)

        def load_dataset(self, path: str) -> pd.DataFrame:
            if path.endswith('.csv'):
                return pd.read_csv(path)
            elif path.endswith('.json'):
                return pd.read_json(path)
            elif path.endswith(('.xlsx', '.xls')):
                return pd.read_excel(path)
            else:
                raise ValueError(f"Unsupported format: {path}")

        def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            strategies = {
                'mean': lambda x: x.fillna(x.mean(numeric_only=True)),
                'median': lambda x: x.fillna(x.median(numeric_only=True)),
                'mode': lambda x: x.fillna(x.mode().iloc[0]),
                'interpolate': lambda x: x.interpolate()
            }
            df[numeric_cols] = strategies[strategy](df[numeric_cols])
            string_cols = df.select_dtypes(include=['object', 'string']).columns
            for col in string_cols:
                if df[col].isnull().any():
                    mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else 'unknown'
                    df[col] = df[col].fillna(mode_val)
            for col in df.columns:
                if 'gender' in col.lower():
                    df[col] = df[col].fillna('unknown')
            return df

        def clean_columns(self, df: pd.DataFrame) -> pd.DataFrame:
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                    df[col] = df[col].clip(lower=0).round().astype(int)
                elif pd.api.types.is_string_dtype(df[col]):
                    df[col] = df[col].astype(str).str.strip().str.lower().replace({'nan': 'unknown', '': 'unknown'}).fillna('unknown')
                    if 'gender' in col.lower():
                        df[col] = df[col].replace({
                            'm': 'male', 'male': 'male',
                            'f': 'female', 'female': 'female'
                        }).fillna('unknown')
            return df

        def scale_features(self, df: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
            scaler = StandardScaler() if method == 'standard' else MinMaxScaler()
            numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
            if not numerical_cols.empty:
                df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
            return df

        def split_dataset(self, df: pd.DataFrame, test_size: float = 0.2):
            return train_test_split(df, test_size=test_size, random_state=42)

    uploaded_file = st.file_uploader("📁 Upload your dataset", type=["csv", "json", "xlsx", "xls"])
    missing_strategy = st.selectbox("🧹 Missing value strategy", ['mean', 'median', 'mode', 'interpolate'])
    scaling_method = st.selectbox("📏 Scaling method", ['standard', 'minmax'])
    test_size = st.slider("📊 Test size", 0.1, 0.5, 0.2, 0.05)
    output_prefix = st.text_input("Filename prefix", value="preprocessed")
    output_dir = st.text_input("Output directory", value="preprocessed_data")

    if uploaded_file:
        with st.spinner("Processing..."):
            suffix = uploaded_file.name.split('.')[-1]
            temp_path = f"temp_uploaded.{suffix}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            try:
                preprocessor = DatasetPreprocessor(output_dir=output_dir)
                df = preprocessor.load_dataset(temp_path)
                df = preprocessor.handle_missing_values(df, missing_strategy)
                df = preprocessor.clean_columns(df)
                df = preprocessor.scale_features(df, scaling_method)
                train_df, test_df = preprocessor.split_dataset(df, test_size)

                os.makedirs(output_dir, exist_ok=True)
                train_path = os.path.join(output_dir, f"{output_prefix}_train.csv")
                test_path = os.path.join(output_dir, f"{output_prefix}_test.csv")
                train_df.to_csv(train_path, index=False)
                test_df.to_csv(test_path, index=False)

                st.success("Preprocessing complete!")
                st.download_button("⬇️ Download Training Set", data=train_df.to_csv(index=False), file_name=f"{output_prefix}_train.csv")
                st.download_button("⬇️ Download Testing Set", data=test_df.to_csv(index=False), file_name=f"{output_prefix}_test.csv")
                st.dataframe(df.head())
            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                os.remove(temp_path)

# === Data Ingestion Section ===
elif section == "Data Ingestion":
    file = st.file_uploader("📥 Upload your file")
    if file:
        df = pd.read_csv(file)
        st.dataframe(df)
        df.to_csv("sourcefile.csv", index=False)
        st.success("Data saved to `sourcefile.csv`")

# === Model Selection Section ===
elif section == "Model Selection":
    if not os.path.exists("sourcefile.csv"):
        st.warning("Please upload and ingest a dataset first.")
    else:
        file = pd.read_csv("sourcefile.csv")
        task_type = st.selectbox("Select Task", ['Regression', 'Classification'])
        target = st.selectbox("Select Target Column", file.columns)
        if st.button("Run Modeling"):
            if task_type == "Regression":
                setup_df = setup_reg(file, target=target)
                st.subheader("Setup Summary")
                st.dataframe(pull_reg())
                best_model = compare_reg()
                st.subheader("Model Comparison")
                st.dataframe(pull_reg())
                evaluate_reg(best_model)
            else:
                setup_df = setup_clf(file, target=target)
                st.subheader("Setup Summary")
                st.dataframe(pull_clf())
                best_model = compare_clfs()
                st.subheader("Model Comparison")
                st.dataframe(pull_clf())
                evaluate_clfs(best_model)

# === Clustering Section ===
elif section == "Clustering (Unsupervised → Supervised)":
    def convert_to_supervised(data, n_clusters):
        data_array = data.values
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(data_array)
        supervised_data = data.copy()
        supervised_data['cluster_label'] = cluster_labels
        return supervised_data

    st.markdown("### Convert Unsupervised Data to Supervised using KMeans")
    st.markdown("Upload your CSV dataset and choose the number of clusters.")

    uploaded_cluster_file = st.file_uploader("📂 Upload CSV for Clustering", type=["csv"])
    if uploaded_cluster_file:
        try:
            df_cluster = pd.read_csv(uploaded_cluster_file)
            st.success(f"Dataset loaded successfully! Shape: {df_cluster.shape}")
            st.write(df_cluster.head())

            n_clusters = st.number_input("🔢 Number of Clusters", min_value=1, max_value=20, value=3, step=1)

            if st.button("Run Clustering"):
                with st.spinner("Clustering in progress..."):
                    result_df = convert_to_supervised(df_cluster, n_clusters)
                    st.success("Clustering complete!")
                    st.dataframe(result_df.head())
                    st.download_button(
                        label="⬇️ Download Clustered Data",
                        data=result_df.to_csv(index=False).encode('utf-8'),
                        file_name="clustered_data.csv",
                        mime="text/csv"
                    )
        except Exception as e:
            st.error(f"Error processing file: {e}")




