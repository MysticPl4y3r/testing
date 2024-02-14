import streamlit as st
import pandas as pd
import numpy as np
from prediction import predict
from pathlib import Path
from model import train
from math import ceil
from visualize import visualization

st.set_page_config(page_title="Automated Machine Learning", layout="wide", initial_sidebar_state="expanded")
#STYLE
# CSS styles
st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
        animation: slideInLeft 0.5s forwards;
    }
    .sidebar .sidebar-content .block-container {
        padding: 2rem;
    }
    .sidebar .sidebar-content .stRadio > label > div:first-child {
        display: none;
    }
    .sidebar .sidebar-content .stRadio > label {
        padding: 10px 20px;
        background-color: #cbd2d9;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .sidebar .sidebar-content .stRadio > label:hover {
        background-color: #aebcc6;
        cursor: pointer;
    }
    .main .block-container {
        padding: 2rem;
        animation: fadeIn 1s forwards;
    }
    .main .block-container .stButton>button {
        background-color: #4caf50;
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        border: none;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }
    .main .block-container .stButton>button:hover {
        background-color: #45a049;
    }
    .main .block-container .stTextInput>div>div>input {
        border-radius: 5px;
        border: 1px solid #ccc;
        padding: 10px 15px;
        transition: border-color 0.3s ease;
    }
    .main .block-container .stTextInput>div>div>input:focus {
        border-color: #4caf50;
    }
    @keyframes slideInLeft {
        from {
            transform: translateX(-100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    @keyframes fadeIn {
        from {
            opacity: 0;
        }
        to {
            opacity: 1;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Add animations
st.markdown(
    """
    <style>
    @keyframes fadeIn {
        0% {
            opacity: 0;
        }
        100% {
            opacity: 1;
        }
    }
    .animated {
        animation-duration: 1s;
        animation-fill-mode: both;
    }
    .fadeIn {
        animation-name: fadeIn;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def downloadbutton(model, save_file):
    with open(model, 'rb') as f:
        mlmodel = f.read()
    st.download_button("Download Model", data=mlmodel, file_name=save_file)

def text_hl(text, color='blue', font_weight='bold', font_size='16px', padding='5px'):
    return f'<span style="color:{color}; font-weight:{font_weight}; font-size:{font_size}; padding:{padding}">{text}</span>'

def Upload_data():
    st.header("Upload Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    if uploaded_file:
        save_folder = "./data"
        save_path = Path(save_folder, uploaded_file.name)
        with open(save_path, 'wb') as w:
            w.write(uploaded_file.getvalue())
        if save_path.exists():
            st.success(f'File {uploaded_file.name} is successfully saved!')
        data = pd.read_csv(save_path)
        return data

def View_data(data):
    if st.button("View Data"):
        st.write(data.head())

def Visualize():
    target = st.selectbox("Select the target variable:", list(st.session_state.data.columns))
    st.write("Target Variable selected:", target)
    if st.button("Perform Data Visualisation"):
        visualization(st.session_state.data, target)

def Training():
    target = st.selectbox("Select the target variable:", list(st.session_state.data.columns))
    st.write("Target Variable selected:", target)
    if st.button("Train"):
        if 'target' in locals():
            train(target, st.session_state.data)
            st.success(f'Model trained successfully')
        else:
            st.error("please select target value")
    X_feat = list(st.session_state.data.drop(columns=[target]).columns)
    st.session_state.X_feat=X_feat
    st.text(f"Features selected for training:{X_feat}")
    return X_feat

def download_model():
    try:
        downloadbutton("./model/best_model.sav", "best_model.sav")
    except:
        st.text("You have now uploaded the data. Click Train to train the model and download it.")

def predictions():
    st.header("Features")
    co1=[]
    col1, col2 = st.columns(2)
    with col1:
        for i in range(0, ceil(len(st.session_state.X_feat) / 2)):
            co1.append(st.number_input(f"Enter value for {st.session_state.X_feat[i]}"))
    with col2:
        for i in range(len(st.session_state.X_feat) // 2, len(st.session_state.X_feat)):
            co1.append(st.number_input(f"Enter value for {st.session_state.X_feat[i]}"))
    if st.button("Predict"):
        res = predict(np.array([co1]))
        styled_pred = text_hl(res[0], color="yellow", font_weight='700', font_size='20px', padding='10px')
        st.markdown(f"Prediction Result:{styled_pred}", unsafe_allow_html=True)

# Build App
st.title("Automated ML")
st.markdown("The service to automate the ml training.")

st.sidebar.title("Menu")
app = st.sidebar.radio("Choose", options=["Load Data", "Data Visualisation", "Model Training", "Best Model", "Prediction"])

if 'data' not in st.session_state:
    st.session_state.data = None

if "X_feat" not in st.session_state:
    st.session_state.X_feat=None

if app == "Load Data":
    st.session_state.data = Upload_data()
    if st.session_state.data is not None:
        View_data(st.session_state.data)

if app == "Data Visualisation":
    if st.session_state.data is not None:
        Visualize()
    else:
        st.warning("Load the data first")

if app == "Model Training":
    if st.session_state.data is not None:
        st.session_state.X_feat=Training()
    else:
        st.warning("Load the data first")

if app == "Best Model":
    download_model()

if app == "Prediction":
    if st.session_state.data is not None:
        predictions()
    else:
        st.warning("Load the data first")