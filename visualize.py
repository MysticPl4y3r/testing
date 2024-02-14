import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import plotly.express as px
import streamlit as st
import copy
from model import preprocess

# Function to center-align text and headings
def centered_text(text):
    return f"<h2 style='text-align: center;'>{text}</h2>"

# Set page title and icon
#st.set_page_config(page_title="df Visualization", page_icon=":bar_chart:")


def visualization(data,target):
    data=preprocess(data,target)
    df=copy.deepcopy(data)
    if df[target].dtype=="object":
        le=LabelEncoder()
        df[target]=le.fit_transform(df[target])
    corr_matrix = df.corr()

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=.5, ax=ax)
    plt.title("Correlation Heatmap")
    st.pyplot(fig)

    highest_corr_features = corr_matrix[target].sort_values(ascending=False).index[1:6]  # Exclude the target variable itself


    for feature in highest_corr_features:
        st.markdown(centered_text(f"Visualization for {feature}"),unsafe_allow_html=True)
        if df[feature].dtype in ['int64', 'float64']:
            st.write("Histogram")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.histplot(df[feature], kde=True, ax=ax)
            ax.set_xlabel(feature)
            ax.set_ylabel("Frequency")
            ax.set_title("Histogram of " + feature)
            st.pyplot(fig)

            st.write("Scatter Plot")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.scatterplot(df=df, x=feature, y=target, ax=ax)
            ax.set_xlabel(feature)
            ax.set_ylabel(target)
            ax.set_title("Scatter Plot of " + feature + " vs " + target)
            st.pyplot(fig)

            st.write("Line Plot")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.lineplot(df=df, x=feature, y=target, ax=ax)
            ax.set_xlabel(feature)
            ax.set_ylabel(target)
            ax.set_title("Line Plot of " + feature + " vs " + target)
            st.pyplot(fig)

            st.write("Area Plot")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.lineplot(df=df, x=feature, y=target, ax=ax, ci=None)
            sns.lineplot(df=df, x=feature, y=target, ax=ax, ci='sd')
            ax.set_xlabel(feature)
            ax.set_ylabel(target)
            ax.set_title("Area Plot of " + feature + " vs " + target)
            st.pyplot(fig)

        elif df[feature].dtype == 'object':
            st.write("Bar Plot")
            bar_plot = px.bar(df[feature].value_counts(), x=df[feature].value_counts().index, y=df[feature].value_counts().values)
            st.plotly_chart(bar_plot)

            st.write("Pie Chart")
            pie_chart = px.pie(df[feature].value_counts(), values=df[feature].value_counts().values, names=df[feature].value_counts().index)
            st.plotly_chart(pie_chart)
