import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Streamlit UI
st.title("Titanic Survival Prediction")

# File Upload
uploaded_file = st.file_uploader("Upload Titanic Dataset", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Raw Dataset")
    st.write(df.head())

    # Preprocessing
    if 'Age' in df.columns:
        df['Age'] = df['Age'].fillna(df['Age'].median())

    if 'Embarked' in df.columns:
        df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    if 'Cabin' in df.columns:
        df = df.drop('Cabin', axis=1)

    df = df.dropna()

    if 'Name' in df.columns:
        df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
        df['Title'] = df['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 
                                           'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
        df['Title'] = df['Title'].replace('Mlle', 'Miss')
        df['Title'] = df['Title'].replace('Ms', 'Miss')
        df['Title'] = df['Title'].replace('Mme', 'Mrs')

    drop_cols = ['PassengerId', 'Name', 'Ticket']
    for col in drop_cols:
        if col in df.columns:
            df = df.drop(col, axis=1)

    df = pd.get_dummies(df, drop_first=True)

    if 'Survived' in df.columns:
        X = df.drop('Survived', axis=1)
        y = df['Survived']
    else:
        st.error("Target column 'Survived' not found!")
        st.stop()

    # Feature Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Splitting Data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Model Training
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Model Evaluation
    st.subheader("Model Evaluation")
    st.write("**Accuracy:**", accuracy_score(y_test, y_pred))
    st.text("Classification Report")
    st.text(classification_report(y_test, y_pred))

    # Visualization: Survival Distribution
    st.subheader("Survival Distribution")
    fig1, ax1 = plt.subplots()
    sns.countplot(x=y, ax=ax1)
    st.pyplot(fig1)

    # Visualization: Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig2, ax2 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted No', 'Predicted Yes'],
                yticklabels=['Actual No', 'Actual Yes'], ax=ax2)
    ax2.set_ylabel('Actual')
    ax2.set_xlabel('Predicted')
    st.pyplot(fig2)
