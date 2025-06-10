import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# App title
st.title("🚢 Titanic Survival Prediction App")

# Load Titanic dataset directly from GitHub
url = "https://raw.githubusercontent.com/Omii2510/Titanic/refs/heads/main/Titanic-Dataset.csv"
df = pd.read_csv(url)
st.success("Titanic dataset loaded from GitHub.")

# Display raw data
st.subheader("🔍 Raw Dataset")
st.write(df.head())

# Data Preprocessing
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

# Drop unnecessary columns
drop_cols = ['PassengerId', 'Name', 'Ticket']
for col in drop_cols:
    if col in df.columns:
        df = df.drop(col, axis=1)

# Convert categorical variables to dummies
df = pd.get_dummies(df, drop_first=True)

# Separate features and target
if 'Survived' not in df.columns:
    st.error("Target column 'Survived' not found in dataset.")
    st.stop()

X = df.drop('Survived', axis=1)
y = df['Survived']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Model evaluation
st.subheader("📊 Model Evaluation")
st.write("**Accuracy:**", accuracy_score(y_test, y_pred))
st.text("Classification Report")
st.text(classification_report(y_test, y_pred))

# Plot survival distribution
st.subheader("🧍 Survival Distribution")
fig1, ax1 = plt.subplots()
sns.countplot(x=y, ax=ax1)
ax1.set_title("Survival Counts")
st.pyplot(fig1)

# Plot confusion matrix
st.subheader("🔁 Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig2, ax2 = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted No', 'Predicted Yes'],
            yticklabels=['Actual No', 'Actual Yes'], ax=ax2)
ax2.set_ylabel('Actual')
ax2.set_xlabel('Predicted')
st.pyplot(fig2)

