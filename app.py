import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

# Load dataset
df = pd.read_csv("Iris.csv")
df.drop("Id", axis='columns', inplace=True)

# Split Data
X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = df['Species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train KNN Model
knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(X_train, y_train)

# Train Decision Tree Model (optional)
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Define prediction function
def predict_species(sepal_length, sepal_width, petal_length, petal_width):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = knn.predict(input_data)[0]
    return prediction

# Streamlit UI
st.title("🌸 Iris Species Classifier")
st.write("Enter the flower measurements to predict its species.")

# Input sliders
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

# Predict button
if st.button("Predict"):
    species = predict_species(sepal_length, sepal_width, petal_length, petal_width)
    st.success(f"Predicted Species: **{species}** 🌿")

# Show dataset statistics and correlation heatmap
st.subheader("📊 Dataset Statistics")
st.write(df.describe())

st.subheader("🔍 Feature Correlation")
fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(df.select_dtypes(include=['number']).corr(), annot=True, cmap='RdYlGn', ax=ax)
st.pyplot(fig)

# Model accuracy
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.subheader("📈 Model Accuracy")
st.write(f"KNN Model Accuracy: **{accuracy * 100:.2f}%**")

