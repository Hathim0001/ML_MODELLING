import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

st.set_page_config(layout="wide")
st.title("Heart Disease Prediction")

df = pd.read_csv('heart.csv')

st.sidebar.header("User Input Parameters")
selected_algorithm = st.sidebar.selectbox("Choose an Algorithm", ["KNN", "Decision Tree", "Random Forest", "SVM", "Gradient Boosting", "AdaBoost"])
knn_neighbors = st.sidebar.slider("K (Neighbors for KNN)", 1, 20, 12)
decision_depth = st.sidebar.slider("Depth for Decision Tree", 1, 10, 3)
forest_estimators = st.sidebar.slider("Estimators for Random Forest", 10, 100, 90, step=10)
svm_kernel = st.sidebar.selectbox("Kernel for SVM", ["linear", "rbf", "poly", "sigmoid"])
gb_estimators = st.sidebar.slider("Estimators for Gradient Boosting", 10, 100, 50, step=10)
ab_estimators = st.sidebar.slider("Estimators for AdaBoost", 10, 100, 50, step=10)

st.write("### Dataset Overview")
st.write(df.head())
st.write("Shape:", df.shape)
st.write("Missing Values:", df.isnull().sum().sum())

fig, ax = plt.subplots(figsize=(15, 15))
df.hist(ax=ax)
st.pyplot(fig)

fig, ax = plt.subplots()
sns.countplot(x='target', data=df, ax=ax)
st.pyplot(fig)

corr_matrix = df.corr()
fig, ax = plt.subplots(figsize=(20, 20))
sns.heatmap(data=corr_matrix, annot=True, cmap='RdYlGn', ax=ax)
st.pyplot(fig)

dataset = pd.get_dummies(df, columns=['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])
scaler = StandardScaler()
columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
dataset[columns_to_scale] = scaler.fit_transform(dataset[columns_to_scale])

X = dataset.drop('target', axis=1)
y = dataset['target']

def evaluate_model(model):
    scores = cross_val_score(model, X, y, cv=10)
    return round(scores.mean(), 4) * 100

st.write("### Model Performance")
if selected_algorithm == "KNN":
    model = KNeighborsClassifier(n_neighbors=knn_neighbors)
elif selected_algorithm == "Decision Tree":
    model = DecisionTreeClassifier(max_depth=decision_depth)
elif selected_algorithm == "Random Forest":
    model = RandomForestClassifier(n_estimators=forest_estimators)
elif selected_algorithm == "SVM":
    model = SVC(kernel=svm_kernel)
elif selected_algorithm == "Gradient Boosting":
    model = GradientBoostingClassifier(n_estimators=gb_estimators)
elif selected_algorithm == "AdaBoost":
    model = AdaBoostClassifier(n_estimators=ab_estimators)

accuracy = evaluate_model(model)
st.write(f"{selected_algorithm} Classifier Accuracy: {accuracy}%")

st.write("### Conclusion")
if accuracy > 80:
    st.success("The selected model is performing well with high accuracy!")
elif accuracy > 70:
    st.warning("The model has decent accuracy but can be improved.")
else:
    st.error("The model has low accuracy. Consider tuning parameters or using a different algorithm.")
