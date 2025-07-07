import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import joblib
import os

st.set_page_config(page_title="ML Model Deployment", layout="wide")

st.title("Machine Learning Model Deployment with Streamlit")


st.sidebar.title("Navigation")
options = st.sidebar.radio("Select a page:", 
                          ["Model Prediction", "Model Analysis", "About"])

@st.cache_resource
def load_model():
    if os.path.exists('model.pkl'):
        model = joblib.load('model.pkl')
    else:
        from sklearn.datasets import load_iris
        iris = load_iris()
        X = pd.DataFrame(iris.data, columns=iris.feature_names)
        y = iris.target
        
        model = RandomForestClassifier(n_estimators=100)
        model.fit(X, y)
        
        joblib.dump(model, 'model.pkl')
    return model

model = load_model()

feature_names = ['sepal length (cm)', 'sepal width (cm)', 
                'petal length (cm)', 'petal width (cm)']
target_names = ['setosa', 'versicolor', 'virginica']

if options == "Model Prediction":
    st.header("Make a Prediction")
    
    st.subheader("Input Features")
    col1, col2 = st.columns(2)
    
    with col1:
        sepal_length = st.slider('Sepal length (cm)', 4.0, 8.0, 5.4)
        sepal_width = st.slider('Sepal width (cm)', 2.0, 4.5, 3.4)
    
    with col2:
        petal_length = st.slider('Petal length (cm)', 1.0, 7.0, 1.3)
        petal_width = st.slider('Petal width (cm)', 0.1, 2.5, 0.2)
    
    input_data = np.array([[sepal_length, sepal_width, 
                          petal_length, petal_width]])
    
    if st.button("Predict"):
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)
        
        st.subheader("Prediction")
        st.write(f"Predicted class: **{target_names[prediction[0]]}**")
        
        st.subheader("Prediction Probability")
        proba_df = pd.DataFrame({
            'Class': target_names,
            'Probability': prediction_proba[0]
        })
        st.bar_chart(proba_df.set_index('Class'))
        
        st.subheader("Input Features")
        input_df = pd.DataFrame(input_data, columns=feature_names)
        st.write(input_df)

elif options == "Model Analysis":
    st.header("Model Analysis")
    
    from sklearn.datasets import load_iris
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    st.subheader("Model Performance Metrics")
    y_pred = model.predict(X_test)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.text("Classification Report:")
        report = classification_report(y_test, y_pred, 
                                     target_names=target_names)
        st.text(report)
    
    with col2:
        st.text("Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', 
                    xticklabels=target_names, 
                    yticklabels=target_names)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        st.pyplot(fig)
    
    st.subheader("Feature Importance")
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        feat_imp = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(feat_imp)
        
        with col2:
            fig, ax = plt.subplots()
            sns.barplot(x='Importance', y='Feature', data=feat_imp)
            plt.title('Feature Importance')
            st.pyplot(fig)
    else:
        st.write("This model doesn't support feature importance.")

else:
    st.header("About")
    st.write("""
    
    
    **Features:**
    - Interactive prediction interface
    - Model performance visualizations
    - Feature importance analysis
    
    **Dataset:** Iris dataset
    
    **Model:** Random Forest Classifier
    """)
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/9/9f/Iris_virginica.jpg/800px-Iris_virginica.jpg", 
             caption="Iris Virginica - One of the flower species in the dataset")