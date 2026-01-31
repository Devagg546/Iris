import streamlit as st
import numpy as np
import joblib

#load the model
model= joblib.load("rfc.pkl")
st.title(" MY FIRST PROJECT : Iris Flower Species Prediction")
st.write("AIML Model Deployment")

#input slider 
sepal_length= st.slider("Sepal Length", 4.0,8.0,5.1)
sepal_width= st.slider("Sepal Width", 2.0,5.0,5.1)
petal_length= st.slider("Petal Length", 1.0,7.0,5.1)
petal_width= st.slider("Petal Width", 0.0,2.5,5.1)

#predict button 
if st.button("Predict"):
    input_data= np.array([[sepal_length,sepal_width,petal_length,petal_width]])
    pred=model.predict(input_data)[0]
    st.success(f"The predicted species is: {pred}")