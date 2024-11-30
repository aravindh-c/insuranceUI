import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.title('Insurance prediction app')

df=pd.read_csv('train.csv')

Gender= st.selectbox("Gender", pd.unique(df["Gender"]))
Age= st.selectbox("Age", pd.unique(df["Age"]))
Driving_License= st.selectbox("Driving_License", pd.unique(df["Driving_License"]))
Region_Code= st.selectbox("Region_Code", pd.unique(df["Region_Code"]))
Previously_Insured= st.selectbox("Previously_Insured", pd.unique(df["Previously_Insured"]))
Vehicle_Age= st.selectbox("Vehicle_Age", pd.unique(df["Vehicle_Age"]))

Vehicle_Damage = st.selectbox("Vehicle_Damage", pd.unique(df["Vehicle_Damage"]))
Annual_Premium = st.number_input('Annual_Premium')
Policy_Sales_Channel = st.selectbox("Policy_Sales_Channel", pd.unique(df["Policy_Sales_Channel"]))
Vintage = st.number_input('Vintage')


inputs={
    'Gender':Gender,
    'Age':Age,
    'Driving_License':Driving_License,
    'Region_Code':Region_Code,
    'Previously_Insured':Previously_Insured,
    'Vehicle_Age':Vehicle_Age,
    'Vehicle_Damage':Vehicle_Damage,
    'Annual_Premium':Annual_Premium,
    'Policy_Sales_Channel':Policy_Sales_Channel,
    'Vintage':Vintage
}

model= joblib.load('insurance_model_pipeline_hyper1.pkl')

if st.button('Predict'):
    x_input=pd.DataFrame(inputs,index=[0])
    prediction=model.predict(x_input)
    st.write(' Predicted value is ::')
    st.write(prediction)

st.subheader('Please upload a csv file for prediction :')
upload_file = st.file_uploader('Choose a csv file ',type=['csv'])

if upload_file is not None:
    df=pd.read_csv(upload_file)
    st.write('File uploaded successfully !!')
    st.write(df.head(2))
    if st.button('Predict for the uploaded file'):

        df['Response'] = model.predict(df)
        st.write(' Predicted value is ::')
        st.write(df['is_promoted'])
        st.download_button(label='Download predicted results', data=df.to_csv(index=False),
                           mime='text/csv',
                           file_name='insurance_predict_output.csv')






