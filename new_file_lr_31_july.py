import streamlit as st
import pandas as pd
import pickle
import sklearn
import numpy as np
from sklearn.metrics import mean_squared_error
#data=pd.read_csv('/Users/sridhar/Documents/PythonProject/cars_24_combined.csv')
yr=st.number_input('Year')
dist=st.slider('Distance, 0,1000000')
fuel=st.radio('Enter the fuel type of the car',options=['PETROL','DIESEL','LPG','CNG'])
vdrive=st.radio("Drive type of vehicle", options=['MANUAL','AUTOMATIC'])
vtype=st.radio('Vehicle type', options=['HatchBack', 'Sedan', 'SUV', 'Lux_SUV', 'Lux_sedan'])

with open('my_model.pkl', 'rb') as my_file:
    my_model=pickle.load(my_file)

enc={'fuel':{'CNG':0,'LPG':1,'DIESEL':2, 'PETROL':3},'vdrive':{'MANUAL':0,'AUTOMATIC':1},'vtype':{'HatchBack':0, 'Sedan':1, 'SUV':3, 'Lux_SUV':4, 'Lux_sedan':2}}

fuel_enc=enc['fuel'][fuel]
vdrive_enc=enc['vdrive'][vdrive]
vtype_enc=enc['vtype'][vtype]

inp_record=np.array([yr,dist,fuel_enc, vdrive_enc, vtype_enc]).reshape(1,-1)
st.text(inp_record)
st.text(f"Shape of input is {inp_record.shape}")
ypred=my_model.predict(inp_record)
pred_button=st.button('Predict')
if(pred_button):
    st.text(ypred)