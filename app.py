import numpy as np
import streamlit as st
import pickle

pipe = pickle.load(open('pipe.pkl','rb'))
df = pickle.load(open('df.pkl','rb'))

st.title("Salary Prediction")

Age = st.number_input('Age', value=0, step=1, format="%d")
Gender = st.selectbox('Sex',df['Gender'].unique())
Education = st.selectbox('Education',df['Education Level'].unique())
Job = st.selectbox('Job Role',df['Job Title'].unique())
Exp = st.number_input('Years of Experience', value=0, step=1, format="%d")
Country = st.selectbox('Country',df['Country'].unique())
Race = st.selectbox('Race',df['Race'].unique())

if st.button('Predict Salary'):
    query = np.array([Age,Gender,Education,Job,Exp,Country,Race])
    query = query.reshape(1,7)
    result = pipe.predict(query)[0]
    # st.write(result)
    # st.title("The predicted Salary of this configuration is " , result)
    st.title("The predicted Salary is  ${}".format(result))
    
