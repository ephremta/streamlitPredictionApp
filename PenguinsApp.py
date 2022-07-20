import pickle

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

st.header("""
**Simple  Penguin Prediction Web App**
""")
st.write("""
This App predicts the **Palmer Penguin** Species!
We found the public data from the [Palmer Penguins Github Repository](https://github.com/dataprofessor/datapalmerpenguins)
""")
st.sidebar.header('User Input Features')
st.sidebar.markdown("""
[Example CSV input files](https://raw.githubcontent.com/dataprofessor/data/master/penguins_example.csv)

""")
upload_file = st.sidebar.file_uploader('Upload your input csv files', type=["csv"])
if upload_file is not None:
    input_df = pd.read_csv(upload_file)
else:
    def user_input_features():
        island = st.sidebar.selectbox('Island', ('Biscoe', 'Dream', 'Torgersen'))
        sex = st.sidebar.selectbox('sex', ('male', 'female'))
        bill_length_mm = st.sidebar.slider('Bill Length(mm)', 32.1, 59.6, 43.9)
        bill_depth_mm = st.sidebar.slider('Bill depth(mm)', 13.1, 21.5, 17.2)
        fliper_length_mm = st.sidebar.slider('Flipper length(mm)', 172.0, 231.0, 201.0)
        body_mass_g = st.sidebar.slider('Body mass(g)', 2700.0, 63000.0, 4207.0)
        data = {'island': island,
                'bill_length_mm': bill_length_mm,
                'bill_depth_mm': bill_depth_mm,
                'flipper_length_mm': fliper_length_mm,
                'body_mass_g': body_mass_g,
                'sex': sex
                }
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

penguins_raw = pd.read_csv('penguins_cleaned.csv')
penguins = penguins_raw.drop(columns=['species'])
df = pd.concat([input_df, penguins], axis=0)

encode = ['sex', 'island']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummy], axis=1)
    del df[col]
df = df[:1]  # select only the first raw

st.subheader('User Input Features')
if upload_file is not None:
    st.write(df)
else:
    st.write('Awaitng csv file to be uploaded. Currently Using Sample Parameters (Shown below) ')
    st.write(df)

load_clf = pickle.load(open('penguins_clf.pkl', 'rb'))
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)

st.subheader('Prediction')
penguins_species = np.array(['Adelie', 'Chinstrap', 'Gentoo'])
st.write(penguins_species[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)
