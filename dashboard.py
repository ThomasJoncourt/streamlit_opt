import streamlit as st
import pandas as pd
import numpy as np
from joblib import dump, load
import math

st.title('Prédiction Time To Market des issues du GLIA')

form = st.form("my_form")

title = st.text_input('Title')
body = st.text_input('Body')

form.form_submit_button("Submit")

st.write('The current title is', title)
st.write('The current body is', body)

vectorizer = load('vectorizer_title_body.joblib')

text_to_vectorize = title + body

X = vectorizer.transform([text_to_vectorize])

reg = load("regr_title_body.joblib")

prediction = reg.predict(X.reshape(1, -1))

st.write('Prediction : ', int(prediction[0]), ' jours', " , ", int(math.modf(prediction[0])[0]*24), 'heures')

# Running
#python -m streamlit run your_script.py