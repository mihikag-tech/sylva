import streamlit as st
import pickle
import xgboost

st.title("Welcome to Sylva!")
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)

loaded_model = pickle.load(open('xgb_model_pickle', 'rb'))

