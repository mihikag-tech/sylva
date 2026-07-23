import streamlit as st
import pickle
import xgboost
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from threading import RLock
import numpy as np

data = pd.read_csv('Combined_dataset_model.csv')
X = data.drop(['health_nor', 'biome', 'county'], axis = 1)
y = data['health_nor']
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.2, random_state = 42)

#Visual web display
st.title("Welcome to Sylva!")

county = st.selectbox(
    "Please choose a county to begin:", 
    data['county']
)
st.write(
    "Here's the data we used:"
)
st.dataframe(data)

st.write(
    "Here's how it matched up to our model's predictions:"
)
loaded_model = pickle.load(open('xgb_model_pickle', 'rb'))
xgb_pred = loaded_model.predict(X_test)

graph_y = [[y_test], [xgb_pred]]
graph_data = pd.DataFrame(data = {
    "x": X_test['priority_i'],
    "y": y_test

})

"""
_lock = RLock()

x = np.random.normal(1, 1, 100)
y = np.random.normal(1, 1, 100)

with _lock:
    fig, ax = plt.subplots()
    ax.scatter(X_test['priority_i'], graph_data)
    st.pyplot(fig)
"""

st.line_chart(data = graph_data, x = 'x', y = 'y', color = ['red'])