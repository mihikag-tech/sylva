import streamlit as st
import pickle
import pandas as pd
import sklearn
import xgboost as xgb
import numpy as np

#Visual web display
st.title("Hello! Welcome to Sylva!")

st.header("Let's get some information first.")
unique_counties = pd.read_csv('county_names.csv')
# Records county choice through dropdown; saves in var 'county'
county = st.selectbox(
    "Please choose a county to target:", 
    unique_counties, 
    placeholder = " "
)
budget = st.number_input("What's your budget (in thousands)?")
st.write("Your budget is " + str(budget) + "k and your county is " + str(county))

st.write(
    "Here's the data we used:"
)

df = pd.read_csv("Combined_dataset_model.csv")
df = pd.get_dummies(df, columns=["biome"], dtype=int)
df = df.drop(columns=['Unnamed: 0'])


features = ['land_area', 'treecanopy', 'tc_gap',
       'priority_i', 'pctpocnorm', 'pctpovnorm', 'unemplnorm', 'dep_perc',
       'depratnorm', 'tes', 'tesctyscor', 'rank',
       'rankgrpsz', 'Mean_Temp', 'Median_Temp', 'STD_Temp', 'Min_Temp',
       'Max_Temp', 'Mean_Rain', 'Median_Rain', 'STD_Rain', 'Min_Rain',
       'Max_Rain', 'biome_Desert', 'biome_Forest', 'biome_Grassland']
st.dataframe(df)
target = ['health_nor']
X_df = df[features]
y_df = df[target]
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X_df, y_df, test_size = 0.2, random_state = 42)

st.write(
    "Here's the predicted change in HBI for each segment of the county:"
)

"""
loaded_model = pickle.load(open('xgb_model_pickle', 'rb'))
xgb_pred = loaded_model.predict(X_test)
"""
model = xgb.XGBRegressor(learning_rate = 0.2, max_depth = 4, n_estimators = 300)
model.fit(X_train, y_train)
model_pred = model.predict(X_test)
"""
graph_y = [[y_test], [xgb_pred]]
graph_data = pd.DataFrame(data = {
    "x": X_test['priority_i'],
    "y": y_test

})
"""

solution_effects = {
    "green_street": {
        "treecanopy": 1.1 #adds 10% tree canopy
    },

    "parking_lot": {
        "treecanopy": 1.06
    },

    "urban_forest": {
        "treecanopy": 1.2
    },

    "green_roof": {
        "treecanopy": 1.05
    },

    "green_belt": {
        "treecanopy": 1.2
    },

    "community_park": {
        "treecanopy": 1.14,
    },

    "community_garden": {
        "treecanopy": 1.03,
    }
}


def impact_calc(model, county, solution_effects, features):
    county_df = df[df["county"] == county].drop(columns='health_nor')
    results = []

    for solution_name, effects in solution_effects.items():
        for row in county_df.itertuples(index=True):
            baseline = df.loc[[row.Index], features]
            prediction = model.predict(baseline)[0]

            modified = baseline.copy()
            for feature_key, multiplier in effects.items():
                modified[feature_key] = modified[feature_key] * multiplier

            update_prediction = model.predict(modified)[0]
            pct_hbi_chg = np.abs((update_prediction - prediction) / prediction) * 100

            results.append({
                "county": county,
                "solution": solution_name,
                "original_hbi": prediction, 
                "modified_hbi": update_prediction,
                "pct_hbi_chg": pct_hbi_chg
            })

    return pd.DataFrame(results)
st.write(impact_calc(model, str(county), solution_effects, features))

