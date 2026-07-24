import streamlit as st
import pickle
import pandas as pd
import sklearn
import xgboost as xgb
import numpy as np
from ortools.sat.python import cp_model

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

df = pd.read_csv('Combined_dataset_model.csv')
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


model = pickle.load(open('new_model.pkl', 'rb'))
xgb_pred = model.predict(X_test)


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
result = impact_calc(model, str(county), solution_effects, features)

# Precomputed from XGBoost model.predict() for each candidate
impacts = result   # predicted health impact
budget  = 1200000

model_cp = cp_model.CpModel()

gstreet = model_cp.new_int_var(0, 5, "gstreet")
gparklot = model_cp.new_int_var(0, 5, "gparklot")
urbforest = model_cp.new_int_var(0, 5, "urbforest")
groof = model_cp.new_int_var(0, 5, "groof")
gbelt = model_cp.new_int_var(0, 5, "gbelt")
park = model_cp.new_int_var(0, 5, "park")
garden = model_cp.new_int_var(0, 5, "garden")

model_cp.Add(
    gstreet * 60000 + 
    gparklot * 300000 + 
    urbforest * 500000 +
    groof * 200000 + 
    gbelt * 750000 + 
    park * 750000 + 
    garden * 40000 
    <= budget
)
model_cp.Maximize(
    gstreet * 0.1 + 
    gparklot * 0.06 + 
    urbforest * 0.2 + 
    groof * 0.05 + 
    gbelt * 0.2 + 
    park * 0.14 + 
    garden * 0.03
)

solver = cp_model.CpSolver()
status = solver.Solve(model_cp)

if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
    print(f"Optimal solutions found!")
    print(f"  Green Street interventions: {solver.Value(gstreet)}")
    print(f"  Green Parking Lot interventions: {solver.Value(gparklot)}")
    print(f"  Urban Forest interventions: {solver.Value(urbforest)}")
    print(f"  Green Roof interventions: {solver.Value(groof)}")
    print(f"  Green Belt interventions: {solver.Value(gbelt)}")
    print(f"  Park interventions: {solver.Value(park)}")
    print(f"  Garden interventions: {solver.Value(garden)}")
    print(f"  Total impact: {solver.ObjectiveValue()}")
    print(f"  Total cost: {solver.Value(gstreet) * 60000 + solver.Value(gparklot) * 300000 + solver.Value(urbforest) * 500000}")
elif status == cp_model.INFEASIBLE:
    print("No solution found that satisfies the constraints.")
else:
    print("Solver could not find an optimal or feasible solution.")