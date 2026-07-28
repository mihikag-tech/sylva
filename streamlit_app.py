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

solution_effects = pd.DataFrame(solution_effects)
solution_effects = solution_effects.T

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
result = impact_calc(model, str(county), solution_effects, features)

# Precomputed from XGBoost model.predict() for each candidate
costs = [c for c in solution_effects['Cost']]

from ortools.sat.python import cp_model

# Precomputed from XGBoost model.predict() for each candidate
costs = [c for c in solution_effects['Cost']]
total_cost = 0
final_results = pd.DataFrame(columns = ["county", "green_streets", "green_parking_lots", 
                                        "urban_forests", "green_roofs", "green_belts", 
                                        "parks", "gardens", "total_impact", "total_cost"])

# make it maximize impacts, not tree canopy
# predicted health impact
for index, row in result.iterrows():
  baseline = row
  temp_changes = {
      "gstreet": baseline['green_street_pct_temp_norm_chg'], 
      "gparklot": baseline['green_parking_lot_pct_temp_norm_chg'], 
      "urbforest": baseline['urban_forest_pct_temp_norm_chg'],
      "groof": baseline['green_roof_pct_temp_norm_chg'],
      "gbelt": baseline['green_belt_pct_temp_norm_chg'],
      "park": baseline['park_pct_temp_norm_chg'],
      "garden": baseline['garden_pct_temp_norm_chg']
  }
  model_cp = cp_model.CpModel()

  gstreet = model_cp.new_int_var(0, 5, "gstreet")
  gparklot = model_cp.new_int_var(0, 5, "gparklot")
  urbforest = model_cp.new_int_var(0, 5, "urbforest")
  groof = model_cp.new_int_var(0, 5, "groof")
  gbelt = model_cp.new_int_var(0, 5, "gbelt")
  park = model_cp.new_int_var(0, 5, "park")
  garden = model_cp.new_int_var(0, 5, "garden")

  model_cp.Add(
     gstreet * int(costs[0]) +
     gparklot * int(costs[1]) +
     urbforest * int(costs[2]) +
     groof * int(costs[3]) +
     gbelt * int(costs[4]) +
     park * int(costs[5]) +
     garden * int(costs[6])
     <= budget
  )
  model_cp.Minimize(
     gstreet * temp_changes["gstreet"] +
     gparklot * temp_changes["gparklot"] +
     urbforest * temp_changes["urbforest"] +
     groof * temp_changes["groof"] +
     gbelt * temp_changes["gbelt"] +
     park * temp_changes["park"] +
     garden * temp_changes["garden"]
  )

  solver = cp_model.CpSolver()
  status = solver.Solve(model_cp) # Corrected to use model_cp

  if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
    final_results = pd.concat([final_results, pd.DataFrame({
      "county": [baseline['county']],
      "green_streets": [solver.Value(gstreet)],
      "green_parking_lots": [solver.Value(gparklot)],
      "urban_forests": [solver.Value(urbforest)],
      "green_roofs": [solver.Value(groof)],
      "green_belts": [solver.Value(gbelt)],
      "parks": [solver.Value(park)],
      "gardens": [solver.Value(garden)],
      "total_impact": [solver.ObjectiveValue()],
      "total_cost": [solver.Value(
        gstreet * int(costs[0]) +
        gparklot * int(costs[1]) +
        urbforest * int(costs[2]) +
        groof * int(costs[3]) +
        gbelt * int(costs[4]) +
        park * int(costs[5]) +
        garden * int(costs[6]))]
    })])
    st.dataframe(final_results)
  elif status == cp_model.INFEASIBLE:
    st.write("No solution found that satisfies the constraints.")
  else:
    st.write("Solver could not find an optimal or feasible solution.")


st.write("An explanation of the solutions:")
explanations = pd.read_csv('Green Intervention Budgets - Sheet2.csv')
for row in explanations.iterrows():
   print(row)
