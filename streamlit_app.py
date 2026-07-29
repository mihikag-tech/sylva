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
)
#Records budget through numerical entry; saves in var 'budget'
budget = st.number_input("What's your budget (in thousands)?")
st.write("Your budget is " + str(budget) + "k and your county is " + str(county))

df = pd.read_csv('Combined_dataset_model.csv')
df = pd.get_dummies(df, columns=["biome"], dtype=int)
df = df.drop(columns=['Unnamed: 0'])

features = ['land_area', 'treecanopy', 'tc_gap',
       'priority_i', 'pctpocnorm', 'pctpovnorm', 'unemplnorm', 'dep_perc',
       'depratnorm', 'health_nor', 'tes', 'tesctyscor', 'rank',
       'rankgrpsz', 'Mean_Temp', 'Median_Temp', 'STD_Temp', 'Min_Temp',
       'Max_Temp', 'Mean_Rain', 'Median_Rain', 'STD_Rain', 'Min_Rain',
       'Max_Rain']
target = ['temp_norm']
X_df = df[features]
y_df = df[target]
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X_df, y_df, test_size = 0.2, random_state = 42)

model = pickle.load(open('optimized_xgb_model.pkl', 'rb'))
xgb_pred = model.predict(X_test)

solution_costs = pd.read_csv("Green Intervention Budgets - Sheet1.csv")

# Define what each solution actually changes, and by how much
solution_effects = {
    "green_street": {
        "treecanopy": 1.10,
        "Mean_Temp": 1.15,
        "Cost": int(solution_costs[solution_costs['intervention'] == "Green Street"]['sq_foot_cost_dollars'].iloc[0] * solution_costs[solution_costs['intervention'] == "Green Street"]["avg_sq_feet"].iloc[0])


    },

    "green_parking_lot": {
        "treecanopy": 1.06,
        "Mean_Temp": 1.09,
        "Cost": int(solution_costs[solution_costs['intervention'] == "Green Parking Lot"]['sq_foot_cost_dollars'].iloc[0] * solution_costs[solution_costs['intervention'] == "Green Parking Lot"]["avg_sq_feet"].iloc[0])

    },

    "urban_forest": {
        "treecanopy": 1.2,
        "Mean_Temp": 1.03,
        "Cost": int(solution_costs[solution_costs['intervention'] == "Urban Forest"]['sq_foot_cost_dollars'].iloc[0] * solution_costs[solution_costs['intervention'] == "Urban Forest"]["avg_sq_feet"].iloc[0])

    },

    "green_roof": {
        "treecanopy": 1.05,
        "Mean_Temp": 1.01,
        "Cost": int(solution_costs[solution_costs['intervention'] == "Green Roof"]['sq_foot_cost_dollars'].iloc[0] * solution_costs[solution_costs['intervention'] == "Green Roof"]["avg_sq_feet"].iloc[0])
    },

    "green_belt": {
        "treecanopy": 1.2,
        "Mean_Temp": 1.1,
        "Cost": int(solution_costs[solution_costs['intervention'] == "Green Belt"]['sq_foot_cost_dollars'].iloc[0] * solution_costs[solution_costs['intervention'] == "Green Belt"]["avg_sq_feet"].iloc[0])
    },

    "park": {
        "treecanopy": 1.14,
        "Mean_Temp": 1.01,
        "Cost": int(solution_costs[solution_costs['intervention'] == "Park"]['sq_foot_cost_dollars'].iloc[0] * solution_costs[solution_costs['intervention'] == "Park"]["avg_sq_feet"].iloc[0])
    },

    "garden": {
        "treecanopy": 1.03,
        "Mean_Temp": 1.06,
        "Cost": int(solution_costs[solution_costs['intervention'] == "Garden"]['sq_foot_cost_dollars'].iloc[0] * solution_costs[solution_costs['intervention'] == "Garden"]["avg_sq_feet"].iloc[0])
    }
}

solution_effects = pd.DataFrame(solution_effects)
solution_effects = solution_effects.T


def impact_calc(model, county, solution_effects, features):
    county_df = df[df["county"] == county]
    final_results = []

    for row_idx, row_data in county_df.iterrows():
        baseline = row_data[features]
        prediction = model.predict(pd.DataFrame([baseline]))[0]

        # Initialize dictionary for the current row
        row_output = {
            "county": county,
            "original_temp": prediction
        }

        for solution_name, effects_series in solution_effects.iterrows():
            modified = baseline.copy()
            for feature_key, multiplier in effects_series.items():
                # Only modify features that are in the model's feature list and are not 'Cost'
                if feature_key in features and feature_key != 'Cost':
                    modified[feature_key] = modified[feature_key] * multiplier

            update_prediction = model.predict(pd.DataFrame([modified]))[0]
            pct_temp_norm_chg = ((update_prediction - prediction) / prediction) * 100 # Calculate percentage change

            # Add solution-specific results as new columns
            row_output[f"{solution_name}_modified_temp"] = update_prediction
            row_output[f"{solution_name}_pct_temp_norm_chg"] = pct_temp_norm_chg

        final_results.append(row_output)

    return pd.DataFrame(final_results)

result = impact_calc(model, county, solution_effects, features)
st.dataframe(result)

costs = [c for c in solution_effects['Cost']]
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
