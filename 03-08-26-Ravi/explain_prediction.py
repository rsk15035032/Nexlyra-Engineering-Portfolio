import argparse
import joblib
import pandas as pd
import shap
import numpy as np

model_name = "tuned_lgb_model.pkl"

model = joblib.load(model_name)

explainer = shap.TreeExplainer(model)

def text_explanation(input_data):

    shap_values = explainer.shap_values(input_data)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap_row = shap_values[0]

    fraud_probablity = model.predict_proba(input_data)[0, 1]
    fraud_prediction = model.predict(input_data)[0]

    exp = pd.DataFrame({
        "Features" : input_data.columns,
        "Values" : input_data.iloc[0].values,
        "SHAP": shap_row
    })

    exp["Importance"] = (np.abs(exp["SHAP"])/np.abs(exp["SHAP"]).sum()*100)
    exp = exp.sort_values("Importance",ascending=False)

    print("=>" * 60)

    if fraud_prediction == 1:
        print(f"Prediction:Fraud")
    else:
        print(f"Prediction:Non-Fraud")

    print(f"Fraud Probability:{fraud_probablity:.3%}\n")

    if fraud_prediction == 1:
        print("Flagged due to:")
        positive = exp[exp["SHAP"] > 0]

        for _, row in positive.head(5).iterrows():
            print(f"{row.Features} = {row.Values}\n")
            print(f"({row.Importance:.2f}% importance)")
    else:

        print("Predicted Non-Fraud because:")
        negative = exp[exp["SHAP"] < 0]

        for _, row in negative.head(5).iterrows():
            print(f"{row.Features} = {row.Values}\n")
            print(f"({row.Importance:.2f}% importance)")
         


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",required=True,help="CSV containing one transaction")
    args = parser.parse_args()

    input_data = pd.read_csv(args.csv)
    text_explanation(input_data)


## "python explain_prediction.py --csv test.csv" run in the terminal
    

    


