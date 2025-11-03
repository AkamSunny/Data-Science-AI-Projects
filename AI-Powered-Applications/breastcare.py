import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import math
from flask import Flask, render_template, request
import re

app = Flask(__name__)



@app.route("/")
def home():
    return render_template("home.html", query="")



@app.route("/about")
def about():
    return render_template("about.html")



@app.route("/contact")
def contact():
    return render_template("contact.html")


@app.route("/predict", methods=['POST'])
def cancerPrediction():
    try:
        df_xgb = pd.read_csv('breast_cancer.csv')

        
        inputQuery1 = float(request.form['query1'])
        inputQuery2 = float(request.form['query2'])
        inputQuery3 = float(request.form['query3'])
        inputQuery4 = float(request.form['query4'])
        inputQuery5 = float(request.form['query5'])

        features = ['perimeter_worst', 'concave points_worst', 'concave points_mean', 'area_mean', 'area_worst',
                    'diagnosis']
        xgb_df = df_xgb[features]

        xgb_df['diagnosis'] = xgb_df['diagnosis'].map({'M': 1, 'B': 0})

        xgb_x2 = xgb_df.drop('diagnosis', axis=1)
        y_xgb = xgb_df['diagnosis']

        
        xgb_train_x, xgb_test_x, y_train, y_test = train_test_split(xgb_x2, y_xgb, test_size=0.2, random_state=3)

        
        xgb_sc = StandardScaler()
        xgb_train_x = xgb_sc.fit_transform(xgb_train_x)
        xgb_test_x = xgb_sc.transform(xgb_test_x)

        
        xgb2 = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, scale_pos_weight=357 / 212,
                             reg_alpha=0.5, reg_lambda=2.0, subsample=0.7, random_state=42)

        xgb2.fit(xgb_train_x, y_train)

       
        data = [[inputQuery1, inputQuery2, inputQuery3, inputQuery4, inputQuery5]]
        new_df = pd.DataFrame(data,
                              columns=['perimeter_worst', 'concave points_worst', 'concave points_mean', 'area_mean',
                                       'area_worst'])

       
        new_df_scaled = xgb_sc.transform(new_df)

        # predicting Out sample data
        real_pred = xgb2.predict(new_df_scaled)
        real_proba = xgb2.predict_proba(new_df_scaled)

        
        if real_pred[0] == 1:
            output1 = "The patient is diagnosed with Breast Cancer"
            # Get the probability for class 1 (second column) and convert to scalar
            confidence = float(real_proba[0][1] * 100)
            output2 = "Confidence: {:.2f}%".format(confidence)
        else:
            output1 = "The patient is not diagnosed with Breast Cancer"
            
            confidence = float(real_proba[0][0] * 100)
            output2 = "Confidence: {:.2f}%".format(confidence)

        return render_template("home.html", output1=output1, output2=output2,
                               query1=request.form['query1'], query2=request.form['query2'],
                               query3=request.form['query3'], query4=request.form['query4'],
                               query5=request.form['query5'])

    except Exception as e:
        
        error_message = f"Error processing request: {str(e)}"
        return render_template("home.html", output1=error_message, output2="",
                               query1=request.form.get('query1', ''),
                               query2=request.form.get('query2', ''),
                               query3=request.form.get('query3', ''),
                               query4=request.form.get('query4', ''),
                               query5=request.form.get('query5', ''))


if __name__ == "__main__":
    app.run()



