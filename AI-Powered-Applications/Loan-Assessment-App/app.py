import os
import threading
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import gdown
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create app - templates are in the 'templates' folder in the same directory
app = Flask(__name__)

# Global variables for models
xgb_model = None
scaler = None
prod_dummies = None
models_loaded = False
load_error = None

def load_models():
    global xgb_model, scaler, prod_dummies, models_loaded, load_error
    
    try:
        logger.info("Starting model loading...")
        
        # Define file paths
        xgb_output = "loan_xgb_model.pkl"
        dummy_output = "loan_dummies.pkl"
        sc_output = "loan_scaler.pkl"

        # Check and download/load files
        file_configs = [
            (xgb_output, "1KP_ixxX-vUp7eilouWVPrpf7llicWEylbX", "XGBoost model"),
            (dummy_output, "1ZhgQlzrmZ5uv6YvbOAd5ONqAEYQrgC9nqSG_", "encoder"),
            (sc_output, "1cszbq3Qedprzgg5FUkVcAf89FKXpke2Omf", "scaler")
        ]

        for file_path, file_id, file_type in file_configs:
            if not os.path.exists(file_path):
                logger.info(f"Downloading {file_type} to {file_path}...")
                file_url = f"https://drive.google.com/uc?id={file_id}"
                gdown.download(file_url, file_path, quiet=False)
            else:
                logger.info(f"Loading existing {file_type} from {file_path}...")

        # Load models with individual error handling
        logger.info("Loading scaler...")
        scaler = joblib.load(sc_output)
        logger.info("✓ Scaler loaded")
        
        logger.info("Loading encoder...")
        prod_dummies = joblib.load(dummy_output)
        logger.info("✓ Encoder loaded")
        
        logger.info("Loading XGBoost model...")
        xgb_model = joblib.load(xgb_output)
        logger.info("✓ XGBoost model loaded")
        
        # Verify all models are loaded
        if all([xgb_model is not None, scaler is not None, prod_dummies is not None]):
            models_loaded = True
            logger.info("🎉 All models loaded successfully!")
        else:
            raise Exception("One or more models failed to load properly")
            
    except Exception as e:
        logger.error(f"Model loading error: {str(e)}")
        models_loaded = False
        load_error = str(e)

def check_models_ready():
    """Check if models are loaded and ready for prediction"""
    return models_loaded and all([xgb_model is not None, scaler is not None, prod_dummies is not None])

# Start model loading in a separate thread
logger.info("Starting model loading thread...")
load_thread = threading.Thread(target=load_models, daemon=True)
load_thread.start()

@app.route("/")
def home():
    status = "Error loading models" if load_error else ("Loading..." if not models_loaded else "Ready")
    return render_template("home.html", 
                         query="", 
                         loading_status=status,
                         load_error=load_error)

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/status")
def status():
    """Endpoint to check model loading status"""
    return {
        "models_loaded": models_loaded,
        "load_error": load_error,
        "models_ready": check_models_ready(),
        "individual_models": {
            "xgb_model": xgb_model is not None,
            "scaler": scaler is not None,
            "prod_dummies": prod_dummies is not None
        }
    }

@app.route("/predict", methods=["POST"])
def loanEvaluator():
    global models_loaded, load_error
    
    # Check if models are ready
    if not check_models_ready():
        if load_error:
            return render_template("home.html", 
                                 output1=f"Model Error: {load_error}", 
                                 output2="")
        return render_template("home.html", 
                             output1="Models are still loading. Please wait...", 
                             output2="")

    try:
        # Get inputs from form with validation
        required_fields = ['query1', 'query2', 'query3', 'query4', 'query5', 
                          'query6', 'query7', 'query8', 'query9', 'query10', 'query11']
        
        for field in required_fields:
            if field not in request.form or not request.form[field].strip():
                return render_template("home.html", 
                                     output1=f"Error: Missing field {field}", 
                                     output2="")

        # Convert inputs
        inputQuery1 = float(request.form['query1'])
        inputQuery2 = float(request.form['query2'])
        inputQuery3 = request.form['query3']
        inputQuery4 = float(request.form['query4'])
        inputQuery5 = request.form['query5']
        inputQuery6 = request.form['query6']
        inputQuery7 = float(request.form['query7'])
        inputQuery8 = float(request.form['query8'])
        inputQuery9 = float(request.form['query9'])
        inputQuery10 = request.form['query10']
        inputQuery11 = float(request.form['query11'])

        # Prepare input data
        data = [[inputQuery1, inputQuery2, inputQuery3, inputQuery4, inputQuery5,
                 inputQuery6, inputQuery7, inputQuery8, inputQuery9, inputQuery10, inputQuery11]]
        
        data_fr = pd.DataFrame(data, columns=[
            "person_age", "person_income", "person_home_ownership",
            "person_emp_length", "loan_intent", "loan_grade",
            "loan_amnt", "loan_int_rate", "loan_percent_income",
            "cb_person_default_on_file", "cb_person_cred_hist_length"
        ])

        # One-hot encoding
        categorical_col = ["person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"]
        data_enc = pd.get_dummies(data_fr, columns=categorical_col, drop_first=True)
        data_enc = data_enc.reindex(columns=prod_dummies, fill_value=0)
        
        # Scale & predict
        data_sc = scaler.transform(data_enc)
        proba = xgb_model.predict_proba(data_sc)

        # Determine prediction
        pred_class = int(proba[0][1] >= 0.5)
        
        if pred_class == 0:
            output1 = "This Client Qualifies for a Loan"
            confidence = f"{round(float(proba[0][0] * 100), 2)}%"
        else:
            output1 = "This Client Does Not Qualify for a Loan"
            confidence = f"{round(float(proba[0][1] * 100), 2)}%"

        output2 = f"Confidence: {confidence}"
        
        # Return results with all form data preserved
        return render_template("home.html", 
                             output1=output1, 
                             output2=output2,
                             **{f'query{i}': request.form[f'query{i}'] for i in range(1, 12)})

    except ValueError as e:
        error_message = f"Invalid input: Please check all fields contain valid numbers"
        logger.error(f"ValueError in prediction: {e}")
        return render_template("home.html", output1=error_message, output2="")
    except Exception as e:
        error_message = f"Prediction error: {str(e)}"
        logger.error(f"Prediction error: {e}")
        return render_template("home.html", output1=error_message, output2="")


# Vercel compatible handler
def handler(request=None):
    if request is None:
        return app
    
    with app.request_context(request.environ):
        try:
            response = app.full_dispatch_request()
        except Exception as e:
            response = app.handle_exception(e)
            
        return response

# For local development
if __name__ == "__main__":
    logger.info("Starting Flask application...")
    app.run(debug=True, host='0.0.0.0', port=7860)

