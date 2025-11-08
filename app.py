from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model and preprocessor
model = joblib.load("model.pkl")
preprocessor = joblib.load("preprocessor.pkl")


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        # Get form data
        length_of_stay = int(request.form["length_of_stay"])
        service = request.form["service"]
        available_beds = int(request.form["available_beds"])
        patient_demand = int(request.form["patient_demand"])
        patients_admitted = int(request.form["patients_admitted"])
        patients_refused = int(request.form["patients_refused"])
        staff_morale = int(request.form["staff_morale"])

        # Create input DataFrame
        input_df = pd.DataFrame(
            [
                {
                    "LengthOfStay": length_of_stay,
                    "service": service,
                    "available_beds": available_beds,
                    "patients_request": patient_demand,
                    "patients_admitted": patients_admitted,
                    "patients_refused": patients_refused,
                    "staff_morale": staff_morale,
                }
            ]
        )

        # Predict
        prediction = model.predict(input_df)[0]
        prediction = round(prediction, 2)

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
