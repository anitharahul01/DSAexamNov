from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)
model = joblib.load("model/best_model.pkl")


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        data = {
            "LengthOfStay": int(request.form["LengthOfStay"]),
            "service": request.form["service"],
            "available_beds": int(request.form["available_beds"]),
            "patients_request": int(request.form["patients_request"]),
            "patients_admitted": int(request.form["patients_admitted"]),
            "patients_refused": int(request.form["patients_refused"]),
            "ActualStaffPresent": int(request.form["ActualStaffPresent"]),
        }
        df = pd.DataFrame([data])
        prediction = model.predict(df)[0]
        return render_template("index1.html", prediction=round(prediction, 2))
    return render_template("index1.html", prediction=None)


if __name__ == "__main__":
    app.run(debug=True)
