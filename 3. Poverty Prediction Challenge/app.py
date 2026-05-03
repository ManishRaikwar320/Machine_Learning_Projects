from flask import request, render_template, Flask
from preprocessor import DataPreprocessing
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("CatboostModel.pkl")



@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["data"]
    data = pd.read_json(data)
    data = DataPreprocessing(data)
    pred = model.predict(data)
    return {"prediction": int(pred[0])}



@app.route("/predict", methods=["POST"])
def predict():
    # HTML form se data lena
    feature = float(request.form["f1"])
    data = pd.read_json(feature)
    data = DataPreprocessing(data)

    prediction = model.predict(data)

    return render_template(
        "index.html",
        result=f"Prediction: {prediction[0]}"
    )

if __name__ == "__main__":
    app.run(debug=True)
