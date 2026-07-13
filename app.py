from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained Random Forest model
model = pickle.load(open("saved_models/random_forest_model.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")
@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/contact")
def contact():
    return render_template("contact.html")


@app.route("/prediction")
def prediction():
    return render_template("prediction.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Read values from the form
        N = float(request.form["nitrogen"])
        P = float(request.form["phosphorous"])
        K = float(request.form["potassium"])
        temperature = float(request.form["temperature"])
        humidity = float(request.form["humidity"])
        ph = float(request.form["ph"])
        rainfall = float(request.form["rainfall"])

        # Create input array
        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

        # Predict
        prediction = model.predict(input_data)

        result = prediction[0]

        return render_template(
            "prediction.html",
            prediction_text=f"🌱 Recommended Crop: {result}"
        )

    except Exception as e:
        return render_template(
            "prediction.html",
            prediction_text=f"Error: {str(e)}"
        )


if __name__ == "__main__":
    app.run(debug=True)