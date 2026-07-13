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
        crop_info = {

    "rice": {
        "season": "Rainy",
        "water": "High",
        "soil": "Clayey",
        "temperature": "20°C - 30°C",
        "description": "Rice grows best in warm and humid climates with abundant rainfall."
    },

    "maize": {
        "season": "Summer",
        "water": "Moderate",
        "soil": "Loamy",
        "temperature": "18°C - 27°C",
        "description": "Maize requires fertile, well-drained soil and moderate rainfall."
    },

    "chickpea": {
        "season": "Winter",
        "water": "Low",
        "soil": "Sandy Loam",
        "temperature": "20°C - 25°C",
        "description": "Chickpea grows best in cool and dry climatic conditions."
    },

    "kidneybeans": {
        "season": "Winter",
        "water": "Moderate",
        "soil": "Loamy",
        "temperature": "15°C - 25°C",
        "description": "Kidney beans require fertile, well-drained soils with moderate moisture."
    },

    "pigeonpeas": {
        "season": "Summer",
        "water": "Moderate",
        "soil": "Loamy",
        "temperature": "25°C - 35°C",
        "description": "Pigeon peas tolerate drought and improve soil fertility."
    },

    "mothbeans": {
        "season": "Summer",
        "water": "Low",
        "soil": "Sandy",
        "temperature": "24°C - 32°C",
        "description": "Moth beans are highly drought resistant and suitable for dry regions."
    },

    "mungbean": {
        "season": "Summer",
        "water": "Moderate",
        "soil": "Loamy",
        "temperature": "25°C - 35°C",
        "description": "Mung bean grows well in warm climates with moderate rainfall."
    },

    "blackgram": {
        "season": "Summer",
        "water": "Moderate",
        "soil": "Clay Loam",
        "temperature": "25°C - 35°C",
        "description": "Black gram prefers warm weather and fertile soils."
    },

    "lentil": {
        "season": "Winter",
        "water": "Low",
        "soil": "Loamy",
        "temperature": "18°C - 25°C",
        "description": "Lentils thrive in cool climates with well-drained soils."
    },

    "pomegranate": {
        "season": "Summer",
        "water": "Moderate",
        "soil": "Loamy",
        "temperature": "20°C - 35°C",
        "description": "Pomegranate is drought tolerant and suitable for semi-arid regions."
    },

    "banana": {
        "season": "All Seasons",
        "water": "High",
        "soil": "Loamy",
        "temperature": "26°C - 30°C",
        "description": "Banana requires high humidity and rich fertile soil."
    },

    "mango": {
        "season": "Summer",
        "water": "Moderate",
        "soil": "Loamy",
        "temperature": "24°C - 30°C",
        "description": "Mango grows well in tropical climates with good sunlight."
    },

    "grapes": {
        "season": "Summer",
        "water": "Moderate",
        "soil": "Sandy Loam",
        "temperature": "20°C - 30°C",
        "description": "Grapes require dry weather during fruit development."
    },

    "watermelon": {
        "season": "Summer",
        "water": "Moderate",
        "soil": "Sandy Loam",
        "temperature": "22°C - 30°C",
        "description": "Watermelon grows well under warm temperatures and sunlight."
    },

    "muskmelon": {
        "season": "Summer",
        "water": "Moderate",
        "soil": "Sandy Loam",
        "temperature": "25°C - 35°C",
        "description": "Muskmelon prefers warm temperatures and well-drained soil."
    },

    "apple": {
        "season": "Winter",
        "water": "Moderate",
        "soil": "Loamy",
        "temperature": "15°C - 24°C",
        "description": "Apple trees require cool climates and fertile well-drained soils."
    },

    "orange": {
        "season": "Winter",
        "water": "Moderate",
        "soil": "Loamy",
        "temperature": "20°C - 30°C",
        "description": "Orange grows best in subtropical climates."
    },

    "papaya": {
        "season": "All Seasons",
        "water": "Moderate",
        "soil": "Loamy",
        "temperature": "22°C - 30°C",
        "description": "Papaya grows rapidly in warm and humid climates."
    },

    "coconut": {
        "season": "All Seasons",
        "water": "High",
        "soil": "Sandy",
        "temperature": "25°C - 32°C",
        "description": "Coconut requires coastal climates with high humidity."
    },

    "cotton": {
        "season": "Summer",
        "water": "Moderate",
        "soil": "Black Soil",
        "temperature": "21°C - 30°C",
        "description": "Cotton grows well in black soils with warm weather."
    },

    "jute": {
        "season": "Rainy",
        "water": "High",
        "soil": "Alluvial",
        "temperature": "24°C - 35°C",
        "description": "Jute requires warm and humid climates with heavy rainfall."
    },

    "coffee": {
        "season": "Winter",
        "water": "Moderate",
        "soil": "Loamy",
        "temperature": "18°C - 24°C",
        "description": "Coffee grows best in cool tropical regions with shade."
    }

}

        return render_template(
    "prediction.html",
    prediction_text=result,
    crop=crop_info.get(result.lower(), None)
)

    except Exception as e:
        return render_template(
            "prediction.html",
            prediction_text=f"Error: {str(e)}"
        )


if __name__ == "__main__":
    app.run(debug=True)