from flask import Flask, render_template, request, jsonify
import pickle

app = Flask(__name__)
import os

if not os.path.exists("model/sentiment_model.pkl"):
    raise FileNotFoundError(
        "Model file not found. Please train the model first using train_model.py"
    )


# Load trained model
with open("model/sentiment_model.pkl", "rb") as f:
    model, vectorizer = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data["text"]

    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)[0]

    result = "Positive 😊" if prediction == 1 else "Negative 😡"
    return jsonify({"sentiment": result})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)


