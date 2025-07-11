from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# 📦 Load model and tokenizer
model = tf.keras.models.load_model("spam_model.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = ""
    email_text = ""
    if request.method == "POST":
        email_text = request.form["email"]
        seq = tokenizer.texts_to_sequences([email_text])
        pad = pad_sequences(seq, maxlen=50, padding="post")
        pred = model.predict(pad)[0][0]
        prediction = "Spam ❌" if pred > 0.5 else "Ham ✅"
    return render_template("index.html", prediction=prediction, email=email_text)

if __name__ == "__main__":
    import webbrowser
    edge_path = "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe %s"
    webbrowser.get(edge_path).open("http://127.0.0.1:5000")
    app.run()
