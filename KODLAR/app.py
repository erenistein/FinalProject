from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from joblib import load
import os
import google.generativeai as genai
from dotenv import load_dotenv

# .env'den API anahtarını yükle
load_dotenv()

# Gemini API anahtarı ayarı
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app = Flask(__name__)

# Model ve semptom listesi yükle
model = load("model/random_forest_model.joblib")
symptom_list = load("model/symptom_list.joblib")

# Gemini AI'dan hastalık hakkında öneriler alınan fonksiyon
def get_gemini_response(disease_name):
    prompt = f"""
    Kullanıcının hastalığı: {disease_name}.
    Bu hastalık hakkında basit, tıbbi olmayan yaşam önerilerini 5 madde halinde sade bir dille açıklar mısın?
    """

    try:
        model = genai.GenerativeModel(model_name="gemini-pro")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Gemini cevabı alınamadı: {str(e)}"


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    chat_user = ""
    chat_bot = ""

    if request.method == "POST":
        selected = request.form.getlist("symptoms")
        input_vector = [1 if s in selected else 0 for s in symptom_list]
        df_input = pd.DataFrame([input_vector], columns=symptom_list)
        prediction = model.predict(df_input)[0]

        # Chat mesajları
        chat_user = f"👤 Kullanıcı: Belirtilerime göre hangi hastalığım var?"
        chat_bot = get_gemini_response(prediction)

    return render_template("index.html", symptoms=symptom_list, prediction=prediction,
                           chat_user=chat_user, chat_bot=chat_bot)

if __name__ == "__main__":
    app.run(debug=True)
