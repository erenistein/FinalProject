import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump

# Klasör oluştur
os.makedirs("model", exist_ok=True)

# Veriyi oku
df = pd.read_csv("training_data.csv")

# Özellikler ve hedefi ayır
X = df.drop("prognosis", axis=1)
y = df["prognosis"]

# Eğitim / test bölmesi
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model oluştur ve eğit
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Test doğruluğu
y_pred = model.predict(X_test)
print("Test Doğruluğu:", accuracy_score(y_test, y_pred))

# Modeli kaydet
dump(model, "model/random_forest_model.joblib")

# Semptom listesi de kaydedilsin (kolay kullanım için)
symptoms = list(X.columns)
dump(symptoms, "model/symptom_list.joblib")
