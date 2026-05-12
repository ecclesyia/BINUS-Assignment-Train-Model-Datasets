# Importing
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

FILE_NAME = "anilist_data.csv"
TARGET_COLUMN = "Status"

def train_anilist_model():
    try:
        print("Membaca data...")
        df = pd.read_csv(FILE_NAME)
        
        # 1. Peak any column that be use
        # Feature: Score & Numerical
        # Target: Status
        cols_to_use = ['Score', 'Episodes', TARGET_COLUMN]
        df = df[cols_to_use]

        # 2. Handling inputing contain NaN or a file that's do not have a value
        initial_count = len(df)
        df = df.dropna()
        print(f"Menghapus {initial_count - len(df)} baris yang mengandung nilai kosong (NaN).")

        # 3. Splitting feature and target
        X = df[['Score', 'Episodes']]
        y = df[TARGET_COLUMN]

        # 4. Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 5. Pipeline
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(n_estimators=100, random_state=42))
        ])

        print(f"Melatih model untuk memprediksi '{TARGET_COLUMN}'...")
        pipeline.fit(X_train, y_train)

        # 6. Evaluation & Save on pkl
        y_pred = pipeline.predict(X_test)
        print(f"Akurasi Model: {accuracy_score(y_test, y_pred):.2f}")
        
        joblib.dump(pipeline, "anilist_model.pkl")
        X_test.to_csv("test_samples.csv", index=False)
        print("Selesai! Model disimpan sebagai 'anilist_model.pkl'")

    except Exception as e:
        print(f"Gagal: {e}")

if __name__ == "__main__":
    train_anilist_model()
