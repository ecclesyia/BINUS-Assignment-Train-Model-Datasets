# Inference and Latency testing of model
import joblib
import time
import pandas as pd

# Loadding model
model = joblib.load("anilist_model.pkl")

print("Model loaded successfully!")

# Sampling Input

sample_data = pd.DataFrame([[8.5, 12]], columns=['Score', 'Episodes'])

# Measuring dataset
model.predict(sample_data)

start_time = time.time()

prediction = model.predict(sample_data)

end_time = time.time()

latency = (end_time - start_time) * 1000

# Output
print("\nPrediction:", prediction[0])
print(f"Latency: {latency:.4f} ms")
