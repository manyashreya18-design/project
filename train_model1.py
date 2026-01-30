import pandas as pd
import pickle
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
from sklearn.model_selection import train_test_split
from utils import normalize_url

# =========================
# SET YOUR DATASET NAME HERE
# =========================
DATASET_PATH = "all_urls_dataset.csv"   # <-- change only this

# =========================
# LOAD DATASET
# =========================
data = pd.read_csv(DATASET_PATH)

# REQUIRED columns:
# url   -> website URL
# label -> 0 (legit) , 1 (phishing)

urls = data["url"].astype(str).values
labels = data["label"].values

# =========================
# TOKENIZATION (CHAR LEVEL) ON NORMALIZED URLS
# =========================
normalized_urls = []
for url in urls:
    domain, normalized, _ = normalize_url(url)
    normalized_urls.append(normalized)

tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(normalized_urls)

X = tokenizer.texts_to_sequences(normalized_urls)
X = pad_sequences(X, maxlen=100)

y = labels

# save tokenizer (used by both models)
pickle.dump(tokenizer, open("tokenizer.pkl", "wb"))

# =========================
# TRAIN / TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Save test data for evaluation
# Since train_test_split shuffles, we need to save the test indices
indices = np.arange(len(urls))
_, test_indices = train_test_split(indices, test_size=0.2, random_state=42)
test_urls = urls[test_indices]
test_labels = labels[test_indices]
test_data = {"X_test": X_test, "y_test": y_test, "urls": test_urls, "labels": test_labels}
pickle.dump(test_data, open("test_data.pkl", "wb"))

# =========================
# CNN MODEL
# =========================
model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128, input_length=100),
    Conv1D(128, 5, activation="relu"),
    GlobalMaxPooling1D(),
    Dense(64, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=64,
    validation_data=(X_test, y_test)
)

model.save("model1_url.h5")

print("✅ CNN model trained using:", DATASET_PATH)
