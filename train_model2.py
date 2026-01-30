import pandas as pd
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense
from sklearn.model_selection import train_test_split

# =========================
# SET YOUR DATASET NAME HERE
# =========================
DATASET_PATH = "all_urls_dataset.csv"   # <-- same file as model1

# =========================
# LOAD DATASET
# =========================
data = pd.read_csv(DATASET_PATH)

urls = data["url"].astype(str).values
labels = data["label"].values

# =========================
# LOAD SAME TOKENIZER
# =========================
tokenizer = pickle.load(open("tokenizer.pkl", "rb"))

X = tokenizer.texts_to_sequences(urls)
X = pad_sequences(X, maxlen=100)

y = labels

# =========================
# TRAIN / TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# BiLSTM MODEL
# =========================
model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128, input_length=100),
    Bidirectional(LSTM(64)),
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

model.save("model2_bilstm.h5")

print("✅ BiLSTM model trained using:", DATASET_PATH)
