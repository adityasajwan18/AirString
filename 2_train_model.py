# 2_train_model.py
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import pickle

DATA_DIR = "data"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Load all CSVs
dfs = []
for f in os.listdir(DATA_DIR):
    if f.endswith(".csv"):
        df = pd.read_csv(os.path.join(DATA_DIR, f), header=None)
        dfs.append(df)

data = pd.concat(dfs, ignore_index=True)
print(f"Total samples: {len(data)}")
print(f"Class distribution:\n{data[0].value_counts()}")

X = data.iloc[:, 1:].values   # 63 landmark features
y = data.iloc[:, 0].values    # chord labels

le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_cat = tf.keras.utils.to_categorical(y_encoded)

X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

# Build model
model = Sequential([
    Dense(128, activation='relu', input_shape=(63,)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(len(le.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=100, batch_size=32,
                    validation_data=(X_test, y_test), callbacks=[es])

loss, acc = model.evaluate(X_test, y_test)
print(f"\nTest accuracy: {acc:.2%}")

# Save model and label encoder
model.save(os.path.join(MODEL_DIR, "chord_model.h5"))
with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "wb") as f:
    pickle.dump(le, f)

print("Model and encoder saved!")