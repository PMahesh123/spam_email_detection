import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pickle

# ✅ More realistic spam + ham messages
data = {
    'message': [
        # Spam
        "Congratulations! You've won a free iPhone.",
        "Your bank account is locked. Verify now.",
        "Claim ₹1,00,000 now. Click the link!",
        "Urgent: Pay your bill to avoid disconnection.",
        "You are selected for a free lottery.",
        "Get rich quick! Join today!",
        "Click here to receive your prize.",
        "You've won a cash reward. Act now.",
        "Exclusive offer! Limited time only.",
        "Win big money now!",
        # Ham
        "Hey, are you free for a call tomorrow?",
        "Can you send me the notes from class?",
        "Let's go out for lunch at 2?",
        "Meeting at 10 AM. Don't be late.",
        "I'm stuck in traffic, will be late.",
        "Assignment deadline is tomorrow.",
        "Call mom when you're done with work.",
        "Bring your ID card for the exam.",
        "Want to hang out this weekend?",
        "Don’t forget the groceries."
    ],
    'label': [1]*10 + [0]*10
}

df = pd.DataFrame(data)

X = df['message'].values
y = df['label'].values

tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(X)
sequences = tokenizer.texts_to_sequences(X)
padded = pad_sequences(sequences, maxlen=50, padding='post')

X_train, X_test, y_train, y_test = train_test_split(padded, y, test_size=0.2, random_state=42)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(5000, 16, input_length=50),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), verbose=0)

model.save("spam_model.h5")
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("✅ Model trained and saved successfully.")

