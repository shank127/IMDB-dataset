"""
Project Title: Movie Review Sentiment Analysis Using CNN
Dataset: IMDB Movie Reviews (Keras)
Framework: TensorFlow / Keras
"""

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix

# ==========================================
# 1. Hyperparameters & Configuration
# ==========================================
VOCAB_SIZE = 10000      # Consider top 10,000 most frequent words
MAX_LEN = 500           # Standardize review length to 500 words
EMBEDDING_DIM = 128     # Dimension of word embedding vectors
NUM_FILTERS = 128       # Number of 1D conv filters
KERNEL_SIZE = 3         # Filter size (captures 3-gram word phrases)
BATCH_SIZE = 64
EPOCHS = 10
SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)


# ==========================================
# 2. Dataset Loading & Preprocessing
# ==========================================
def load_and_preprocess_imdb():
    """Load the IMDB dataset and pad word sequences to uniform length."""
    print("Loading IMDB Movie Review dataset...")
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=VOCAB_SIZE)
    
    print(f"Total Training Reviews: {len(X_train)}")
    print(f"Total Test Reviews:     {len(X_test)}")
    
    # Pad or truncate sequences to a fixed length of 500 words
    X_train_padded = pad_sequences(X_train, maxlen=MAX_LEN, padding='post', truncating='post')
    X_test_padded = pad_sequences(X_test, maxlen=MAX_LEN, padding='post', truncating='post')
    
    print(f"Shape of Padded Train Data: {X_train_padded.shape}")
    print(f"Shape of Padded Test Data:  {X_test_padded.shape}")
    
    return X_train_padded, y_train, X_test_padded, y_test


# ==========================================
# 3. Model Architecture Construction
# ==========================================
def build_movie_sentiment_cnn():
    """Construct and compile the 1D CNN Architecture."""
    model = Sequential([
        # 1. Word Embedding Layer
        Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, input_length=MAX_LEN),
        
        # 2. 1D Convolutional Layer (Extracts phrase-level features)
        Conv1D(filters=NUM_FILTERS, kernel_size=KERNEL_SIZE, activation='relu'),
        
        # 3. Global Max Pooling (Captures strongest sentiment signals)
        GlobalMaxPooling1D(),
        
        # 4. Dense Layer with Regularization
        Dense(64, activation='relu'),
        Dropout(0.5),
        
        # 5. Output Layer (Sigmoid activation for binary classification)
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


# ==========================================
# 4. Training History Visualization
# ==========================================
def plot_metrics(history):
    """Plot Loss and Accuracy curves for Training vs Validation."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss Plot
    ax1.plot(history.history['loss'], label='Train Loss', color='navy', linewidth=2)
    ax1.plot(history.history['val_loss'], label='Val Loss', color='crimson', linewidth=2)
    ax1.set_title('Loss Trajectory over Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Accuracy Plot
    ax2.plot(history.history['accuracy'], label='Train Accuracy', color='navy', linewidth=2)
    ax2.plot(history.history['val_accuracy'], label='Val Accuracy', color='crimson', linewidth=2)
    ax2.set_title('Accuracy Trajectory over Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('cnn_movie_sentiment_training.png', dpi=300)
    print("\nTraining curves successfully saved to 'cnn_movie_sentiment_training.png'.")


# ==========================================
# 5. Inference Utility for Custom Reviews
# ==========================================
def predict_movie_review_sentiment(review_text, model, maxlen=MAX_LEN):
    """Clean, tokenize, and classify sentiment for any custom text review."""
    word_index = imdb.get_word_index()
    
    # Clean and split raw review text into lowercase words
    clean_text = review_text.lower().replace('.', '').replace(',', '').replace('!', '').replace('?', '')
    words = clean_text.split()
    
    # Map words to IMDB index offset
    encoded = []
    for word in words:
        if word in word_index and word_index[word] + 3 < VOCAB_SIZE:
            encoded.append(word_index[word] + 3)
        else:
            encoded.append(2)  # Index 2 represents Out-of-Vocabulary (OOV)
            
    padded = pad_sequences([encoded], maxlen=maxlen, padding='post', truncating='post')
    raw_score = model.predict(padded, verbose=0)[0][0]
    
    sentiment = "POSITIVE" if raw_score >= 0.5 else "NEGATIVE"
    confidence = raw_score if raw_score >= 0.5 else 1.0 - raw_score
    return sentiment, raw_score, confidence


# ==========================================
# 6. Main Program Execution
# ==========================================
if __name__ == "__main__":
    # Step 1: Data Preparation
    X_train, y_train, X_test, y_test = load_and_preprocess_imdb()
    
    # Step 2: Model Definition & Summary
    cnn_model = build_movie_sentiment_cnn()
    print("\n" + "="*50)
    print("MODEL ARCHITECTURE")
    print("="*50)
    cnn_model.summary()
    
    # Step 3: Setup Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True),
        ModelCheckpoint('movie_sentiment_cnn_best.h5', monitor='val_loss', save_best_only=True)
    ]
    
    # Step 4: Model Training
    print("\n" + "="*50)
    print("TRAINING CNN MODEL")
    print("="*50)
    history = cnn_model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    
    # Plot performance graphs
    plot_metrics(history)
    
    # Step 5: Test Set Evaluation
    print("\n" + "="*50)
    print("EVALUATING MODEL ON TEST DATA")
    print("="*50)
    test_loss, test_acc = cnn_model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Set Loss:     {test_loss:.4f}")
    print(f"Test Set Accuracy: {test_acc * 100:.2f}%")
    
    # Detailed Precision, Recall, F1-Score, and Confusion Matrix
    y_pred_probs = cnn_model.predict(X_test, verbose=0)
    y_pred = (y_pred_probs >= 0.5).astype(int)
    
    print("\n--- Detailed Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=['Negative Review', 'Positive Review']))
    
    print("--- Confusion Matrix ---")
    print(confusion_matrix(y_test, y_pred))
    
    # Step 6: Test Custom Movie Reviews
    print("\n" + "="*50)
    print("LIVE CUSTOM MOVIE REVIEW PREDICTIONS")
    print("="*50)
    
    sample_movie_reviews = [
        "This film was an absolute cinematic masterpiece with outstanding acting, breathtaking visuals, and a brilliant direction!",
        "The story was boring, the acting felt totally forced, and the pacing was terribly slow. I wanted to leave the theater early."
    ]
    
    for review in sample_movie_reviews:
        sentiment, score, confidence = predict_movie_review_sentiment(review, cnn_model)
        print(f"\nReview Input: \"{review}\"")
        print(f"Predicted Class: {sentiment} | Score: {score:.4f} | Confidence: {confidence * 100:.2f}%")
