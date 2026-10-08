# IMDB Movie Review Sentiment Analysis using 1D CNN

## Project Overview
This repository implements a 1D Convolutional Neural Network (CNN) in TensorFlow/Keras to perform binary sentiment analysis on the IMDB movie reviews dataset (classifying reviews as **Positive** or **Negative**).

While CNNs are traditionally used in computer vision, 1D CNNs serve as fast, effective phrase-level feature extractors for natural language text sequences.

---

## Architecture & Design
1. **Embedding Layer**: Converts sequence word indices (vocabulary size = 10,000) into 128-dimensional dense vectors.
2. **1D Convolutional Layer (`Conv1D`)**: Uses 128 filters of size `kernel_size=3` to extract 3-gram feature patterns along the word sequence.
3. **Global Max Pooling (`GlobalMaxPooling1D`)**: Extracts the dominant activation feature across time steps, isolating strong sentiment phrases regardless of sentence position.
4. **Dense & Dropout Layers**: A 64-unit dense layer with a `0.5` dropout rate to prevent overfitting.
5. **Output Layer**: A single neuron with a Sigmoid activation function outputting class probability $P(y=1\vert{}X)$.

---

## Experimental Setup & Results

| Metric | Score |
| :--- | :--- |
| **Test Accuracy** | ~87.5% - 88.5% |
| **Test Loss** | ~0.28 - 0.32 |
| **Training Time** | ~15-20 seconds per epoch (CPU/GPU) |

---

## How to Run

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
