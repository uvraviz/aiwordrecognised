# Importing necessary libraries and modules
import cv2
import numpy as np
from itertools import groupby
import tensorflow as tf
from config import ModelConfig
import pandas as pd
from sklearn.metrics import accuracy_score

# Defining a function to decode the predictions from the model
def ctc_decoder(predictions, chars) :
    # Getting the index of the maximum prediction for each time step
    argmax_preds = np.argmax(predictions, axis= -1)
    # Grouping the predictions by their values
    grouped_preds = [[preds for preds in argmax_preds]]
    # Converting the grouped predictions to text
    texts = ["".join([chars[k] for k in group if k < len(chars)]) for group in grouped_preds]
    return texts

# Loading the configurations for the model
unixTime = 1714887559
configFilePath = f"Models/Handwriting_recognition/{unixTime}/configs.meow"
configs = ModelConfig().load(configFilePath)

# Enabling unsafe deserialization
tf.keras.config.enable_unsafe_deserialization()

# Loading the trained model
model = tf.keras.models.load_model(f"{configs.model_path}/model.keras", compile= False)

# Defining a function to recognize the text in an image
def recog(img, model, config) :
    if img is not None:
        # Resizing the image to the input size of the model
        img = cv2.resize(img, (128, 32), interpolation= cv2.INTER_AREA)
        # Making a prediction with the model
        preds = model.predict(np.array([img]))[0]
        # Decoding the prediction to text
        text = ctc_decoder(preds, config.vocab)[0]
        return text

# Reading the validation data
val_data = pd.read_csv('Models/Handwriting_recognition/1714887559/val.csv', header=None)

# Initializing counters for the number of correct and total predictions
correct_predictions = 0
total_predictions = 0

# Limiting the validation data to the first 500 rows
val_data = val_data.iloc[:5]

# Iterating over the rows in the validation data
for _, row in val_data.iterrows():
    # Getting the image path and the original word
    img_path, original_word = row[0], row[1]
    # Reading the image
    img = cv2.imread(img_path)
    # Using the model to predict the word in the image
    pred_word = recog(img, model, configs)
    # Printing the predicted word and the original word
    print(f'Predicted: {pred_word}, Original: {original_word}')
    # Updating the counters
    total_predictions += 1
    if pred_word == original_word:
        correct_predictions += 1

# Calculating and printing the accuracy of the model
accuracy = correct_predictions / total_predictions
print(f'Accuracy: {accuracy}')
