# Importing necessary libraries
import os
import time
from pickle import Pickler, Unpickler

# Defining a class ModelConfig
class ModelConfig() :
    # Constructor for the class
    def __init__(self) :
        # Getting the current time as an integer
        t = int(time.time())
        # Setting the path to save the model
        self.model_path = f"Models/Handwriting_recognition/{t}"
        # Initializing the vocabulary
        self.vocab = ''
        # Setting the height and width of the images
        self.height = 32
        self.width = 128
        # Initializing the maximum text length
        self.max_text_length = 0
        # Setting the batch size for training
        self.batch_size = 16
        # Setting the learning rate for training
        self.learning_rate = 0.0005
        # Setting the number of epochs for training
        self.train_epochs = 100
        # Setting the split ratio for validation data
        self.validation_split = 0.9

    # Defining a method to save the configurations as a file
    def save(self) :
        # Creating the directory for the model path
        os.makedirs(self.model_path)
        # Setting the file name for the configurations
        file_name = f"{self.model_path}/configs.meow"
        
        # Opening the file in write mode and saving the configurations using Pickler
        with open(file_name, 'wb') as wf :
            Pickler(wf).dump(self)

    # Defining a method to load the configurations from a file
    def load(self, filePath) :
        # Opening the file in read mode and loading the configurations using Unpickler
        with open(filePath, 'rb') as rf :
            config = Unpickler(rf).load()
            return config
