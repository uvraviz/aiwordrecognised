# Importing necessary libraries and modules
import tensorflow as tf
from config import ModelConfig
from model import train_model
from CustomTF import CTCloss, CWERMetric, DataProvider
from tqdm import tqdm
import os

# Initializing empty lists and variables for preprocessing
dataset = []
vocab = set()
max_len = 0

# Reading the words.txt file line by line
words = open("Datasets/words.txt", "r").readlines()
for line in tqdm(words) :
    # Skip lines that start with a hashtag as those are comments
    if line.startswith('#') :
        continue

    # Splitting the line into different components
    line_split = line.split(' ')

    # If the segmentation result is 'err', skip this line
    if line_split[1] == 'err' :
        continue

    # Extracting the folder names and file name from the line
    folder1 = line_split[0][:3]
    folder2 = line_split[0][:8]
    file_name = line_split[0] + ".png"
    label = line_split[-1].rstrip('\n')

    # Constructing the path to the image file
    path = f"Datasets/words/{folder1}/{folder2}/{file_name}"
    # If the file does not exist, skip this line
    if not os.path.exists(path) :
        continue

    # Appending the path and label to the dataset
    dataset.append([path, label])
    # Updating the vocabulary with the characters in the label
    vocab.update(list(label))
    # Updating the maximum length of the labels
    max_len = max(max_len, len(label))

# Creating a ModelConfig object to store the configurations
configs = ModelConfig()

# Saving the vocabulary and maximum text length in the configurations
configs.vocab = "".join(vocab)
configs.max_text_length = max_len
configs.save()

# Creating a DataProvider object for TensorFlow
data_provider = DataProvider(
    dataset,
    configs.vocab,
    configs.max_text_length,
)

# Splitting the data into training and validation sets
train_data_provider, val_data_provider = data_provider.split(split= configs.validation_split)

# Creating the model architecture
model = train_model(
    input_dim = (configs.height, configs.width, 3),
    output_dim = len(configs.vocab)
)

# Compiling the model and printing the summary
model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate = configs.learning_rate),
    loss = CTCloss(),
    metrics = [CWERMetric(padding_token = len(configs.vocab))]
)
model.summary(line_length = 110)

# Defining the callbacks for training
earlystopper = tf.keras.callbacks.EarlyStopping(monitor='val_CER', patience=20, verbose=1, mode='min')
checkpoint = tf.keras.callbacks.ModelCheckpoint(f"{configs.model_path}/model.keras", monitor='val_CER', verbose=1, save_best_only=True, mode='min')
tb_callback = tf.keras.callbacks.TensorBoard(f"{configs.model_path}/logs", update_freq= 1)
reduceLROnPlat = tf.keras.callbacks.ReduceLROnPlateau(monitor= 'val_CER', factor= 0.9, min_delta= 1e-10, patience= 10, verbose= 1, mode= 'auto')

# Training the model
model.fit(
    train_data_provider,
    validation_data = val_data_provider,
    epochs = configs.train_epochs,
    callbacks = [earlystopper, checkpoint, tb_callback, reduceLROnPlat],
)

# Saving the trained model
model.save(f"{configs.model_path}/model.keras")

# Saving the training and validation datasets as CSV files
train_data_provider.to_csv(f"{configs.model_path}/train.csv")
val_data_provider.to_csv(f"{configs.model_path}/val.csv")
