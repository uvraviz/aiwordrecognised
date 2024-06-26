# Importing the tensorflow library
import tensorflow as tf

# Defining a class CTCloss which inherits from tf.keras.losses.Loss
class CTCloss(tf.keras.losses.Loss) :
    # Constructor for the class
    def __init__(self, name = 'CTCloss'):
        # Calling the constructor of the parent class
        super(CTCloss, self).__init__()
        # Setting the name of the loss function
        self.name = name
        # Setting the loss function to be the CTC (Connectionist Temporal Classification) batch cost
        self.loss_fn = tf.keras.backend.ctc_batch_cost

    # Defining the call method which calculates the loss
    def __call__(self, y_true, y_pred, sample_weight = None) -> tf.Tensor :
        # Getting the batch size
        batch_len = tf.cast(tf.shape(y_true)[0], dtype = "int64")
        # Getting the length of the input sequence
        input_length = tf.cast(tf.shape(y_pred)[1], dtype = "int64")
        # Getting the length of the label sequence
        label_length = tf.cast(tf.shape(y_pred)[1], dtype = "int64")

        # Creating tensors of size (batch_size, 1) for input and label lengths
        input_length = input_length * tf.ones(shape = (batch_len, 1), dtype= "int64")
        label_length = label_length * tf.ones(shape = (batch_len, 1), dtype= "int64")

        # Calculating the loss using the CTC batch cost function
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)

        # Returning the loss
        return loss

# Defining a class CWERMetric which inherits from tf.keras.metrics.Metric
class CWERMetric(tf.keras.metrics.Metric) :
    # Constructor for the class
    def __init__(self, padding_token, name='CWER', **kwargs) :
        # Calling the constructor of the parent class
        super(CWERMetric, self).__init__(name = name, **kwargs)

        # Initializing the accumulators for character error rate (CER) and word error rate (WER)
        self.cer_accumulator = tf.Variable(0.0, name= "cer_accumulator", dtype= tf.float32)
        self.wer_accumulator = tf.Variable(0.0, name= "wer_accumulator", dtype= tf.float32)
        # Initializing the counter for the number of batches
        self.batch_counter = tf.Variable(0, name= "batch_counter", dtype= tf.int32)

        # Setting the padding token
        self.padding_token = padding_token

    # Defining the method to update the state of the metric
    def update_state(self, y_true, y_pred, sample_weight = None) :
        # Getting the shape of the input
        input_shape = tf.keras.backend.shape(y_pred)
        # Creating a tensor of size (batch_size,) for the input length
        input_length = tf.ones(shape= input_shape[0], dtype='int32') * tf.cast(input_shape[1], 'int32')

        # Decoding the predicted sequence
        decode_predicted, log = tf.keras.backend.ctc_decode(y_pred, input_length, greedy=True)

        # Converting the dense labels to sparse format
        predicted_labels_sparse = tf.keras.backend.ctc_label_dense_to_sparse(decode_predicted[0], input_length)
        true_labels_sparse = tf.cast(tf.keras.backend.ctc_label_dense_to_sparse(y_true, input_length), "int64")

        # Removing the padding tokens from the predicted labels
        predicted_labels_sparse = tf.sparse.retain(predicted_labels_sparse, tf.not_equal(predicted_labels_sparse.values, -1))

        # Calculating the edit distance between the true and predicted labels
        distance = tf.edit_distance(predicted_labels_sparse, true_labels_sparse, normalize=True)

        # Updating the accumulators for CER and WER
        self.cer_accumulator.assign_add(tf.reduce_sum(distance))
        self.batch_counter.assign_add(tf.shape(y_true)[0])
        self.wer_accumulator.assign_add(tf.reduce_sum(tf.cast(tf.not_equal(distance, 0), tf.float32)))

    # Defining the method to get the result of the metric
    def result(self) :
        # Returning the CER and WER
        return {
            "CER" : tf.math.divide_no_nan(self.cer_accumulator, tf.cast(self.batch_counter, tf.float32)),
            "WER" : tf.math.divide_no_nan(self.wer_accumulator, tf.cast(self.batch_counter, tf.float32))
        }
    
# Importing necessary libraries
import numpy as np
import copy
import pandas as pd
import cv2
    
# Defining a class DataProvider which inherits from tf.keras.utils.Sequence
class DataProvider(tf.keras.utils.Sequence) :
    # Constructor for the class
    def __init__(
            self,
            dataset,
            vocab,
            max_text_length,
            batch_size = 16,
            initial_epoch = 1
    ) :
        # Initializing the dataset, vocabulary, maximum text length, batch size, epoch, and step
        self._dataset = dataset
        self._batch_size = batch_size
        self._epoch = initial_epoch
        self._step = 0
        self._vocab = vocab
        self._max_text_length = max_text_length

    # Defining the method to get the length of the sequence
    def __len__(self) :
        return int(np.ceil(len(self._dataset)/ self._batch_size))

    # Defining the method to get the current epoch
    @property
    def epoch(self) :
        return self._epoch
    
    # Defining the method to get the current step
    @property
    def step(self) :
        return self._step
    
    # Defining the method to increment the epoch at the end of each epoch
    def on_epoch_end(self) :
        self._epoch += 1

    # Defining the method to split the dataset into training and validation sets
    def split(self, split = 0.9, shuffle = True) :
        if shuffle :
            np.random.shuffle(self._dataset)

        # Creating copies of the current object for the training and validation data
        train_data = copy.deepcopy(self)
        val_data = copy.deepcopy(self)
        # Splitting the dataset into training and validation sets
        train_data._dataset = self._dataset[:int(len(self._dataset) * split)]
        val_data._dataset = self._dataset[int(len(self._dataset) * split):]

        return train_data, val_data
    
    # Defining the method to save the dataset to a CSV file
    def to_csv(self, path, index = False) :
        df = pd.DataFrame(self._dataset)
        df.to_csv(path, index= index)

    # Defining the method to get the annotations for a batch
    def get_batch_annotations(self, index) :
        self._step = index
        start_index = index * self._batch_size

        # Getting the indexes for the batch
        batch_indexes = [i for i in range(start_index, start_index + self._batch_size) if i < len(self._dataset)]

        # Getting the annotations for the batch
        batch_annotations = [self._dataset[index] for index in batch_indexes]

        return batch_annotations
    
    # Defining the method to resize the image
    def ImageResizer(self, data, label) :
        return cv2.resize(data, (128, 32), interpolation= cv2.INTER_AREA), label
    
    # Defining the method to index the labels
    def LabelIndexer(self, data, label) :
        return data, np.array([self._vocab.index(l) for l in label if l in self._vocab])
    
    # Defining the method to pad the labels to a uniform length
    def LabelPadding(self, data, label) :
        return data, np.pad(label, (0, self._max_text_length - len(label)), 'constant', constant_values= len(self._vocab))
    
    # Defining the method to get an item from the sequence
    def __getitem__(self, index) :
        # Getting the annotations for the batch
        dataset_batch = self.get_batch_annotations(index)

        batch_data, batch_annotations = [], []
        for index, (data, annotation) in enumerate(dataset_batch) :
            # Reading the image
            data = cv2.imread(data, cv2.IMREAD_COLOR)

            # If the image is None, remove it from the dataset
            if data is None :
                self._dataset.remove(dataset_batch[index])
                continue

            # Appending the image and annotation to the batch
            batch_data.append(data)
            batch_annotations.append(annotation)

        # Applying the transformers to the batch
        batch_data, batch_annotations = zip(*[self.ImageResizer(data, annotation) for data, annotation in zip(batch_data, batch_annotations)])
        batch_data, batch_annotations = zip(*[self.LabelIndexer(data, annotation) for data, annotation in zip(batch_data, batch_annotations)])
        batch_data, batch_annotations = zip(*[self.LabelPadding(data, annotation) for data, annotation in zip(batch_data, batch_annotations)])

        return np.array(batch_data), np.array(batch_annotations)
