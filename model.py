# Importing the tensorflow library
import tensorflow as tf

# Defining a function for a residual block
def residual_block(
        x,
        filter_num,
        strides = 2,
        kernel_size = 3,
        skip_convo = True,
        padding = 'same',
        kernel_initializer = 'he_uniform',
        dropout = 0.2
    ) :
        # Create a skip connection tensor
        x_skip = x

        # Perform the first convolution
        x = tf.keras.layers.Conv2D(filter_num, kernel_size, padding = padding, strides = strides, kernel_initializer = kernel_initializer)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(0.1)(x)

        # Perform the second convolution
        x = tf.keras.layers.Conv2D(filter_num, kernel_size, padding = padding, kernel_initializer = kernel_initializer)(x)
        x = tf.keras.layers.BatchNormalization()(x)

        # If skip_convo is True, perform a third convolution to match the number of filters and the shape of the skip connection tensor
        if skip_convo :
              x_skip = tf.keras.layers.Conv2D(filter_num, 1, padding = padding, strides = strides, kernel_initializer = kernel_initializer)(x_skip)

        # Add x and the skip connection layers and apply an activation function
        x = tf.keras.layers.Add()([x, x_skip])
        x = tf.keras.layers.LeakyReLU(0.1)(x)

        # Apply dropout if dropout is not None
        if dropout :
              x = tf.keras.layers.Dropout(dropout)(x)

        return x

# Defining a function to train the model
def train_model(input_dim, output_dim, dropout = 0.2) :

      # Define the input layer
      inputs = tf.keras.layers.Input(shape = input_dim, name = "input")

      # Normalize the images
      input = tf.keras.layers.Lambda(lambda x : x / 255)(inputs)

      # Define the architecture of the model using residual blocks
      x1 = residual_block(input, 16, skip_convo = True, strides = 1, dropout = dropout)
      x2 = residual_block(x1, 16, skip_convo = True, strides = 2, dropout = dropout)
      x3 = residual_block(x2, 16, skip_convo = False, strides = 1, dropout = dropout)
      x4 = residual_block(x3, 32, skip_convo = True, strides = 2, dropout = dropout)
      x5 = residual_block(x4, 32, skip_convo = False, strides = 1, dropout = dropout)
      x6 = residual_block(x5, 64, skip_convo = True, strides = 2, dropout = dropout)
      x7 = residual_block(x6, 64, skip_convo = True, strides = 1, dropout = dropout)
      x8 = residual_block(x7, 64, skip_convo = False, strides = 1, dropout = dropout)
      x9 = residual_block(x8, 64, skip_convo = False, strides = 1, dropout = dropout)

      # Reshape the output of the last residual block
      squeeze = tf.keras.layers.Reshape((x9.shape[-3] * x9.shape[-2], x9.shape[-1]))(x9)

      # Define a bidirectional LSTM layer
      blstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences = True))(squeeze)
      blstm = tf.keras.layers.Dropout(dropout)(blstm)

      # Define the output layer
      output = tf.keras.layers.Dense(output_dim + 1, activation = 'softmax', name = "output")(blstm)

      # Create the model
      model = tf.keras.Model(inputs = inputs, outputs = output)
      return model