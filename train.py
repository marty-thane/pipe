import numpy as np
from scipy.io import wavfile
from tensorflow.keras import models, layers

# Tweaking parameters
EPOCHS = 50
BATCH_SIZE = 32

# Load training data
_, X_train = wavfile.read("in.wav")
_, y_train = wavfile.read("out.wav")

# Reshape to what layers expect
X_train = X_train.reshape(-1,1,1)
y_train = y_train.reshape(-1,1)

# Define model architecture
model = models.Sequential([
    layers.LSTM(20, input_shape=(1, 1)),
    layers.Dense(1),
])

# Compile model (maybe change optimizer?)
model.compile(optimizer='adam', loss='mean_squared_error')

# Train on data
model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)

# Save model to file
# model.save('model.h5')
