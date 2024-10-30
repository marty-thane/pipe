import sys
from scipy.io import wavfile
from tensorflow.keras import models, layers

# Tweaking parameters
OPTIMIZER="sgd"
LOSS="mean_squared_error"
EPOCHS = 10
BATCH_SIZE = 32

# Load training data
_, X_train = wavfile.read(sys.argv[1])
_, y_train = wavfile.read(sys.argv[2])

# Reshape to what layers expect
X_train = X_train.reshape(-1,1,1)
y_train = y_train.reshape(-1,1)

# Define model architecture
model = models.Sequential([
    layers.LSTM(20, input_shape=(1, 1)),
    layers.Dense(1),
])

# Compile model
model.compile(optimizer=OPTIMIZER, loss=LOSS)

# Train on data
model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)

# Save model to file
# model.save("model.h5")
