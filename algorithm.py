# Import necessary libraries
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten

# Load the dataset
(X_train, y_train), (X_test, y_test) = load_data()

# Preprocess the data
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

# Define the model architecture
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(28, 28)))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print('Loss:', loss)
print('Accuracy:', accuracy)

# Use the model to make predictions on new data
predictions = model.predict(new_data)
