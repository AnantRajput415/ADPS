import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input

# Paths
preprocessed_image_dir = "preprocessed_images"
preprocessed_label_file = "preprocessed_labels.csv"

# Load labels
labels = pd.read_csv(preprocessed_label_file)

# Dataset splitting
train_data, test_data = train_test_split(labels, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

# Data generators
def data_generator(data, batch_size, image_dir):
    while True:
        for start in range(0, len(data), batch_size):
            end = min(start + batch_size, len(data))
            batch_data = data[start:end]
            
            images = []
            targets = []
            for _, row in batch_data.iterrows():
                image_path = os.path.join(image_dir, row["image_name"])
                image = tf.keras.preprocessing.image.load_img(image_path, target_size=(96, 96))
                image = tf.keras.preprocessing.image.img_to_array(image) / 255.0
                
                # Targets: car + road bounding box
                target = row[1:].values.astype(np.float32)  # Skip the image name column
                images.append(image)
                targets.append(target)
            
            yield np.array(images), np.array(targets)

batch_size = 32
train_generator = data_generator(train_data, batch_size, preprocessed_image_dir)
val_generator = data_generator(val_data, batch_size, preprocessed_image_dir)
test_generator = data_generator(test_data, batch_size, preprocessed_image_dir)

# CNN Model
def create_model():
    model = Sequential([
        Input(shape=(96, 96, 3)),  # Use Input layer here
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(8, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

model = create_model()

# Model summary
model.summary()

# Training
steps_per_epoch = len(train_data) // batch_size
validation_steps = len(val_data) // batch_size

history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=20,
    validation_data=val_generator,
    validation_steps=validation_steps
)

# Save the model
model.save("car_road_bbox_model.h5")
print("Model training complete and saved as 'car_road_bbox_model.h5'.")
