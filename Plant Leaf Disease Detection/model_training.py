import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set constants
img_size = 224
batch_size = 32
epochs = 50
train_path = 'dataset/train'
test_path = 'dataset/test'

# Model Definition
def create_model(input_shape=(img_size, img_size, 3), num_classes=8):  # 8 classes
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))  # 8 classes
    return model


# Plot Training History
def plot_training_history(history):
    """
    Plot training and validation accuracy and loss curves.
    """
    plt.figure(figsize=(12, 4))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss', marker='o')
    plt.plot(history.history['val_loss'], label='Validation Loss', marker='o')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    # Show the plots
    plt.show()

    # Save the plot
    plt.savefig('training_history.png')
    print("Plot saved as 'training_history.png'")

# Data Preparation
def prepare_data():
    # Augment the training data
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    # Augment the test data to increase diversity for evaluation
    test_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    # Load the training and validation data
    train_generator = train_datagen.flow_from_directory(
        'dataset/train',
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        'dataset/train',
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    # Load the test data
    test_generator = test_datagen.flow_from_directory(
        'dataset/test',
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    return train_generator, validation_generator, test_generator


# Training the Model
def train_model():
    train_generator, validation_generator, test_generator = prepare_data()
    model = create_model(input_shape=(img_size, img_size, 3), num_classes=len(train_generator.class_indices))

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size
    )

    print("Evaluating model...")
    val_loss, val_accuracy = model.evaluate(validation_generator, steps=validation_generator.samples // batch_size)
    print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

    print("Testing model...")
    test_loss, test_accuracy = model.evaluate(test_generator, steps=test_generator.samples // batch_size)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    # Plot training history
    plot_training_history(history)

    # Save the model
    model.save('plant_disease_model.h5')
    print("Model saved as 'plant_disease_model.h5'")

if __name__ == "__main__":
    train_model()
