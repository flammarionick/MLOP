import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def retrain_cnn_model(data_dir, model_path="models/best_model.h5", epochs=5):
    image_size = (64, 64)
    batch_size = 32

    # Generate unique suffix for layer naming
    suffix = str(int(time.time()))

    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_gen = datagen.flow_from_directory(
        data_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    val_gen = datagen.flow_from_directory(
        data_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    num_classes = train_gen.num_classes

    # Load the base model
    old_model = load_model(model_path)

    # Build a new Sequential model without the last layer
    new_model = Sequential()

    for layer in old_model.layers[:-1]:  # exclude original final Dense
        layer.trainable = False
        new_model.add(layer)

    new_model.build((None, 64, 64, 3))

    # Add new output layers with dynamic names
    new_model.add(Dense(128, activation='relu', name=f"retrain_dense_{suffix}"))
    new_model.add(Dense(num_classes, activation='softmax', name=f"retrain_output_{suffix}"))

    # Compile
    new_model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

    # Train
    history = new_model.fit(train_gen, validation_data=val_gen, epochs=epochs)

    # Save model
    new_model.save(model_path)

    # Create output directory for visualizations
    os.makedirs("outputs", exist_ok=True)

    # 1. 📊 Class distribution plot
    labels, counts = np.unique(train_gen.classes, return_counts=True)
    class_names = list(train_gen.class_indices.keys())

    plt.figure(figsize=(10, 6))
    sns.barplot(x=class_names, y=counts)
    plt.title("Class Distribution in Training Data")
    plt.ylabel("Image Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("outputs/class_distribution.png")
    plt.close()

    # 2. 📈 Accuracy/Loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Training/Validation Accuracy & Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/training_curves.png")
    plt.close()

    # 3. 🔍 Confidence histogram
    val_gen.reset()
    predictions = new_model.predict(val_gen, verbose=0)
    confidences = np.max(predictions, axis=1)

    plt.figure(figsize=(10, 6))
    sns.histplot(confidences, bins=20, kde=True)
    plt.title("Prediction Confidence Histogram")
    plt.xlabel("Confidence")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("outputs/confidence_histogram.png")
    plt.close()

    print("✅ Retraining complete. Model saved. Visualizations saved in /outputs")