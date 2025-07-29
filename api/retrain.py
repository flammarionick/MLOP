import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def retrain_cnn_model(data_dir, model_path="models/best_model.h5", epochs=5):
    image_size = (100, 100)
    batch_size = 32

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found.")

    # Load existing model
    model = load_model(model_path)

    # Prepare data generators with grayscale images
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_gen = datagen.flow_from_directory(
        os.path.join(data_dir,),
        target_size=image_size,
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    val_gen = datagen.flow_from_directory(
        os.path.join(data_dir,),  # Note: validation split from 'train'
        target_size=image_size,
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    # Compile and retrain the model
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_gen, validation_data=val_gen, epochs=epochs)

    # Save the updated model
    model.save(model_path)