import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix

# Ensure the plots directory exists
plots_dir = "plots"
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

# Image Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0/255, 
    rotation_range=15, 
    width_shift_range=0.1, 
    height_shift_range=0.1, 
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    r"path_to_train", 
    target_size=(120, 120), 
    batch_size=16, 
    class_mode="categorical",
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    r"path_to_val", 
    target_size=(120, 120), 
    batch_size=16, 
    class_mode="categorical",
    shuffle=False  # Important for Confusion Matrix
)

# CNN Model
model = Sequential([
    Conv2D(32, (3,3), padding='same', input_shape=(120, 120, 3)),
    BatchNormalization(),
    LeakyReLU(alpha=0.1),
    MaxPooling2D(),

    Conv2D(64, (3,3), padding='same'),
    BatchNormalization(),
    LeakyReLU(alpha=0.1),
    MaxPooling2D(),

    Conv2D(128, (3,3), padding='same'),
    BatchNormalization(),
    LeakyReLU(alpha=0.1),
    MaxPooling2D(),

    Flatten(),
    Dense(128),
    LeakyReLU(alpha=0.1),
    Dropout(0.3),

    Dense(64),
    LeakyReLU(alpha=0.1),
    Dropout(0.3),

    Dense(train_generator.num_classes, activation="softmax")
])

# Compile Model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Callbacks
early_stopping = EarlyStopping(
    monitor='val_accuracy', 
    patience=5, 
    restore_best_weights=True, 
    verbose=1,
    mode='max',
    min_delta=0.001
)

model_checkpoint = ModelCheckpoint(
    "best_model.h5", 
    monitor="val_accuracy", 
    save_best_only=True, 
    mode="max",
    verbose=1
)

# Train Model with Early Stopping
history = model.fit(
    train_generator,
    epochs=100,
    validation_data=val_generator,
    callbacks=[early_stopping, model_checkpoint]
)

# Save the best validation accuracy
best_val_acc = max(history.history['val_accuracy'])
with open(os.path.join(plots_dir, "best_val_accuracy.txt"), "w") as f:
    f.write(f"Best Validation Accuracy: {best_val_acc:.4f}")

# Plot Accuracy vs Epoch
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.legend()
plt.grid()
plt.savefig(os.path.join(plots_dir, "accuracy_vs_epoch.png"))
plt.close()

# Generate Confusion Matrix
y_true = val_generator.classes
y_pred = model.predict(val_generator)
y_pred_classes = np.argmax(y_pred, axis=1)

# Plot Confusion Matrix
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=val_generator.class_indices.keys(), yticklabels=val_generator.class_indices.keys())
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.savefig(os.path.join(plots_dir, "confusion_matrix.png"))
plt.close()

print(f"Plots and best validation accuracy saved in 'plots' folder.")
