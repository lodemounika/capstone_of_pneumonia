import os
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

print("\n=== Final Chest X-ray CNN Model ===\n")

# -----------------------------
# DATASET PATH
# -----------------------------
DATASET_DIR = r"C:\Users\DELL\OneDrive\Desktop\cu project\COVID19_Pneumonia_Normal_Chest_Xray_PA_Dataset"

print("Folders:", os.listdir(DATASET_DIR))

# -----------------------------
# PARAMETERS
# -----------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15

# -----------------------------
# DATA AUGMENTATION
# -----------------------------
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=30,
    zoom_range=0.3,
    width_shift_range=0.3,
    height_shift_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3]
)

train_data = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_data = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

class_names = list(train_data.class_indices.keys())
print("Class Labels:", class_names)

# -----------------------------
# CLASS WEIGHTS
# -----------------------------
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_data.classes),
    y=train_data.classes
)
class_weights = dict(enumerate(class_weights))

# -----------------------------
# MODEL
# -----------------------------
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

for layer in base_model.layers[:-50]:
    layer.trainable = False

for layer in base_model.layers[-50:]:
    layer.trainable = True

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(train_data.num_classes, activation="softmax")
])

model.compile(
    optimizer=Adam(learning_rate=5e-6),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# -----------------------------
# TRAIN
# -----------------------------
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=[early_stop]
)

# -----------------------------
# EVALUATE
# -----------------------------
loss, acc = model.evaluate(val_data)
print(f"\n✅ Final Accuracy: {acc * 100:.2f}%")

# -----------------------------
# 📊 PLOTS (IMPORTANT FOR PPT)
# -----------------------------
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'])
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'])
plt.show()

# -----------------------------
# 📌 CONFUSION MATRIX
# -----------------------------
predictions = model.predict(val_data)
y_pred = np.argmax(predictions, axis=1)
y_true = val_data.classes

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=class_names,
            yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# -----------------------------
# 📄 CLASSIFICATION REPORT
# -----------------------------
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_names))

# -----------------------------
# 💾 SAVE MODEL (BOTH FORMATS)
# -----------------------------
model.save("chest_model.h5")        # for Streamlit
model.save("chest_model.keras")     # for latest Keras

print("✅ Models saved (.h5 + .keras)")