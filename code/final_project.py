import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from scipy.io import loadmat
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
# -----------------------------
# 1. Load Model
# -----------------------------
print("Loading ResNet50 model...")
model = ResNet50(weights='imagenet')
print("Model loaded successfully!\n")

# -----------------------------
# 2. Preprocessing
# -----------------------------
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

# -----------------------------
# # 3. Visualization
# # -----------------------------
def visualize_predictions(image_paths):
    plt.figure(figsize=(12, 6))

    for i, img_path in enumerate(image_paths):
        img = preprocess_image(img_path)
        preds = model.predict(img, verbose=0)
        decoded = decode_predictions(preds, top=1)[0][0]

        plt.subplot(1, len(image_paths), i + 1)
        plt.imshow(image.load_img(img_path))
        plt.axis('off')
        plt.title(f"{decoded[1]}\n{decoded[2]:.2f}")

    plt.tight_layout()
    plt.show()

# -----------------------------
# 4. Grad-CAM
# -----------------------------
def make_gradcam_heatmap(img_array, model):
    last_conv_layer = model.get_layer("conv5_block3_out")
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, model.output]
    )
    img_tensor = tf.cast(img_array, tf.float32)
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)

    max_val = tf.reduce_max(heatmap)
    if max_val != 0:
        heatmap = heatmap / max_val

    return heatmap.numpy()

def show_gradcam(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))

    img_array = preprocess_image(img_path)
    heatmap = make_gradcam_heatmap(img_array, model)

    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    plt.imshow(superimposed)
    plt.axis('off')
    plt.title("Grad-CAM")
    plt.show()

# -----------------------------
# 5. Dataset
# -----------------------------
test_images = [
    "llama.jpeg",
    "sports_car.jpeg",
    "volcano.jpeg",
    "pizza.jpeg",
    "safety_pin.jpeg"
]

ground_truth = {
    "llama.jpeg": "llama",
    "sports_car.jpeg": "sports_car",
    "volcano.jpeg": "volcano",
    "pizza.jpeg": "pizza",
    "safety_pin.jpeg": "safety_pin"
}
# -----------------------------
# 6. Visualization
# -----------------------------
print("===== Prediction Visualization =====")
visualize_predictions(test_images)

# -----------------------------
# 7. Evaluation (Top-1 + Top-5)
# -----------------------------
print("\n===== Evaluation =====")

top1_correct = 0
top5_correct = 0

for img_path in test_images:
    img = preprocess_image(img_path)
    preds = model.predict(img, verbose=0)

    decoded_top1 = decode_predictions(preds, top=1)[0][0][1].lower()
    decoded_top5 = [d[1].lower() for d in decode_predictions(preds, top=5)[0]]

    true_label = ground_truth[img_path].lower()

    print(f"\nImage: {img_path}")
    print("Top-5:", decoded_top5)

    if true_label == decoded_top1:
        top1_correct += 1

    if true_label in decoded_top5:
        top5_correct += 1

# -----------------------------
# 8. Results
# -----------------------------
top1_acc = top1_correct / len(test_images)
top5_acc = top5_correct / len(test_images)

print("\n===== Final Results =====")
print("Top-1 Accuracy:", round(top1_acc * 100, 2), "%")
print("Top-5 Accuracy:", round(top5_acc * 100, 2), "%")

# -----------------------------
# 9. Grad-CAM Visualization
# -----------------------------
print("\n===== Grad-CAM =====")
for img_path in test_images[:3]:
    show_gradcam(img_path)

# -----------------------------
# 10. Summary
# -----------------------------
print("\n===== Summary =====")
print("Model: ResNet50 (ImageNet pretrained)")
print("Evaluation: Top-1 + Top-5 + Grad-CAM + Visualization")




# ==========================================
# PART B: STANFORD CARS TRAINING
# ==========================================
print("\n--- Phase 1: Preparing Data ---")
data_dir = 'archive'
train_img_dir = os.path.join(data_dir, 'cars_train', 'cars_train')
train_annos   = os.path.join(data_dir, 'car_devkit', 'devkit', 'cars_train_annos.mat')
class_meta    = os.path.join(data_dir, 'car_devkit', 'devkit', 'cars_meta.mat')

# 1. LOAD & SPLIT DATA
annos = loadmat(train_annos)
annos = loadmat(train_annos)
annotations = annos['annotations'][0]

fnames = [str(a['fname'][0]).split('/')[-1] for a in annotations]
labels = [str(int(a['class'][0][0]) - 1) for a in annotations]

full_df = pd.DataFrame({'filename': fnames, 'class': labels})

# Split 15% for Testing
train_df_temp, test_df = train_test_split(full_df, test_size=0.15, random_state=42, stratify=full_df['class'])

# Split another 20% of the remaining for Validation
train_df, val_df = train_test_split(train_df_temp, test_size=0.2, random_state=42, stratify=train_df_temp['class'])
class_names = [str(c[0]) for c in loadmat(class_meta)['class_names'][0]]

print(f"Total labeled images: {len(full_df)}")
print(f"Training/Val images: {len(train_df)}")
print(f"Testing images: {len(test_df)}")

# 2. GENERATORS
# Augmented the train data
train_datagen_resnet = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=15, width_shift_range=0.1, height_shift_range=0.1,
    horizontal_flip=True, zoom_range=0.1
)

train_gen_resnet = train_datagen_resnet.flow_from_dataframe(
    train_df, train_img_dir, x_col="filename", y_col="class",
    target_size=(224, 224), batch_size=32, class_mode="categorical"
)

# Test is not augmented
val_test_datagen_resnet = ImageDataGenerator(preprocessing_function=preprocess_input)


val_gen_resnet = val_test_datagen_resnet.flow_from_dataframe(
    val_df, train_img_dir, x_col="filename", y_col="class",
    target_size=(224, 224), batch_size=32, class_mode="categorical", shuffle=False
)

test_gen_resnet = val_test_datagen_resnet.flow_from_dataframe(
    test_df, train_img_dir, x_col="filename", y_col="class",
    target_size=(224, 224), batch_size=32, class_mode="categorical", shuffle=False
)

# 3. RESNET MODEL (STAGE 1)
print("\n--- Phase 2: Training ResNet (Stage 1) ---")
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
base_model.trainable = False

resnet_model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(196, activation='softmax')
])

resnet_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
early_stop1 = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history_stage1 = resnet_model.fit(
    train_gen_resnet, validation_data=val_gen_resnet, epochs=5, callbacks=[early_stop1]
)

# 4. RESNET MODEL (STAGE 2)
print("\n--- Phase 3: Fine-Tuning ResNet ---")
# To unfreeze the last block dynamically
base_model.trainable = True
set_trainable = False
for layer in base_model.layers:
    if layer.name == 'conv5_block1_1_conv': # Start of the final block
        set_trainable = True
    layer.trainable = set_trainable


# To see exactly what will be trained
for i, layer in enumerate(resnet_model.layers[0].layers):
    if layer.trainable:
        print(f"Layer {i}: {layer.name} is trainable")

resnet_model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
early_stop2 = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)

train_gen_resnet.reset()
val_gen_resnet.reset()

history_stage2 = resnet_model.fit(
    train_gen_resnet, validation_data=val_gen_resnet, epochs=80, callbacks=[early_stop2, reduce_lr]
)

# 5. BASELINE CNN
print("\n--- Phase 4: Training Baseline CNN ---")
cnn_datagen = ImageDataGenerator(rescale=1./255)

train_gen_cnn = cnn_datagen.flow_from_dataframe(
    train_df, train_img_dir,
    x_col="filename", y_col="class",
    target_size=(224,224), batch_size=32, class_mode="categorical"
)

val_gen_cnn = cnn_datagen.flow_from_dataframe(
    val_df, train_img_dir,
    x_col="filename", y_col="class",
    target_size=(224,224), batch_size=32, class_mode="categorical", shuffle=False
)

test_gen_cnn = ImageDataGenerator(rescale=1./255).flow_from_dataframe(
    test_df, train_img_dir,
    x_col="filename", y_col="class",
    target_size=(224,224), batch_size=32, class_mode="categorical", shuffle=False
)

baseline_cnn = models.Sequential([
    layers.Input(shape=(224,224,3)),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dense(196, activation='softmax')
])

baseline_cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
early_stop_cnn = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)

history_cnn = baseline_cnn.fit(
    train_gen_cnn, validation_data=val_gen_cnn, epochs=20, callbacks=[early_stop_cnn]
)

# 6. TEST EVALUATION
print("\n--- Phase 5: Final Evaluation ---")
test_gen_resnet.reset()
resnet_acc = resnet_model.evaluate(test_gen_resnet)[1]
print(f"ResNet Test Accuracy: {resnet_acc*100:.2f}%")

test_gen_cnn.reset()
cnn_acc = baseline_cnn.evaluate(test_gen_cnn)[1]
print(f"CNN Test Accuracy: {cnn_acc*100:.2f}%")

# 7. PLOTS
print("\n--- Phase 6: Plotting ---")
resnet_val_acc = history_stage1.history['val_accuracy'] + history_stage2.history['val_accuracy']
cnn_val_acc = history_cnn.history['val_accuracy']

resnet_val_loss = history_stage1.history['loss'] + history_stage2.history['loss']
cnn_val_loss = history_cnn.history['loss']

max_len = max(len(resnet_val_acc), len(cnn_val_acc))

# Pad Accuracy
resnet_val_acc += [np.nan] * (max_len - len(resnet_val_acc))
cnn_val_acc += [np.nan] * (max_len - len(cnn_val_acc))

# Pad Loss
resnet_val_loss += [np.nan] * (max_len - len(resnet_val_loss))
cnn_val_loss += [np.nan] * (max_len - len(cnn_val_loss))

# Plot Accuracy
plt.figure(figsize=(10,6))
plt.plot(resnet_val_acc, label='ResNet50')
plt.plot(cnn_val_acc, label='CNN')
plt.axvline(x=len(history_stage1.history['val_accuracy'])-1, linestyle='--', color='gray')
plt.legend()
plt.title("Validation Accuracy")
plt.show()

# Plot Loss
plt.figure(figsize=(10,6))
plt.plot(resnet_val_loss, label='ResNet Loss')
plt.plot(cnn_val_loss, label='CNN Loss')
plt.axvline(x=len(history_stage1.history['loss'])-1, linestyle='--', color='gray')
plt.legend()
plt.title("Training Loss")
plt.show()

# 8. CONFUSION MATRIX
print("\n--- Phase 7: Confusion Matrix ---")
test_gen_resnet.reset()
y_pred = np.argmax(resnet_model.predict(test_gen_resnet), axis=1)
y_true = test_gen_resnet.classes

cm = confusion_matrix(y_true, y_pred)
confusion_scores = cm.sum(axis=1) - np.diag(cm)
top_classes = np.argsort(confusion_scores)[-10:]
cm_subset = cm[top_classes][:, top_classes]

inv_map = {v: int(k) for k, v in test_gen_resnet.class_indices.items()}
top_labels = [class_names[inv_map[i]][:20] for i in top_classes]

plt.figure(figsize=(12,10))
sns.heatmap(cm_subset, cmap='Blues', xticklabels=top_labels, yticklabels=top_labels, annot=True, fmt='d')
plt.title("Top 10 Most Confused Classes")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ==========================================
# 9. SAVE MODELS
# ==========================================
print("\n--- Phase 8: Saving Models ---")

# Save the fine-tuned ResNet model
resnet_model.save('final_resnet_cars2.keras')
print("Saved ResNet50 model as 'final_resnet_cars2.keras'")

# Save the Baseline CNN model
baseline_cnn.save('final_baseline_cnn2.keras')
print("Saved Baseline CNN model as 'final_baseline_cnn2.keras'")