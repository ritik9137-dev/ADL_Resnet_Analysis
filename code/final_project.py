import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

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
# 3. Visualization
# -----------------------------
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