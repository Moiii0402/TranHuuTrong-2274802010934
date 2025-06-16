import cv2
import os
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array
import pickle

face_embeddings = []
labels = []

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model = Model(inputs=base_model.input, outputs=base_model.output)

for file in os.listdir("dataset"):
    if file.endswith(".jpg"):
        img_path = os.path.join("dataset", file)
        image = cv2.imread(img_path)
        image = cv2.resize(image, (224, 224))
        image = img_to_array(image)
        image = preprocess_input(image)
        image = np.expand_dims(image, axis=0)

        embedding = model.predict(image, verbose=0)
        embedding = embedding.flatten()

        student_id = file.split("_")[0]
        face_embeddings.append(embedding)
        labels.append(student_id)

# Lưu vector và nhãn
with open("face_embeddings.pkl", "wb") as f:
    pickle.dump((face_embeddings, labels), f)

print(" Trích xuất và lưu đặc trưng xong.")
