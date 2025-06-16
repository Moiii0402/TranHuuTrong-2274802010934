import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array
import pickle
from scipy.spatial.distance import cosine
import time

# Load embedding và nhãn
with open("face_embeddings.pkl", "rb") as f:
    embeddings_db, labels_db = pickle.load(f)

# Load sinh viên
df = pd.read_csv("students.csv", encoding="utf-8-sig")
df.columns = df.columns.str.strip()

def get_info_by_id(sid):
    row = df[df['student_id'].astype(str) == str(sid)]
    return row.iloc[0].to_dict() if not row.empty else None

# Model đặc trưng
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model = Model(inputs=base_model.input, outputs=base_model.output)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)

print(" Đang nhận diện...")

recognized = False
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        if time.time() - start_time >= 10:
            print(" Không có gương mặt sau 10s. Nhập MSSV.")
            break
    else:
        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (224, 224))
            face_img = img_to_array(face_img)
            face_img = preprocess_input(face_img)
            face_img = np.expand_dims(face_img, axis=0)

            embedding = model.predict(face_img, verbose=0).flatten()

            # So sánh khoảng cách
            distances = [cosine(embedding, db_vec) for db_vec in embeddings_db]
            min_dist = min(distances)
            best_match_index = np.argmin(distances)

            if min_dist < 0.3:  # ngưỡng nhận diện
                mssv = labels_db[best_match_index]
                info = get_info_by_id(mssv)
                if info:
                    print(f"\n Nhận diện: {info['full_name']} ({mssv})")
                    print(f" Quê quán: {info['hometown']}")
                    print(f" Ngày sinh: {info['birth_date']} | Giới tính: {info['gender']}")
                    recognized = True
                    break
            else:
                print(" Không có gương mặt trong dữ liệu.")

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == 27 or recognized:
        break

cap.release()
cv2.destroyAllWindows()

if not recognized:
    sid = input(" Nhập MSSV để tìm thông tin: ")
    info = get_info_by_id(sid)
    if info:
        print(f"\n Tìm thấy sinh viên: {info['full_name']} ({sid})")
        print(f" Quê quán: {info['hometown']}")
        print(f" Ngày sinh: {info['birth_date']} | Giới tính: {info['gender']}")
    else:
        print(" Không tìm thấy sinh viên.")
