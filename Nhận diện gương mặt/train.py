import cv2
import os

student_id = input("Nhập MSSV: ").strip()
save_dir = "dataset"
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

count = 0
print(" Bắt đầu thu thập gương mặt... Nhấn ESC để dừng.")

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (100, 100))
        filename = f"{save_dir}/{student_id}_{count}.jpg"
        cv2.imwrite(filename, face_img)
        count += 1
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Face Collector", frame)
    if cv2.waitKey(1) == 27 or count >= 50:
        break

cap.release()
cv2.destroyAllWindows()
print(f" Đã lưu {count} ảnh vào thư mục '{save_dir}'")
