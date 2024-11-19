import cv2
import mysql.connector
from datetime import datetime

# Установите соединение с базой данных
db_connection = mysql.connector.connect(
    host="localhost",      # или IP-адрес вашего MySQL-сервера
    user="root",  # ваше имя пользователя
    password="0000",          # ваш пароль
    database="computerVision"     # имя вашей базы данных
)

cursor = db_connection.cursor()

# Инициализация каскадного классификатора для обнаружения лиц
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Захват видео с камеры
capture = cv2.VideoCapture(0)

while True:
    ret, img = capture.read()
    
    if not ret:
        print("Не удалось захватить изображение")
        break

    # Обнаружение лиц
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.5, minNeighbors=2, minSize=(20, 20))

    # Рисуем прямоугольники вокруг обнаруженных лиц
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0))

    # Записываем количество лиц и текущее время в базу данных
    face_count = len(faces)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # SQL запрос для вставки данных
    sql = "INSERT INTO face_detection (timestamp, face_count) VALUES (%s, %s)"
    cursor.execute(sql, (timestamp, face_count))
    db_connection.commit()  # Сохраняем изменения в базе данных

    print(f"количество лиц: {face_count} на {timestamp}")

    # Отображаем изображение с обнаружением лиц
    cv2.imshow("My webcam", img)
    k = cv2.waitKey(30) & 0xFF
    if k == 27:  # нажмите ESC для выхода
        break

# Освобождение ресурсов
capture.release()
cv2.destroyAllWindows()

# Закрываем соединение с базой данных
cursor.close()
db_connection.close()
