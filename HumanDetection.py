import cv2
import numpy as np
import uuid
import time
from ultralytics import YOLO
import sys
import logging
#отключаем стандартные логи 
logging.getLogger("ultralytics").setLevel(logging.CRITICAL)
sys.stdout = open('/dev/null', 'w') 
sys.stderr = open('/dev/null', 'w')  

model = YOLO("yolov8n.pt", verbose=False)
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__
# Словарь для хранения данных о людях
people_data = {}

# Время исчезновения ID
DISAPPEAR_TIME = 5  

# Время логирования
LOG_INTERVAL = 10  
last_log_time = time.time()

# Память о людях за последние 10 секунд
people_count_history = []


VALIDATOR_X1, VALIDATOR_Y1 = 300, 200
VALIDATOR_X2, VALIDATOR_Y2 = 400, 350

# Список оплативших людей
paid_people = set()


cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    detected_people = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]
            class_id = int(box.cls[0])

            if class_id == 0 and confidence > 0.3:  #!!!!!
                detected_people.append((x1, y1, x2, y2))

    new_people_data = {}

    for (x1, y1, x2, y2) in detected_people:
        found = False

        for ID, data in people_data.items():
            old_x1, old_y1, old_x2, old_y2 = data["bbox"]

          
            if abs(x1 - old_x1) < 50 and abs(y1 - old_y1) < 50:
                new_people_data[ID] = {"bbox": (x1, y1, x2, y2), "last_seen": time.time()}
                found = True

                # взаимодействие с валидатором
                if VALIDATOR_X1 < x1 < VALIDATOR_X2 and VALIDATOR_Y1 < y1 < VALIDATOR_Y2:
                    paid_people.add(ID)

                break

        # не нашли совпадений – создаем новый ID
        if not found:
            new_ID = str(uuid.uuid4())[:8]  # Короткий уникальный ID
            new_people_data[new_ID] = {"bbox": (x1, y1, x2, y2), "last_seen": time.time()}

    # Удаляем людей, которые пропали
    current_time = time.time()
    for ID in list(people_data.keys()):
        if ID not in new_people_data and current_time - people_data[ID]["last_seen"] > DISAPPEAR_TIME:
            print(f"[INFO] Удаление ID {ID}, он пропал из кадра")
        else:
            new_people_data[ID] = people_data[ID]

    people_data = new_people_data  # Обновляем данные

    # Запоминаем количество людей
    people_count_history.append(len(people_data))
    if len(people_count_history) > LOG_INTERVAL:
        people_count_history.pop(0)

    # Логи
    if current_time - last_log_time >= LOG_INTERVAL:
        last_log_time = current_time
        avg_people = round(sum(people_count_history) / len(people_count_history), 2)
        print(f"[INFO] Среднее количество людей за 10 секунд: {avg_people}")

        for ID, data in people_data.items():
            status = "✅ Оплатил" if ID in paid_people else "❌ Не оплатил"
            print(f"  - ID: {ID} | В кадре {round(time.time() - data['last_seen'], 2)} сек | {status}")

    # Рисуем рамки и ID на изображении
    for ID, data in people_data.items():
        x1, y1, x2, y2 = data["bbox"]
        color = (0, 255, 0) if ID in paid_people else (0, 0, 255)  # Зеленый = оплатил, Красный = не оплатил
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID: {ID}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Рисуем зону валидатора
    cv2.rectangle(frame, (VALIDATOR_X1, VALIDATOR_Y1), (VALIDATOR_X2, VALIDATOR_Y2), (255, 0, 0), 2)


    cv2.imshow("YOLOv8 Person Tracking", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
