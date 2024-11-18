import cv2
import face_recognition

# Инициализация переменных
known_face_encodings = []
current_id = 0  # Счетчик для уникальных идентификаторов
face_id_mapping = {}  # Словарь, сопоставляющий координаты лиц с уникальными идентификаторами

# Видео поток
video_capture = cv2.VideoCapture(0)  # '0' для веб-камеры, или укажите путь к видеофайлу

while True:
    # Захват кадра
    ret, frame = video_capture.read()
    
    # Преобразование изображения из BGR в RGB
    rgb_frame = frame[:, :, ::-1]

    # Обнаружение лиц в кадре
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    detected_ids = []  # Список для хранения обнаруженных идентификаторов

    for face_encoding, face_location in zip(face_encodings, face_locations):
        # Уникальный идентификатор для этого лица
        # Проверка, встречали ли мы это лицо ранее
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        
        if True not in matches:
            # Если совпадений нет, назначить новое имя
            detected_id = current_id
            known_face_encodings.append(face_encoding)  # Добавляем новое лицо в известные лица
            current_id += 1  # Увеличиваем идентификатор для следующего лица
        else:
            # Если лицо уже известно, находим его идентификатор
            first_match_index = matches.index(True)
            detected_id = first_match_index  # Используем индекс известных лиц как идентификатор

        # Сохраняем найденный идентификатор
        detected_ids.append(detected_id)

        # Прямоугольник вокруг лица
        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, str(detected_id), (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Вывод количества обнаруженных лиц и их идентификаторы в консоль
    print(f"Обнаружено лиц: {len(detected_ids)}; Идентификаторы: {detected_ids}")

    # Отображение результата
    cv2.imshow('Video', frame)

    # Выйти при нажатии 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
video_capture.release()
cv2.destroyAllWindows()
