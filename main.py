import cv2

capture = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
while True:
    ret, img = capture.read()
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.5, minNeighbors=2, minSize=(20, 20))
    for(x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0))

    cv2.imshow("My webcam", img)
    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break
capture.release()
cv2.destroyAllWindows()
