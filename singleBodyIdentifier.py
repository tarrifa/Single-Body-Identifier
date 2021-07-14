import cv2 as cv

body_classifier = cv.CascadeClassifier("haarcascade_fullbody.xml")
cap = cv.VideoCapture("videos/video.mp4")

while cap.isOpened():
    ret , frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    bodies = body_classifier.detectMultiScale(gray, 1.1, 4)

    for (x,y,w,h) in bodies:
        cv.rectangle(frame, (x,y), (x+w, y+h), (0,255,255), 2)
        cv.imshow("body detection",  frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()  
