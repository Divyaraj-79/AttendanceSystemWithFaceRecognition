import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# Step 1: Load images and names of the attendees
path = 'Image'
images = []
classNames = []
myList = os.listdir(path)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

# Step 2: Encode faces
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown = findEncodings(images)
print('Encoding complete')

# Step 3: Initialize camera and face detection
cap = cv2.VideoCapture(0)
scale_factor = 0.30

while True:
    ret, img = cap.read()
    img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor)
    imgSmall = cv2.resize(img, None, fx=scale_factor, fy=scale_factor)

    # Step 4: Detect faces and their encodings
    facesCurFrame = face_recognition.face_locations(imgSmall)
    encodesCurFrame = face_recognition.face_encodings(imgSmall, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        encodeFace = np.array(encodeFace)
        # Step 5: Compare with known faces
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)

            # Step 6: Record attendance
            with open('Attendance.csv', 'r+') as f:
                myDataList = f.readlines()
                nameList = []
                for line in myDataList:
                    entry = line.split(',')
                    nameList.append(entry[0])
                if name not in nameList:
                    now = datetime.now()
                    dtString = now.strftime('%H:%M:%S')
                    f.writelines(f'\n{name},{dtString}')

        # Step 7: Draw rectangles and text on the video feed
        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = int(y1/scale_factor), int(x2/scale_factor), int(y2/scale_factor), int(x1/scale_factor)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, name, (x1+6, y2-6),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    # Step 8: Show video feed and exit on 'q'
    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
