from itertools import count
import cv2 as cv

face_classifier = cv.CascadeClassifier(
    'Face Recognition using Keras\haarcascade_frontalface_default.xml')


def face_extractor(img):
    faces = face_classifier.detectMultiScale(img, 1.3, 5)
    if faces is ():
        return None
    for (x, y, w, h) in faces:
        x = x-10
        y = y-10
        cropped_face = img[y:y+h+50, x:x+w+50]
    return cropped_face


cap = cv.VideoCapture(0)
count = 0

while True:
    ret, frame = cap.read()
    if face_extractor(frame) is not None:
        count += 1
        face = cv.resize(face_extractor(frame), (400, 400))
        file_path = './Face Recognition using Keras\DataSets\Train\Kushagra' + str(count) + '.jpg'
        cv.imwrite(file_path, face)
        cv.putText(face, str(count), (50, 50),
                   cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv.imshow('Face Cropper', face)
    else:
        print("Face not Found!")
        pass
    if cv.waitKey(1) == 3 or count == 200:
        break
cap.release()
cv.destroyAllWindows()
print("Sample Collection Complete!")
