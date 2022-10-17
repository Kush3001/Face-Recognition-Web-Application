from PIL import Image
import cv2 as cv
from keras.models import load_model
import numpy as np
import streamlit as st

# st.title('Face Recognition')
st.markdown("<h1 style='text-align: center'>Face Recognition System</h1>",
            unsafe_allow_html=True)
# run = st.button(label="Turn on Video!")
# stop = st.button(label="Turn off Video!")
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    pass
with col2:
    run = st.button(label="Turn on Video!")
with col4:
    stop = st.button(label="Turn off Video!")
with col5:
    pass
with col3:
    pass

frame_window = st.image([], channels="BGR")

model = load_model('Face Recognition using Keras\FaceModel1.h5')
face_cascade = cv.CascadeClassifier(
    'Face Recognition using Keras\haarcascade_frontalface_default.xml')


def face_extractor(img):
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    if faces is ():
        return None
    for (x, y, w, h) in faces:
        cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 2)
        cropped_face = img[y:y+h, x:x+w]
    return cropped_face


video_capture = cv.VideoCapture(0)
# while True:
while run:
    _, frame = video_capture.read()
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    face = face_extractor(frame)
    if type(face) is np.ndarray:
        face = cv.resize(face, (224, 224))
        im = Image.fromarray(face, 'RGB')
        # Resizing into 128x128 because we trained the model with this image size.
        img_array = np.array(im)
        # Changing Dimentions:
        img_array = np.expand_dims(img_array, axis=0)
        pred = model.predict(img_array)
        print(pred)
        name = "None matching"
        if(pred[0][0] > 0.5):
            name = 'Aniket'
        elif(pred[0][1] > 0.5):
            name = 'Sushovan'
        elif(pred[0][2] > 0.5):
            name = 'Kushagra'
        cv.putText(frame, name, (50, 50),
                   cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    else:
        cv.putText(frame, "No face found", (50, 50),
                   cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    # cv.imshow('Video', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    frame_window.image(frame)
if (stop):
    video_capture.release()
cv.destroyAllWindows()
