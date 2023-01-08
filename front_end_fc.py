import cv2
import numpy as np
from keras.preprocessing import image
from PIL import Image
from tensorflow.keras.models import load_model

#importing model and weights
check = load_model('fr_model.h5')
check.load_weights('weights.h5')

#definig the haarcascade face classifier
fr_haar_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades+'haarcascade_fr_default.xml')

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while cap.isOpened:
    t, s = cap.read()
    f= fr_haar_cascade.detectMultiScale(s, 1.3, 5)
    for (x, y, w, h) in f:
        cv2.rectangle(s, (x, y), (x+w, y+h), (255, 0, 0), 2)
    if type(s) is np.ndarray:
        s = cv2.resize(s, (250, 250))
        im = Image.fromarray(s, 'RGB')
        img_array = np.array(im)
        img_array = np.expand_dims(img_array, axis=0)
        prdt = check.predict(img_array)
        print(prdt)

        if(prdt[0][0] > 0.3):
            name = 'Phalguni' # enter your name here
        cv2.putText(s, name, (50, 50),
                    cv2.FONT_ARIAL, 1, (0, 255, 0), 1)

    else:
        cv2.putText(s, "Face not found", (50, 50),
                    cv2.FONT_ARIAL, 1, (0, 255, 0), 1)

    cv2.imshow('image', s)
    if cv2.waitKey(1) & 0xFF == ord('q'): #the windwos will close on pressing q
        break

cap.release()
cv2.destroyAllWindows()