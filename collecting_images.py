import cv2
import os
import uuid

# replace person with your own name below and then other person name name to create a dataset of binary classification

train_path = os.path.join('Dataset', 'train', 'phalguni')
test_path = os.path.join('Dataset', 'test', 'phalguni')

face_haar_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while cap.isOpened():
    ret, s = cap.read()

    # on pressing c the images will be captured and stored on the train path
    if cv2.waitKey(1) & 0xFF == ord('c'):
        i_name = os.path.join(train_path, '{}.jpg'.format(uuid.uuid1()))
        cv2.imwrite(i_name, s)

    # the images will be captured and stored on pressing l
    if cv2.waitKey(1) & 0xFF == ord('l'):
        imgname = os.path.join(test_path, '{}.jpg'.format(uuid.uuid1()))
        cv2.imwrite(i_name, s)

    cv2.imshow('images', s)

    if cv2.waitKey(1) & 0xFF == ord('e'):  # window closes on pressing e
        break

cap.release()
cv2.destroyAllWindows()
