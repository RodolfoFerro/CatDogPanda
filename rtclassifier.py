from keras.models import load_model
import argparse
import pickle
import cv2


model_path = "model/smallvggnet.model"
label_bin = "model/smallvggnet_lb.pickle"
lb = pickle.loads(open(label_bin, "rb").read())
model = load_model(model_path)

# Initialize videocapture:
cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame, make a copy and resize:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 360))
    image = frame.copy()
    image = image[30:280, 50:300]
    cv2.imshow('What computer sees', image)
    image = cv2.resize(image, (64, 64))
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

    # Make predictions:
    preds = model.predict(image)
    i = preds.argmax(axis=1)[0]
    label = lb.classes_[i]

    # draw the class label + probability on the output image
    cv2.rectangle(frame, (30, 50), (230, 250), (250, 150, 10), 2)
    text = "{}: {:.2f}%".format(label, preds[0][i] * 100)
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 0), 2)

    # Display the resulting frame:
    cv2.imshow('Prediction', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture:
cap.release()
cv2.destroyAllWindows()
