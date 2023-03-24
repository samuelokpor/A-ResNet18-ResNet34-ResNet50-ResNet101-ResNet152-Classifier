import cv2
from keras.metrics import Precision, Recall, CategoricalAccuracy
# from dataset import test
import numpy as np
from datatf import test
from allresnets import ResLayer
from resnet18 import resnet18instance



resnet18instance.load_weights('model.h5')



pre = Precision()
re = Recall()
acc = CategoricalAccuracy()

for batch in test.as_numpy_iterator():
    X, y = batch
    yhat = resnet18instance.predict(X)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)

print(f'Precision: {pre.result().numpy()}, Recall:{re.result().numpy()}, Accuracy: {acc.result().numpy()}')

img = cv2.imread('happytest.jpg')
resize = cv2.resize(img, (224, 224))
resize = cv2.cvtColor(resize, cv2.COLOR_BGR2RGB)
resize = resize / 255.0

yhat = resnet18instance.predict(np.expand_dims(resize, 0))
pred_label = np.argmax(yhat)

if pred_label == 0:
    mood = "happy"
elif pred_label == 1:
    mood = "Sad"

cv2.putText(img, mood, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
cv2.imshow("Test Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()