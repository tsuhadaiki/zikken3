from keras import models
from keras.models import model_from_json
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
import numpy as np

#保存したモデルの読み込み
model = model_from_json(open('/Users/e175764/Desktop/DataMining/colorful/tea_predict.json').read())
#保存した重みの読み込み
model.load_weights('/Users/e175764/Desktop/DataMining/colorful/tea_predict.hdf5')

categories = ["映える","映えない"]

#画像を読み込む
img_path = str(input())
img = image.load_img(img_path,target_size=(150, 150, 3))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

x = preprocess_input(x)
#予測
preds = model.predict(x)
#予測結果によって処理を分ける
if preds[0,0] == 1:
    print ("映える")
else:
    message = "映えぬ"
    print(message)

print('preds.shape: {}'.format(preds.shape))  # preds.shape: (1, 1000)
"""
result = decode_predictions(preds, top=2)[0]
print(result)
"""