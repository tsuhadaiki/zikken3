import json
from keras import models
from keras.models import model_from_json
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
import numpy as np

f = open("set.json", 'r')
json_data = json.load(f) #JSON形式で読み込む
modelpass = json_data["model"]

#保存したモデルの読み込み
model = model_from_json(open(modelpass +'/tea_predict.json').read())
#保存した重みの読み込み
model.load_weights(modelpass +'/tea_predict.hdf5')

categories = ["building","food","pet","view"]

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
	print("入力画像はbuilding")
elif preds[0,1] == 1:
	print("入力画像はfood")
elif preds[0,2] == 1:
	print("入力画像はpet")
elif preds[0,3] == 1:
	print("入力画像はview")
else:
	print("だいたいペット")

print(preds)
print('preds.shape: {}'.format(preds.shape))  # preds.shape: (1, 1000)
"""
result = decode_predictions(preds, top=2)[0]
print(result)
"""