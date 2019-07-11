#画像の水増し
from keras import models
from keras.models import model_from_json
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
import numpy as np
import json
import os, glob
from keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array, array_to_img
import random, math
from PIL import Image
import my_pickle as mp


f = open('set.json', 'r', encoding="utf-8")
json_data = json.load(f)
#画像が保存されているルートディレクトリのパス(水増し前)
root_dir = json_data["images"]
save = json_data["sav"]
infx = json_data["infx"]
infy = json_data["infy"]
# 商品名
categories = ["sweets映えてる","sweets映えてない"]


#画像データごとにadd_sample()を呼び出し、X,Yの配列を返す関数
def make_sample(files,flag):
    #local X, Y
    X = []
    Y = []
    for cat, fname in files:
        X,Y = add_sample(cat, fname,X,Y,flag)
    return np.array(X), np.array(Y)

datagen = ImageDataGenerator(rotation_range=30,
                            width_shift_range=20,
                            height_shift_range=0.,
                            zoom_range=0.1,
                            horizontal_flip=True,
                            vertical_flip=True)
#渡された画像データを読み込んでXに格納し、また、
#画像データに対応するcategoriesのidxをY格納する関数
def add_sample(cat, fname, X, Y, flag):
	#print(fname)
	img = Image.open(fname)
	img = img.convert("RGB")
	img = img.resize((150, 150))
	data = np.asarray(img)
	if(flag):
		for i in range(30):
			data2 = draw_images(datagen,data)
			X.append(data2)
			Y.append(cat)
	else:
		X.append(data)
		Y.append(cat)
	return X,Y
#全データ格納用配列
allfiles=[]
big=[]

#カテゴリ配列の各値と、それに対応するidxを認識しtestとtrainそれぞれで読みこむ
for idx, cat in enumerate(categories):
    image_dir = root_dir + cat
    files = glob.glob(image_dir + "/*.jpg")
    for f in files:
        allfiles.append((idx, f))

#シャッフル後、学習データと検証データに分ける
random.shuffle(allfiles)
th = math.floor(len(allfiles) * 0.8)
train = allfiles[0:th]
test  = allfiles[th:]

def draw_images(generator,x):
	x2=(x[np.newaxis,:,:,:])
	g = generator.flow(x2,batch_size=1)
	# 1つの入力画像から何枚拡張するかを指定（今回は50枚）
	bach=np.reshape(g.next(),(150,150,3))
	return bach

# ImageDataGeneratorを定義
datagen = ImageDataGenerator(rotation_range=30,
                            width_shift_range=20,
                            height_shift_range=0.,
                            zoom_range=0.1,
                            horizontal_flip=True,
                            vertical_flip=True)

random.shuffle(allfiles)
th = math.floor(len(allfiles) * 0.8)
train = allfiles[0:th]
test  = allfiles[th:]
#すでに水増ししたデータがあるか無いか
if not(os.path.exists(infx) and os.path.exists(infy)):
	X_train, y_train = make_sample(train,True)
	mp.pickle_dump(X_train,infx)
	mp.pickle_dump(y_train,infy)
#水増ししたデータを読み込み
X_train = mp.pickle_load(infx)
y_train = mp.pickle_load(infy)
#テスト用の画像を読み込み
X_test, y_test = make_sample(test,False) #Falseでは水増しなし

xy = (X_train, X_test, y_train, y_test)
#データを保存する（データの名前を「insta_data3.sav」としている）
mp.pickle_dump(xy, save+"/sweets.sav")