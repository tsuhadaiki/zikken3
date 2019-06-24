#ラベリングによる学習/検証データの準備
import json
from PIL import Image
import os, glob
import numpy as np
import random, math
from keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array, array_to_img
f = open('set.json', 'r', encoding="utf-8")
json_data = json.load(f)
#画像が保存されているルートディレクトリのパス(水増し前)
root_dir = json_data["images"]
save = json_data["save_npy"]
# 商品名
categories = ["building","food","pet","view"]


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
	img = Image.open(fname)
	img = img.convert("RGB")
	img = img.resize((150, 150))
	data = np.asarray(img)
	if(flag):
		for i in range(10):
			data2 = draw_images(datagen,data)
			X.append(data2)
			Y.append(cat)
	else:
		X.append(data)
		Y.append(cat)
	return X,Y
#全データ格納用配列
allfiles=[]

#画像の水増しを行う関数
def draw_images(generator,x):
	x2=(x[np.newaxis,:,:,:])
	g = generator.flow(x2,batch_size=1)
	# 1つの入力画像から何枚拡張するかを指定（今回は50枚）
	bach=np.reshape(g.next(),(150,150,3))
	return bach


#カテゴリ配列の各値と、それに対応するidxを認識しtestとtrainそれぞれで読みこむ
for idx, cat in enumerate(categories):
    image_dir = root_dir + cat
    files = glob.glob(image_dir + "/*.jpg")
    for f in files:
        allfiles.append((idx, f))

#シャッフル後、学習データと検証データに分ける
random.shuffle(allfiles)
th = math.floor(len(allfiles) * 0.8)#リーブワンアウトとかどうか
train = allfiles[0:th]
test  = allfiles[th:]

X_train, y_train = make_sample(train,True) #Trueの時に水増し
X_test, y_test = make_sample(test,False) #Falseでは水増しなし
xy = (X_train, X_test, y_train, y_test)
#データを保存する（データの名前を「insta_data2.npy」としている）
np.save(save+"/insta_data2.npy", xy)