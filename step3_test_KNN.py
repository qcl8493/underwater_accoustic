import numpy as np
from Util.dtw import dtw
from scipy.linalg import norm
import librosa.display
from sklearn.neighbors import KNeighborsClassifier
import os

# 需要测试的音频数据
y, sr = librosa.load('data/test_after_process/德_鱼雷/德_鱼雷_电子鱼雷0.wav')

"""展示测试样本在哪一类"""
temp = np.load('save_model/KNN/data_1.npz')
temp2 = np.load('save_model/KNN/data2_1.npz')
distances = temp.f.arr_0
label = temp2.f.arr_0
classifier = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
# fit可以简单的认为是表格存储
classifier.fit(distances, label)
mfcc = librosa.feature.mfcc(y=y, sr=sr)
dirname = "./data/train_after_process"
distanceTest = []
# 存放每一类的距离和以及样本点数
distance_dict = {}
dirs = [f for f in os.listdir(dirname)]
for dir in dirs:
    distance_dict[dir] = [0, 0]

for dir in dirs:
    files = os.listdir(dirname + '/' + dir)
    for i in range(len(files)):
        y1, sr1 = librosa.load(dirname + "/" + dir + '/' + files[i])
        mfcc1 = librosa.feature.mfcc(y=y1, sr=sr1)
        dist, _, _, _ = dtw(mfcc.T, mfcc1.T, dist=lambda x, y: norm(x - y, ord=1))
        distanceTest.append(dist)
        distance_dict[dir][0] += dist
        distance_dict[dir][1] += 1

for k, v in distance_dict.items():
    print("测试样本与{}类的距离是{}".format(k, v[0] / v[1]))
pre = classifier.predict([distanceTest])[0]
print(pre)
print("Predict audio is: '{}'".format(dirs[int(pre)]))
