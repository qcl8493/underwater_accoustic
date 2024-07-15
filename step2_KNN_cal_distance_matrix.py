from Util.dtw import dtw
import librosa.display
from scipy.linalg import norm
import numpy as np
import os

# 计算所使用的数据路径
dirname = r".\data\train_after_process"

"""计算dtw矩阵"""
dirs = [f for f in os.listdir(dirname)]
res = 0
for dir1 in dirs:
    files1 = os.listdir(dirname + '/' + dir1)
    res += len(files1)
# 初始化距离矩阵和标签
distances = np.ones((res, res))
label = np.ones(res)

res1 = 0
l = 0
for dir1 in dirs:
    files1 = os.listdir(dirname + '/' + dir1)
    for i in range(len(files1)):
        print(f"进度：{i + 1 + res1}/{len(files1) + res1}")
        y1, sr1 = librosa.load(dirname + '/' + dir1 + '/' + files1[i])
        mfcc1 = librosa.feature.mfcc(y=y1, sr=sr1)
        res2 = 0
        for dir2 in dirs:
            files2 = os.listdir(dirname + '/' + dir2)
            for j in range(len(files2)):
                y2, sr2 = librosa.load(dirname + "/" + dir2 + '/' + files2[j])
                mfcc2 = librosa.feature.mfcc(y=y2, sr=sr2)
                dist, _, _, _ = dtw(mfcc1.T, mfcc2.T, dist=lambda x, y: norm(x - y, ord=1))
                distances[i + res1, j + res2] = dist
            res2 += len(files2)
        label[i + res1] = l
    res1 += len(files1)
    l += 1

print(distances)

# 保存距离矩阵，注释掉防止误操作
np.savez('.\save_model\KNN\data_1', distances)
np.savez('.\save_model\KNN\data2_1', label)
