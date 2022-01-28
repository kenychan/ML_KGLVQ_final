import pandas as pd
import numpy as np
from kglvq import kglvq
from sklearn.model_selection import ShuffleSplit,KFold
#https://archive-beta.ics.uci.edu/ml/datasets/Hepatitis
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/hepatitis/hepatitis.data',usecols=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19])
data = df.iloc[:, 1:20]
label = df.iloc[:, 0]-1 # 提前处理label

data = data.replace("?", 0)

data = np.array(data).astype(float)
label = np.array(label).astype(int)


clf = kglvq()
ss = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
graph_index_list = []
for train_index, test_index in ss.split(data):
    graph_index_list.append(test_index)

mean = []
std = []

epochs = 200
for prototype_per_class in range(1,4):
    accuracy = []
    for i in range(10):
        datalist = []
        labellist = []
        for index in graph_index_list[i]:
            datalist.append(data[index])
            labellist.append(label[index]) 
        datalist = np.array(datalist)
        labellist = np.array(labellist)
        acc = clf.fit('hepatitis',datalist,labellist, len(np.unique(labellist)), prototype_per_class ,0.01, epochs ,0.6)

        accuracy.append(acc)
    mean.append(np.mean(accuracy))
    std.append(np.std(accuracy))    


print(mean,'\n',std)    