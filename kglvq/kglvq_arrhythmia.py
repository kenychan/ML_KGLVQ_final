import pandas as pd
import numpy as np
from kglvq import kglvq
from sklearn.model_selection import ShuffleSplit,KFold
#https://archive-beta.ics.uci.edu/ml/datasets/Hepatitis
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/arrhythmia/arrhythmia.data')
data = df.iloc[:, 0:279]
label = df.iloc[:, 279]-1 # when label starts with 1

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
for sigma in np.arange(0.1,1.2,0.1):
    accuracy = []
    for i in range(10):
        datalist = []
        labellist = []
        for index in graph_index_list[i]:
            datalist.append(data[index])
            labellist.append(label[index]) 
        datalist = np.array(datalist)
        labellist = np.array(labellist)
        acc = clf.fit('arrhythmia',datalist,labellist, len(np.unique(labellist)), 1 ,0.01, epochs ,sigma)

        accuracy.append(acc)
    mean.append(np.mean(accuracy))
    std.append(np.std(accuracy))    


print(mean,'\n',std)    