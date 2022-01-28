
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from kglvq import kglvq
import numpy as np

from sklearn.cluster import kmeans_plusplus
from sklearn.datasets import make_blobs
from sklearn.model_selection import ShuffleSplit,KFold
from sklearn import cluster, datasets

n_class = 2 
n_samples = 800
samples, labels = datasets.make_circles(n_samples=n_samples, factor=.2,
                                      noise=.01)


input_data = samples.copy() 
data_label = labels.copy()

epochs = 200
learning_rate = 0.01 
clf = kglvq()

ss = KFold(n_splits=10)
graph_index_list = []
for train_index, test_index in ss.split(input_data.data):
    graph_index_list.append(test_index)

result = []

for sigma  in np.arange(0.1,1.2,0.1):
    for i in range(10):
        datalist = []
        labellist = []
        for index in graph_index_list[i]:
            datalist.append(input_data[index])
            labellist.append(data_label[index]) 
        datalist = np.array(datalist)
        labellist = np.array(labellist)
        acc = clf.fit('kmeans',datalist,labellist, n_class, 1 ,learning_rate, epochs ,sigma)

        result.append(acc)


print("accuracy:",result)
print("Average accuracy:", np.mean(result))
print("Standard deviation:", np.std(result))




