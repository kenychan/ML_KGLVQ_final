
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from glvq import Glvq
import numpy as np

from sklearn.cluster import kmeans_plusplus
from sklearn.datasets import make_blobs
from sklearn.model_selection import ShuffleSplit,KFold
from sklearn import cluster, datasets

n_samples = 800 
n_class = 4 
samples, labels = make_blobs(n_samples=n_samples, centers=n_class, cluster_std=1, random_state=0)




input_data = samples.copy() #make it static
data_label = labels.copy()

epochs = 200#iterations
learning_rate = 0.01 

glvq = Glvq()


ss = KFold(n_splits=10)

graph_index_list = []
for train_index, test_index in ss.split(input_data.data):
    graph_index_list.append(test_index)

mean = []
std = []

for prototype_per_class  in range(1,4):
    accuracy = []
    for i in range(10):
        datalist = []
        labellist = []
        for index in graph_index_list[i]:
            datalist.append(input_data[index])
            labellist.append(data_label[index]) 
        datalist = np.array(datalist)
        labellist = np.array(labellist)
        acc = glvq.fit(datalist, labellist, learning_rate, epochs, prototype_per_class)

        accuracy.append(acc)
    mean.append(np.mean(accuracy))
    std.append(np.std(accuracy))    


print(mean,'\n',std)   


