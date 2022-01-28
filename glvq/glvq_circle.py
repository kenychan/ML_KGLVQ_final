
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from glvq import Glvq
import numpy as np

from sklearn.cluster import kmeans_plusplus
from sklearn.datasets import make_blobs
from sklearn.model_selection import ShuffleSplit,KFold
from sklearn import cluster, datasets


n_samples = 800
samples, labels = datasets.make_circles(n_samples=n_samples, factor=.8,
                                      noise=.08)




input_data = samples.copy() #make it static
data_label = labels.copy()

epochs = 200#iterations
learning_rate = 0.01 

glvq = Glvq()

ss = KFold(n_splits=10)
graph_index_list = []
for train_index, test_index in ss.split(input_data.data):
    graph_index_list.append(test_index)

result = []

for i in range(10):
        datalist = []
        labellist = []
        for index in graph_index_list[i]:
            datalist.append(input_data[index])
            labellist.append(data_label[index]) 
        acc = glvq.fit(datalist, labellist, learning_rate, epochs, 1)
        
        result.append(acc)

print("accuracy:",result)
print("Average accuracy:", np.mean(result))
print("Standard deviation:", np.std(result))




