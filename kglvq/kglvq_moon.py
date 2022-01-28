
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from kglvq import kglvq
import numpy as np

from sklearn.cluster import kmeans_plusplus
from sklearn.datasets import make_blobs
from sklearn.model_selection import ShuffleSplit,KFold
from sklearn import cluster, datasets

n_samples = 600
n_class = 2 #how many clusters to make , aka class numbers 
samples, labels = datasets.make_moons(n_samples=n_samples, shuffle=True, noise=0.08, random_state=0)

input_data = samples.copy() #make it static
data_label = labels.copy()

epochs = 200#iterations
learning_rate = 0.01 #也叫升级公式中的gain factor, 影响prototype 的动量, paper 中是0.001

clf = kglvq()

#inputsample_cut1, inputsample_cut2, sample_cut1_labels, sample_cut2_labels = train_test_split(input_data,
#                                                    data_label,
#                                                    test_size=0.3,
#                                                    random_state=0)
#从数据中抽样
#kernel_para = 1.1

#glvq.fit('kmeans',input_data,data_label, n_class, prototype_per_class ,learning_rate, epochs ,kernel_para)


#glvq.fit('kmeans',input_data,data_label, n_class, 1 ,learning_rate, epochs ,0.09)
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
        acc = clf.fit('moon',datalist,labellist, n_class, prototype_per_class ,learning_rate, epochs ,1.1)

        accuracy.append(acc)
    mean.append(np.mean(accuracy))
    std.append(np.std(accuracy))    


print(mean,'\n',std)    




