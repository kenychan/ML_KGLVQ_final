from matplotlib.colors import same_color
from grakel.kernels import WeisfeilerLehman, VertexHistogram
from grakel.datasets import fetch_dataset
from sklearn.model_selection import ShuffleSplit
from kglvq_for_graph import kglvq
import numpy as np
#prototype_per_class=1
learning_rate = 0.01
epochs = 400

#inputdata = fetch_dataset("MSRC_9", verbose=False)
#inputdata = fetch_dataset("ENZYMES", verbose=False)
#inputdata = fetch_dataset("COX2",verbose=False)
#inputdata = fetch_dataset("BZR",verbose=False)
inputdata = fetch_dataset("Cuneiform",verbose=False)
clf = kglvq()

def fit(prototype_per_class__,kernel_iteration,data,label):
    Graph, graphlabels = data,label
    gk = WeisfeilerLehman(n_iter=kernel_iteration, base_graph_kernel=VertexHistogram, normalize=True)
    kernelmatrix=gk.fit_transform(Graph)


    #accuracy = clf.fit('MSRC_9, n_iter=1',kernelmatrix, graphlabels, prototype_per_class__ ,learning_rate, epochs )
    #accuracy = clf.fit('ENZYMES, n_iter=1',kernelmatrix, graphlabels, prototype_per_class__ ,learning_rate, epochs )
    #accuracy = clf.fit('COX2',kernelmatrix, graphlabels, prototype_per_class ,learning_rate, epochs )
    #accuracy = clf.fit('BZR',kernelmatrix, graphlabels, prototype_per_class ,learning_rate, epochs )
    accuracy = clf.fit('Cuneiform, n_iter=30',kernelmatrix, graphlabels, prototype_per_class__ ,learning_rate, epochs )
    # File "/Users/guochen/opt/miniconda3/lib/python3.8/site-packages/grakel/datasets/base.py", line 302, in read_data
    #    node_labels[ngc[i]][i] = int(line[:-1])  => can run without 'int'

    return accuracy



ss = ShuffleSplit(n_splits=10, test_size=0.2, random_state=10)

graph_index_list = []
for train_index, test_index in ss.split(inputdata.data):
    graph_index_list.append(test_index)



mean = []
std = []

for prototype_per_class in range(1,4):
    accuracy = []
    for i in range(10):
        graphdatalist = []
        graphlabellist = []
        for index in graph_index_list[i]:
            graphdatalist.append(inputdata.data[index])
            graphlabellist.append(inputdata.target[index])#if graph label starts from 1 ,then -1
        graphdatalist = np.array(graphdatalist)
        graphlabellist = np.array(graphlabellist)
        acc = fit(prototype_per_class,4,graphdatalist, graphlabellist)    
        accuracy.append(acc)
    mean.append(np.mean(accuracy))
    std.append(np.std(accuracy))    

print(mean,'\n',std)    
