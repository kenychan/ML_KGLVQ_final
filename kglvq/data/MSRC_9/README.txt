README for datasets MSRC_9


=== Usage ===

This folder contains the following comma separated text files 
(replace DS by the name of the dataset):

n = total number of nodes
m = total number of edges
N = number of graphs

(1) 	DS_A.txt (m lines) 
	sparse (block diagonal) adjacency matrix for all graphs,
	each line corresponds to (row, col) resp. (node_id, node_id)

	总共m行 连线, 按顺序, 表述连线两端的node id

(2) 	DS_graph_indicator.txt (n lines)
	column vector of graph identifiers for all nodes of all graphs,
	the value in the i-th line is the graph_id of the node with node_id i

	总共n个node,n行, 按顺序, 每个node都属于哪个graph, 最后的数字就是总共就多少graph
		(一个数字i重复多少次,就代表该图i里有几个node)

(3) 	DS_graph_labels.txt (N lines) 
	class labels for all graphs in the dataset,
	the value in the i-th line is the class label of the graph with graph_id i
	
	N个Graph,按顺序, 每个graph的class label (真正使用的class)

(4) 	DS_node_labels.txt (n lines)
	column vector of node labels,
	the value in the i-th line corresponds to the node with node_id i
	
	n个node,按顺序, 每个node的label (不是上面的class label,而是是相当于每个node的值)
	=>实际用于dot product的? 每两个graph 的所有node的值对应计算, node数量不对等, 设置=0


There are OPTIONAL files if the respective information is available:

(5) 	DS_edge_labels.txt (m lines; same size as DS_A_sparse.txt)
	labels for the edges in DS_A_sparse.txt 

(6) 	DS_edge_attributes.txt (m lines; same size as DS_A.txt)
	attributes for the edges in DS_A.txt 

(7) 	DS_node_attributes.txt (n lines) 
	matrix of node attributes,
	the comma seperated values in the i-th line is the attribute vector of the node with node_id i

(8) 	DS_graph_attributes.txt (N lines) 
	regression values for all graphs in the dataset,
	the value in the i-th line is the attribute of the graph with graph_id i


=== Description of the dataset === 

The original image database of all MSRC datasets can be found here:
http://research.microsoft.com/en-us/projects/objectclassrecognition/

It is licensed under the  Microsoft Research Digital Image license agreement ("MSR-DILA"), 
which can also be downloaded from the above website.  


Additional information: 

DS_map_id2im.txt (list of image names)


=== Previous Use of the Dataset ===

Neumann, M., Patricia, N., Garnett, R., Kersting, K.: Efficient Graph Kernels by 
Randomization. In: P.A. Flach, T.D. Bie, N. Cristianini (eds.) ECML/PKDD, Notes in 
Computer Science, vol. 7523, pp. 378-393. Springer (2012).

Neumann, M., Garnett R., Bauckhage Ch., Kersting K.: Propagation Kernels: Efficient Graph 
Kernels from Propagated Information. Under review at MLJ.


=== References ===

Neumann, M., Patricia, N., Garnett, R., Kersting, K.: Efficient Graph Kernels by 
Randomization. In: P.A. Flach, T.D. Bie, N. Cristianini (eds.) ECML/PKDD, Notes in 
Computer Science, vol. 7523, pp. 378-393. Springer (2012).



