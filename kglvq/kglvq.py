import numpy as np
import matplotlib.pyplot as plt
import itertools
from numpy import random



class kglvq:
    #https://www.geeksforgeeks.org/self-in-python-class/
    #self refer to the instance of the current class, kinda like this in java

    
    kernel_matrix = np.array([])
    coefficient_vectors = np.array([])

    #about dimension :https://www.youtube.com/watch?v=vN5dAZrS58E&ab_channel=RyanChesler


    def vec_normalize(self,arr):
        #arr1 = arr / arr.min()
        arr1 = arr / arr.sum()
        return arr1


    def coeff_initial(self,classnumber,prototype_per_class,unique_prototypes,samplesize):  #use random numbers to inital the first coefficient vectors of each class
        prototype_number = classnumber * prototype_per_class
        arr = [] #list
        for i in range(0,prototype_number): #coeff vectors for diffrent classes
            for x in range(0, samplesize):
                x = random.rand()
                arr.append(x)   

        arr = np.array(arr)
        arr = np.reshape(arr,(prototype_number,samplesize)) #reshape to a 2d matrix (12,3000)
        
        self.coefficient_vectors = np.apply_along_axis(self.vec_normalize,1,arr)
            #normalize each coefficient vector , sum up to 1 ,save to class 

        #define prototype labels by order 0000 1111 2222. their indices  = prototype indices
        prototype_labels = np.repeat(unique_prototypes,prototype_per_class)
        return prototype_labels #(12,)     








    #gaussian_kernelfunction
    def gaussian_kernelfunction(self,xi,xj,kernel_para):
        sigma = kernel_para #sigma should be changed to fit the data ,0.1
        dist = np.linalg.norm(xi-xj)#euclidean distance
        return np.exp((-(dist)**2)/2*(sigma**2))

        
    def kernelmatrix(self,inputdata,kernel_para):

        matrix = np.array([])#empty 1d array
        all_possible_pairs = list(itertools.product(inputdata, repeat=2)) #repetitive pairing
        arr = np.array(all_possible_pairs) # convert list to array

        for row in arr:
            paras = np.array([])
            for element in row:#2 elements in one row
                paras = np.append(paras,element)

            newparas = np.reshape(paras,(2,len(inputdata[0]))) #2xn matrix, cuz there's only 2 vectors,len(inputdata[0] to get attribute len
            kernel_result = self.gaussian_kernelfunction(newparas[0],newparas[1],kernel_para)  
            matrix= np.append(matrix,kernel_result)
            
        newmatrix = np.reshape(matrix,(len(inputdata),len(inputdata))) #2d NxN
        self.kernel_matrix=newmatrix #save matrix to class data
        #matrix[i][j] = required result, diagonal = 1

    def feature_space_distance_forall_samples(self,prototype_number,samplesize): 
        #calculate the distance between all samples and all prototypes, for initialization

        distance_arr = []
        # s: removed code redundancy
        for index in range(0, samplesize):
            distance = self.feature_space_distance_for_singlesample(prototype_number, index, samplesize)
            distance_arr.append(distance)
        distance_arr = np.array(distance_arr)
        distance_arr = np.reshape(distance_arr,(prototype_number,samplesize))
        distance_arr = np.transpose(distance_arr) # ret(samplesize, prototype_number)

        return distance_arr              

    def feature_space_distance_for_singlesample(self,prototype_number,index,samplesize): #distances of one sample to every prototype

        distance_arr = []

        for p in range(0,prototype_number):  #from prototype to sample  

            part1 = self.kernel_matrix[index][index] #diagonal = 1
            
            part2 = (self.coefficient_vectors[p]*self.kernel_matrix[index]).sum()

            sum2 = np.sum(np.outer(self.coefficient_vectors[p], self.coefficient_vectors[p]) * self.kernel_matrix)

            part3 = sum2
            
            distance =  part1 - (2*part2) + part3

            distance_arr.append(distance)

        distance_arr = np.array(distance_arr)


        return distance_arr           



    
    def distance_plus(self, data_labels, prototype_labels, 
                       distance):
        expand_dimension = np.expand_dims(prototype_labels, axis=1) #expaned to 2d array
        label_transpose = np.transpose(np.equal(expand_dimension, data_labels)) #samples with same label
    
        
        plus_dist = np.where(label_transpose, distance, np.inf) 
        

        d_plus = np.min(plus_dist, axis=1)#(samplesize,)
        w_plus_index = np.argmin(plus_dist, axis=1) 
       
        return d_plus, w_plus_index

    def distance_minus(self, data_labels, prototype_labels,
                        distance):
        expand_dimension = np.expand_dims(prototype_labels, axis=1)
        label_transpose = np.transpose(np.not_equal(expand_dimension, 
                                                    data_labels))

        # distance of non matching prototypes
        minus_dist = np.where(label_transpose, distance, np.inf)
        d_minus = np.min(minus_dist, axis=1)

        # index of minimum distance for non best matching prototypes
        w_minus_index = np.argmin(minus_dist, axis=1)
  
        return d_minus,  w_minus_index

    # define classifier function, <0 means correct classfication
    def classifier_function(self, d_plus, d_minus):
        classifier = (d_plus - d_minus) / (d_plus + d_minus) 
        return classifier #(samplesize,)

    # define sigmoid function
    def sigmoid(self, classifier_result, time_parameter): #xi = ξ
        return 1/(1 + np.exp((-time_parameter) * classifier_result)) 

    def update_ks(self,sample_index,prototype_plus,learning_rate,classifier_result,dk,dl,time_parameter):
        coeff = learning_rate * (self.sigmoid(classifier_result, time_parameter)) * (1 - self.sigmoid(classifier_result,time_parameter))
        #coeff is always the same, from xi to wj
        self.coefficient_vectors[prototype_plus]  =(1 - (coeff * ((4*dl[sample_index])/(dk[sample_index]+dl[sample_index]))))*self.coefficient_vectors[prototype_plus]   #(3000,) normalise all weights for this 
        #self.coefficient_vectors[prototype_plus] => (samplesize,)
        #PS: here for each weight in vector, the complete list of dk, dl for each data sample is needed, otherwise the classification results will show errors only
        self.coefficient_vectors[prototype_plus][sample_index] = (1 - (coeff * ((4*dl[sample_index])/(dk[sample_index]+dl[sample_index]))))*self.coefficient_vectors[prototype_plus][sample_index]\
            + (coeff * ((4*dl[sample_index])/(dk[sample_index]+dl[sample_index])))
        #self.coefficient_vectors[prototype_plus] = self.vec_normalize(self.coefficient_vectors[prototype_plus])  

        
        #override, single update to right sample's p coeff weight

    def update_kl(self,sample_index,prototype_minus,learning_rate,classifier_result,dk,dl,time_parameter):
        #coeff = learning_rate * (self.sigmoid(classifier_result,time_parameter)) * (1 - self.sigmoid(classifier_result,time_parameter)) #(3000,) coff of all samples
        coeff = learning_rate * (self.sigmoid(classifier_result, time_parameter)) * (1 - self.sigmoid(classifier_result,time_parameter))
        self.coefficient_vectors[prototype_minus]  =(1 + (coeff * ((4*dk[sample_index])/(dk[sample_index]+dl[sample_index]))))*self.coefficient_vectors[prototype_minus]   #(3000,)
        #self.coefficient_vectors[prototype_plus] => (samplesize,)

        self.coefficient_vectors[prototype_minus][sample_index] = (1 + (coeff * ((4*dk[sample_index])/(dk[sample_index]+dl[sample_index]))))*self.coefficient_vectors[prototype_minus][sample_index]\
            -  (coeff* ((4*dk[sample_index])/(dk[sample_index]+dl[sample_index])))
        #override, single update to right sample's p coeff        
        #self.coefficient_vectors[prototype_minus] = self.vec_normalize(self.coefficient_vectors[prototype_minus])  

        



    # plot  data
    def plot(self, input_data, data_labels, prototypes, prototype_labels):
        plt.scatter(input_data[:, 0], input_data[:, 1], c=data_labels, #0=x ,1=y
                    s=10,cmap='viridis') #cmap = convert data values to rgba
        plt.scatter(prototypes[:, 0], prototypes[:, 1], c=prototype_labels,
                    s=200, marker='P', edgecolor='black',linewidth=2,alpha=0.6)
       
           
    def visualize_2d(self,inputdata):   
        prototype2d = np.dot(self.coefficient_vectors,inputdata)  
        return prototype2d



    # fit function
    def fit(self, dataname ,inputdata, data_labels, classnumber,prototype_per_class, learning_rate, epochs ,kernel_para):

        input_data = inputdata #normalise? nope
        samplesize = len(input_data)
        prototype_number = classnumber * prototype_per_class
        unique_prototypes = np.unique(data_labels)
        prototype_labels = self.coeff_initial(classnumber,prototype_per_class,unique_prototypes,samplesize)
        
        #initial first coeff vectors for prototypes and their labels
        self.kernelmatrix(input_data,kernel_para)

        distance = self.feature_space_distance_forall_samples(prototype_number,samplesize)
        distance_plus, prototype_plus_index = self.distance_plus(data_labels,prototype_labels,distance)
        distance_minus, prototype_minus_index = self.distance_minus(data_labels,prototype_labels,distance)
        classifier = self.classifier_function(distance_plus, distance_minus)
        #initialize distance and closest distances and classifier results for all samples, 
        #later updates will only update the changes for each single sample, to save calculation time


        cost_function_arr = np.array([])    #cost function array 
        error_count = np.array([])  #error numbers of each iteration
        plt.figure()
               
        time_para = [1]
        time = 1
        for t in range(1,samplesize):
            time = time * 1.0001
            time_para.append(time)

        time_para=np.array(time_para)
        for i in range(epochs): #epochs  

            
            for sample_index_t in range(0,samplesize):
                #for index in range(prototype_number):
                #    print("sum of weight vector for prototype {}:".format(index), self.coefficient_vectors[index].sum())
                distance[sample_index_t] = self.feature_space_distance_for_singlesample(prototype_number,sample_index_t,samplesize) 
                distance_plus[sample_index_t], prototype_plus_index[sample_index_t] = self.distance_plus(data_labels[sample_index_t],prototype_labels,distance[sample_index_t])
                distance_minus[sample_index_t], prototype_minus_index[sample_index_t] = self.distance_minus(data_labels[sample_index_t],prototype_labels,distance[sample_index_t])
                classifier[sample_index_t] = self.classifier_function(distance_plus[sample_index_t], distance_minus[sample_index_t])
                #updates for each single sample
                #print("data:{}'s closetest same label prototype:{}, closetest different label prototype:{} ".format(sample_index_t,prototype_plus_index[sample_index_t],prototype_minus_index[sample_index_t]))

                self.update_ks(sample_index_t,prototype_plus_index[sample_index_t],learning_rate,classifier[sample_index_t],distance_plus,distance_minus,time_para[sample_index_t])
                self.update_kl(sample_index_t,prototype_minus_index[sample_index_t],learning_rate,classifier[sample_index_t],distance_plus,distance_minus,time_para[sample_index_t])

                
            cost_function = np.sum(self.sigmoid(classifier,time_para), axis=0) 
            
            change_in_cost = 0 

            if (i == 0):
                change_in_cost = 0

            else:
                change_in_cost = cost_function_arr[-1] - cost_function 

            cost_function_arr = np.append(cost_function_arr, cost_function) #append single cost to cost arr
            print("Epoch : {}, Cost : {} Cost change : {}".format(
                i + 1, cost_function, change_in_cost))

            #plt.subplot(1, 2, 1,facecolor='white')
            #plt.cla()#so that the updates will not overlap

            
            prototype2d = self.visualize_2d(input_data) #visualize 2d data with the final coeff
            self.plot(input_data, data_labels, prototype2d, prototype_labels) #left pic 

            
            count  = np.count_nonzero(distance_plus > distance_minus) #count wrong classification
            error_count = np.append(error_count,count)
            
            #plt.subplot(1, 2, 2,facecolor='black')
            #plt.plot(np.arange(i+1), cost_function_arr, marker="d") #right pic

            #plt.pause(0.1)
            time_para = 1.0001 * time_para 



        accuracy = np.count_nonzero(distance_plus < distance_minus) #Counts the number of non-zero values in the array 
        acc = accuracy / len(distance_plus) * 100 
        #print(accuracy)
        #print(len(d_plus))
        print("error number per epoch: ",error_count) 
        print("accuracy = {}".format(acc))

        #plt.annotate('dataset: {}, sigma :{}, samplesize: {}, learning rate:{}, \nclass numer: {}, prototype per class: {}, epoch: {}'.format(dataname,kernel_para,samplesize,learning_rate,classnumber,prototype_per_class,i+1), xy=(-1.2, 1.05), xycoords='axes fraction')
        #plt.annotate('accuracy: {}'.format(acc), xy=(-1.2, 1.01), xycoords='axes fraction')        
        #figName = 'KGLVQ_'+dataname+ '_'+ str(samplesize) +'data_samples__'+ str(prototype_number)+'prototypes__' + str(i) + 'epochs'+'_'+str(acc)+'accuracy.png'
        #plt.savefig('result/'+figName)
        #plt.show()#show last pic
        return acc
  