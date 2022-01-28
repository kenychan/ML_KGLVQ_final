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


        prototype_labels = np.repeat(unique_prototypes,prototype_per_class)
        return prototype_labels #(12,)     



    def feature_space_distance_forall_samples(self,prototype_number,samplesize): 

        distance_arr = []
        # s: removed code redundancy
        for index in range(0, samplesize):
            distance = self.feature_space_distance_for_singlesample(prototype_number, index, samplesize)
            distance_arr.append(distance)
        distance_arr = np.array(distance_arr)
        distance_arr = np.reshape(distance_arr,(prototype_number,samplesize))
        distance_arr = np.transpose(distance_arr) # ret(samplesize, prototype_number)
        return distance_arr           

    def feature_space_distance_for_singlesample(self,prototype_number,index,samplesize): 

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
        expand_dimension = np.expand_dims(prototype_labels, axis=1) 
        label_transpose = np.transpose(np.equal(expand_dimension, data_labels)) 
       
        plus_dist = np.where(label_transpose, distance, np.inf) 
        
        d_plus = np.min(plus_dist, axis=1)
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

    # define classifier function
    def classifier_function(self, d_plus, d_minus):
        classifier = (d_plus - d_minus) / (d_plus + d_minus)
        return classifier 

    # define sigmoid function
    def sigmoid(self, classifier_result, time_parameter): #xi = ξ
        return 1/(1 + np.exp((-time_parameter) * classifier_result))


    def update_ks(self,sample_index,prototype_plus,learning_rate,classifier_result,dk,dl,time_parameter):
        coeff = learning_rate * (self.sigmoid(classifier_result, time_parameter)) * (1 - self.sigmoid(classifier_result,time_parameter))
        #coeff is always the same, from xi to wj
        self.coefficient_vectors[prototype_plus]  =(1 - (coeff * ((4*dl[sample_index])/(dk[sample_index]+dl[sample_index]))))*self.coefficient_vectors[prototype_plus]   
        #PS: here for each weight in vector, the complete list of dk, dl for each data sample is needed, otherwise the classification results will show errors only
        self.coefficient_vectors[prototype_plus][sample_index] = (1 - (coeff * ((4*dl[sample_index])/(dk[sample_index]+dl[sample_index]))))*self.coefficient_vectors[prototype_plus][sample_index]\
            + (coeff * ((4*dl[sample_index])/(dk[sample_index]+dl[sample_index])))
        #self.coefficient_vectors[prototype_plus] = self.vec_normalize(self.coefficient_vectors[prototype_plus])  

        
        #override, single update to right sample's p coeff weight

    def update_kl(self,sample_index,prototype_minus,learning_rate,classifier_result,dk,dl,time_parameter):
        #coeff = learning_rate * (self.sigmoid(classifier_result,time_parameter)) * (1 - self.sigmoid(classifier_result,time_parameter)) 
        coeff = learning_rate * (self.sigmoid(classifier_result, time_parameter)) * (1 - self.sigmoid(classifier_result,time_parameter))
        self.coefficient_vectors[prototype_minus]  =(1 + (coeff * ((4*dk[sample_index])/(dk[sample_index]+dl[sample_index]))))*self.coefficient_vectors[prototype_minus]  

        self.coefficient_vectors[prototype_minus][sample_index] = (1 + (coeff * ((4*dk[sample_index])/(dk[sample_index]+dl[sample_index]))))*self.coefficient_vectors[prototype_minus][sample_index]\
            -  (coeff* ((4*dk[sample_index])/(dk[sample_index]+dl[sample_index])))
        #override, single update to right sample's p coeff        
        #self.coefficient_vectors[prototype_minus] = self.vec_normalize(self.coefficient_vectors[prototype_minus])  

      
  

    # fit function
    def fit(self, dataname ,kernelmatrix, graphlabels,prototype_per_class, learning_rate, epochs):

        samplesize = len(graphlabels)
        unique_prototypes = np.unique(graphlabels)
        classnumber = len(unique_prototypes)
        prototype_number = classnumber * prototype_per_class
        prototype_labels = self.coeff_initial(classnumber,prototype_per_class,unique_prototypes,samplesize)
        
        #initial first coeff vectors for prototypes and their labels
        self.kernel_matrix = kernelmatrix

        distance = self.feature_space_distance_forall_samples(prototype_number,samplesize)
        distance_plus, prototype_plus_index = self.distance_plus(graphlabels,prototype_labels,distance)
        distance_minus, prototype_minus_index = self.distance_minus(graphlabels,prototype_labels,distance)
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
                distance_plus[sample_index_t], prototype_plus_index[sample_index_t] = self.distance_plus(graphlabels[sample_index_t],prototype_labels,distance[sample_index_t])
                distance_minus[sample_index_t], prototype_minus_index[sample_index_t] = self.distance_minus(graphlabels[sample_index_t],prototype_labels,distance[sample_index_t])
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

            cost_function_arr = np.append(cost_function_arr, cost_function) 
            print("Epoch : {}, Cost : {} Cost change : {}".format(
                i + 1, cost_function, change_in_cost))


            count  = np.count_nonzero(distance_plus > distance_minus) 
            error_count = np.append(error_count,count) 
       

            #plt.plot(np.arange(i+1), cost_function_arr, marker="d") 
            #plt.pause(0.1)

            time_para = 1.0001 * time_para 


        accuracy = np.count_nonzero(distance_plus < distance_minus) 
        acc = accuracy / len(distance_plus) * 100 
        #print(accuracy)
        #print(len(d_plus))
        print("error number per epoch: ",error_count) 
        print("accuracy = {}".format(acc))
    
        #plt.annotate('dataset: {}, samplesize: {}, learning rate:{}, \nclass numer: {}, prototype per class: {}, epoch: {}'.format(dataname,samplesize,learning_rate,classnumber,prototype_per_class,i+1), xy=(0.05, 0.85), xycoords='axes fraction')
        #plt.annotate('accuracy: {}'.format(acc), xy=(0.05, 0.78), xycoords='axes fraction')

        
        #figName = 'KGLVQ_'+dataname+ '_'+ str(samplesize) +'data_samples__'+ str(classnumber)+'classeds__'+str(prototype_per_class)+'prototype_per_class__' + str(i+1) + 'epochs'+'_'+str(acc)+'accuracy.png'
        #plt.savefig('result/'+figName)
        #plt.show()#show last pic
        return acc


  