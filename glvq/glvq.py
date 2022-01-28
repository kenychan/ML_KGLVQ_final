

import numpy as np
import matplotlib.pyplot as plt


class Glvq:
    #https://www.geeksforgeeks.org/self-in-python-class/
    #self refer to the instance of the current class, kinda like this in java

    update_prototypes = np.array([])
    prototype_labels = np.array([])

    

    #about dimension :https://www.youtube.com/watch?v=vN5dAZrS58E&ab_channel=RyanChesler

    # define prototypes
    def prototype(self, input_data, data_labels, prototype_per_class):

        # prototype_labels are
        prototype_labels = np.unique(data_labels) #Find the unique elements of an array. 
        prototype_labels = np.repeat(prototype_labels,prototype_per_class) #prototype per class multiplication
        # prototypes are
        prototype_labels = np.expand_dims(prototype_labels, axis=1) #expand for comparison 
 
        
        expand_dimension = np.expand_dims(np.equal(prototype_labels, data_labels),
                                          axis=2)#expand to 3 - dimensional, and each dimension has a eg: 3x21 2d matrix 
        print(expand_dimension.shape)                                  
        count = np.count_nonzero(expand_dimension, axis=1) 
        proto = np.where(expand_dimension, input_data, 0)  
        prototypes = np.sum(proto, axis=1)/count    

        self.prt_labels = prototype_labels 
        return self.prototype_labels, prototypes

    def euclidean_dist(self, input_data, prototypes):
        expand_dimension = np.expand_dims(input_data, axis=1)
        distance = expand_dimension - prototypes
        distance_square = np.square(distance)
        sum_distance = np.sum(distance_square, axis=2)
        eu_dist = np.sqrt(sum_distance)
        return eu_dist

    def distance_plus(self, data_labels, prototype_labels,
                      prototypes, eu_dist):
        expand_dimension = np.expand_dims(prototype_labels, axis=1)
        label_transpose = np.transpose(np.equal(expand_dimension, data_labels))

        # distance of matching prototypes
        plus_dist = np.where(label_transpose, eu_dist, np.inf)
        d_plus = np.min(plus_dist, axis=1)

        # index of minimum distance for best matching prototypes
        w_plus_index = np.argmin(plus_dist, axis=1)
        w_plus = prototypes[w_plus_index] #which closest prototype to data x has the same label
        return d_plus, w_plus, w_plus_index

    def distance_minus(self, data_labels, prototype_labels,
                       prototypes, eu_dist):
        expand_dimension = np.expand_dims(prototype_labels, axis=1)
        label_transpose = np.transpose(np.not_equal(expand_dimension,
                                                    data_labels))

        # distance of non matching prototypes
        minus_dist = np.where(label_transpose, eu_dist, np.inf)
        d_minus = np.min(minus_dist, axis=1)

        # index of minimum distance for non best matching prototypes
        w_minus_index = np.argmin(minus_dist, axis=1)
        w_minus = prototypes[w_minus_index]
        return d_minus, w_minus, w_minus_index

    def classifier_function(self, d_plus, d_minus):
        classifier = (d_plus - d_minus) / (d_plus + d_minus) 
        return classifier

    def sigmoid(self, classifier_result, time):
        return (1/(1 + np.exp(-classifier_result * time))) 

    def change_in_w_plus(self, input_data, prototypes, learning_rate, classifier,
                         w_plus, w_plus_index, d_plus, d_minus,time_para):

        sai = (d_minus / (np.square(d_plus + d_minus))) * \
        (self.sigmoid(classifier,time_para)) * (1 - self.sigmoid(classifier,time_para)) #gradient(f/u)
        expand_dimension = np.expand_dims(sai, axis=1)
        change_w_plus = expand_dimension * (input_data - w_plus) * learning_rate #delta w_plus

        unique_w_plus_index = np.unique(w_plus_index)
        unique_w_plus_index = np.expand_dims(unique_w_plus_index, axis=1)

        add_row_change_in_w = np.column_stack((w_plus_index, change_w_plus))
        check = np.equal(add_row_change_in_w[:, 0], unique_w_plus_index)
        check = np.expand_dims(check, axis=2)
        check = np.where(check, change_w_plus, 0)
        sum_change_in_w_plus = np.sum(check, axis=1)
        return sum_change_in_w_plus, unique_w_plus_index

    def change_in_w_minus(self, input_data, prototypes, learning_rate, classifier,
                          w_minus, w_minus_index, d_plus, d_minus,time_para):

        sai = (d_plus / (np.square(d_plus + d_minus))) * (self.sigmoid(classifier,time_para)) * (1 - self.sigmoid(classifier,time_para))

        expand_dimension = np.expand_dims(sai, axis=1)
        change_w_minus = (expand_dimension) * (input_data - w_minus) * learning_rate

        unique_w_minus_index = np.unique(w_minus_index)
        unique_w_minus_index = np.expand_dims(unique_w_minus_index, axis=1)

        add_row_change_in_w = np.column_stack((w_minus_index, change_w_minus))
        check = np.equal(add_row_change_in_w[:, 0], unique_w_minus_index)
        check = np.expand_dims(check, axis=2)
        check = np.where(check, change_w_minus, 0)
        sum_change_in_w_minus = np.sum(check, axis=1)
        return sum_change_in_w_minus, unique_w_minus_index

    # plot  data
    def plot(self, input_data, data_labels, prototypes, prototype_labels):
        plt.scatter(input_data[:, 0], input_data[:, 1], c=data_labels, 
                    s=10,cmap='viridis') #cmap = convert data values to rgba
        plt.scatter(prototypes[:, 0], prototypes[:, 1], c=prototype_labels,
                    s=200, marker='P', edgecolor='black',linewidth=2)
       
           
                 

    # fit function
    def fit(self, input_data, data_labels, learning_rate, epochs,prototype_per_class):
        normalized_data = input_data #normalize?
        prototype_lables, prototypes = self.prototype(normalized_data, data_labels,
                                           prototype_per_class) #generate prototypes
        error = np.array([])    #cost function array 
        error_count = np.array([])  #error numbers of each iteration
        plt.figure()

        datasize = len(input_data)
        time_para = [1]
        time = 1
        for t in range(1,datasize):
            time = time * 1.0001
            time_para.append(time)

        time_para=np.array(time_para)
        

        for i in range(epochs):
            eu_dist = self.euclidean_dist(normalized_data, prototypes) 
            print("euclidean distance size:  " ,len(eu_dist))
            d_plus, w_plus, w_plus_index = self.distance_plus(data_labels,
                                                              prototype_lables,
                                                              prototypes,
                                                              eu_dist)
            d_minus, w_minus, w_minus_index = self.distance_minus(data_labels,
                                                                  prototype_lables,
                                                                  prototypes,
                                                                  eu_dist)

            print("data size: ",len(normalized_data))
            print("d_plus: ",len(d_plus))
            print("w_plus: ",len(w_plus))
            print("d_minus: ",len(d_minus))
            print("w_minus: ",len(w_minus))

            classifier = self.classifier_function(d_plus, d_minus)#if neg, then correct
            print("classifier size: ",len(classifier))# = data size

            sum_change_in_w_plus, unique_w_plus_index = self.change_in_w_plus( 
                normalized_data, prototypes, learning_rate, classifier,
                w_plus, w_plus_index,  d_plus, d_minus,time_para)

            update_w_p = np.add(np.squeeze(
                prototypes[unique_w_plus_index]), sum_change_in_w_plus) 
            np.put_along_axis(prototypes, unique_w_plus_index,  #put updated same label prototype into prototypes at this index and axis 0 
                              update_w_p, axis=0)

            sum_change_in_w_m, unique_w_minus_index = self.change_in_w_minus( 
                normalized_data, prototypes, learning_rate, classifier,
                w_minus, w_minus_index, d_plus, d_minus,time_para)

            update_w_m = np.subtract(np.squeeze(
                prototypes[unique_w_minus_index]), sum_change_in_w_m) 

            np.put_along_axis(
                prototypes, unique_w_minus_index, update_w_m, axis=0) #put updated different label prototype into prototypes at this index and axis 0
            

            err = np.sum(self.sigmoid(classifier,time_para), axis=0) 

            change_in_error = 0 

            if (i == 0):
                change_in_error = 0

            else:
                change_in_error = error[-1] - err 
            error = np.append(error, err) #append err to error
            print("Epoch : {}, Error : {} Error change : {}".format(
                i + 1, err, change_in_error))

            #plt.subplot(1, 2, 1,facecolor='white')
            #plt.cla()#so that the updates will not overlap
            #self.plot(normalized_data, data_labels, prototypes, prototype_lables) #left

            
            count  = np.count_nonzero(d_plus > d_minus)
            error_count = np.append(error_count,count)
            
            #lt.subplot(1, 2, 2,facecolor='black')
            #plt.plot(np.arange(i+1), error, marker="d") #right

            #plt.pause(0.1)

            #figName = 'LVQ_' + str(i) + '.png'
            #plt.savefig('result/'+figName)
            time_para = 1.0001*time_para
        #plt.show()#show last pic

        accuracy = np.count_nonzero(d_plus < d_minus) #Counts the number of non-zero values in the array 
        acc = accuracy / len(d_plus) * 100 
        #print(accuracy)
        #print(len(d_plus))
        print("error count array: ",error_count)
        print("accuracy = {}".format(acc))
        self.update_prototypes = prototypes 
        print(self.prototype_labels)
        return acc

  