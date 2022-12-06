# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 18:52:27 2022

@author: dell
"""

#Complete procedure of paper
import time 
start = time.perf_counter() 
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']


#MS-VRs generation
def generate_centers_r(center_num, dist_min, dist_max):
    use_pts = np.random.uniform(25,275,(1,2))
    while len(use_pts) < center_num:  
        pts = np.random.uniform(25,275,(1,2))              
        m = pts
        n = use_pts
        d = ed(m,n)
        d_min = min(d)
        
        if dist_min <= d_min <= dist_max:
            use_pts = np.r_[n,m]
            if len(use_pts) >= center_num:
                break            
    centers = np.array(use_pts)
    r = np.random.randint(25,50,(center_num,1))
    return centers,r



#Draw circular MS-VRs
def plot_circle(centers,r,axes):
    N = centers.shape[0]  
    for i in range(N):
        theta = np.arange(0, 2*np.pi, 0.01)  
        x = centers[i,0] + r[i] * np.cos(theta)
        y = centers[i,1] + r[i] * np.sin(theta)
        axes.plot(x, y, 'b-', alpha=0.8) 
        plt.plot(centers[:,0],centers[:,1],'o',c='red', ms=6,mec='r')
    plt.plot(centers[0,0],centers[0,1],'o',c='red',ms=6,mec='r',label = 'Cluster centre')    


       
#Calculate the Euclidean distance from a point to each point in a point set
def ed(m, n):
    # 1-dimensional array
    dist = np.sqrt(np.sum(np.asarray(m - n) ** 2,axis=1))
    return dist



#Generate Location-VR dataset
def Labeled_points(centers, r, x_axis, MSVR_num, Use_points_num, Random_points_num):    
    #generate beacon users
    beacon_user_list = []
    while len(beacon_user_list) < Use_points_num:   
        pts = np.random.uniform(0,300,(Random_points_num,2))       
        for i in range(Random_points_num):
            m = pts[i]
            n = centers
            distances = ed(m,n).reshape(MSVR_num,1) 
            #distances = np.reshape(distances,(MSVR_num,1)) 
            diff_dists = distances - r
            dist_min = min(diff_dists)
            if dist_min < 0:
                beacon_user_list.append(m)
                if len(beacon_user_list) >= Use_points_num:
                    break
    #beacon user set
    Use_pts = np.array(beacon_user_list)#2-dimensional array
    
    #generate vector labels
    dists_list = []
    for i in range(Use_points_num):
        m = Use_pts[i]
        n = centers
        distances = ed(m,n).reshape(MSVR_num,1)
        diff_dists = distances - r
        new_dists = np.int64(diff_dists < 0).reshape(1,MSVR_num)
        dists_list.append(new_dists)
    np_new_dists = np.array(dists_list) #3-dimensional array
    
    #vector labels(2-dimensional array)
    np_L = np_new_dists.reshape(Use_points_num,MSVR_num)
    list_L = list([tuple(t) for t in np_L])
    dict_L = dict((t, list_L.count(t)) for t in list_L)
    np_keys = np.array(list(dict_L.keys())) 
    
    #Number of VR regions
    vr_num = len(np_keys)
    
    #Convert vector labels to digital labels
    labels_list = []
    for i in range(len(np_L)):
        for j in range(len(np_keys)):
            if np.array_equal(np_L[i],np_keys[j]):
                labels_list.append(j)
    #digital labels
    np_labels = np.array(labels_list).reshape(-1,1)
    
    
    #Divide beacon user set into known point set and test point set
    Known_labeled_pts_list = []
    Known_np_keys_list = []
    Known_pts_list = []
    Known_np_labels_list = []
    Test_pts_list = []
    Test_np_labels_list = []
    known_num_list = x_axis
    for i in range(len(known_num_list)):
        known_num = known_num_list[i]
        known_pts = Use_pts[:known_num] #known point set
        known_np_labels = np_labels[:known_num] #label of known point set
        test_pts = Use_pts[known_num:] #test point set
        test_np_labels = np_labels[known_num:] #label of test point set
        
        Known_pts_list.append(known_pts)
        Known_np_labels_list.append(known_np_labels)
        Test_pts_list.append(test_pts)
        Test_np_labels_list.append(test_np_labels)
        
        
        #divide known point set into multiple sub point sets according to the VR regions
        list_known_L = list([tuple(t) for t in known_np_labels])
        dict_known_L = dict((t, list_known_L.count(t)) for t in list_known_L)
        known_np_keys = np.array(list(dict_known_L.keys()))

        known_labeled_pts_list = []
        for i in range(len(known_np_keys)):    
            knwon_subpts_list = []
            for j in range(len(known_np_labels)):
                if np.array_equal(known_np_keys[i],known_np_labels[j]):
                     knwon_subpts_list.append(known_pts[j])                 
            known_labeled_pts_list.append(np.array(knwon_subpts_list))
        known_labeled_pts = np.array(known_labeled_pts_list)
        Known_labeled_pts_list.append(known_labeled_pts) 
        Known_np_keys_list.append(known_np_keys)
    
    return Known_labeled_pts_list,Known_pts_list, Known_np_labels_list,Test_pts_list,Test_np_labels_list,vr_num



#Data generation
def generate_dataset(Known_labeled_pts_list,Known_pts_list,Known_np_labels_list,Test_pts_list,Test_np_labels_list):
    np_train_data_list = []
    np_test_data_list = []
    for T in range(len(Known_labeled_pts_list)): 
        #calculate the VR region centroids  
        subpointsets_mean_list = []
        for i in range(len(Known_labeled_pts_list[T])): 
            #if len(Known_labeled_pts[i]) !=0:
            m = np.mean(Known_labeled_pts_list[T][i], axis=0) 
            subpointsets_mean_list.append(m)
        np_subsets_mean = np.array(subpointsets_mean_list)
        
        
        #generate training point dataset
        known_dists_list = []
        for i in range(len(Known_pts_list[T])):
            m = Known_pts_list[T][i] 
            n = np_subsets_mean
            distances = ed(m,n)
            known_dists_list.append(distances)
        np_known_dists = np.array(known_dists_list) 
        #np.insert(arr,obj,value,axis=None)
        index0_known_data = Known_np_labels_list[T].reshape(1,-1)
        np_known_data = np.insert(np_known_dists,0,index0_known_data,axis=1)
        np_train_data_list.append(np_known_data)
        
        
        #generate test point dataset
        test_dists_list = []
        for i in range(len(Test_pts_list[T])):
            m = Test_pts_list[T][i] 
            n = np_subsets_mean
            distances = ed(m,n)
            test_dists_list.append(distances)
        np_test_dists = np.array(test_dists_list)
        #np.insert(arr,obj,value,axis=None)
        index0_test_data = Test_np_labels_list[T].reshape(1,-1)
        np_test_data = np.insert(np_test_dists,0,index0_test_data,axis=1)
        np_test_data_list.append(np_test_data)        
    return np_train_data_list, np_test_data_list



#Data pre-processing
def pre_process(load_data,units_num):
    data_list = []
    labels_list = []   
    for T in range(len(load_data)):
        #expand the dimension of data to make it consistent
        while len(load_data[T][0]) < units_num+1:          
            load_data[T] = np.insert(load_data[T], len(load_data[T][0]), 0, axis=1)
            if len(load_data[T][0]) >= units_num+1:
                break
                    
        #normalizing
        np.random.shuffle(load_data[T])
        data = load_data[T][:,1:] / 300 
        data_list.append(data)
        
        #convert digital labels to one-hot coding
        labels = load_data[T][:,0]
        one_hot_labels = tf.keras.utils.to_categorical(labels)
        while len(one_hot_labels[0]) < units_num:
            one_hot_labels = np.insert(one_hot_labels,len(one_hot_labels[0]),0,axis=1)
            if len(one_hot_labels[0]) >= units_num:
                break
        labels_list.append(one_hot_labels)    
    return data_list, labels_list



#Voronoi Cell partition
def Test_acc(Test_pts_list,Test_np_labels_list,Known_pts_list,Known_np_labels_list):
    Accuracy = []
    for T in range(len(Known_pts_list)): 
        true_num = 0
        for i in range(len(Test_pts_list[T])):
            m = Test_pts_list[T][i] 
            n = Known_pts_list[T]
            distances = ed(m,n)
            dists_min_index = np.argmin(distances)
            if np.array_equal(Test_np_labels_list[T][i], Known_np_labels_list[T][dists_min_index]):
                true_num += 1
            else:
                true_num = true_num       
        acc = (true_num*100) / len(Test_pts_list[T])
        Accuracy.append(acc)
    return Accuracy



#Neural network model(VR-Net1, 1-hidden layer)
class MyNetwork(tf.keras.Model):
    def __init__(self, units_num):
        super().__init__()
        '''
        When a popular kwarg input_shape is passed, then keras will create an input layer to insert before the current layer. 
        This can be treated equivalent to explicitly defining an InputLayer.
        '''
        #Hidden layer
        self.Dense1 = tf.keras.layers.Dense(units = units_num,activation = "sigmoid",
                                            kernel_initializer = tf.zeros_initializer(),
                                            bias_initializer = tf.zeros_initializer(),
                                            input_shape = (units_num,))
        
        #Output layer
        self.Output = tf.keras.layers.Dense(units = units_num)
        #add Softmax layer
        self.softmax = tf.keras.layers.Softmax()
        
    def call(self,input):
        x = input
        x = self.Dense1(x)
        x = self.Output(x)
        output = self.softmax(x)
        return output



#Neural network model(VR-Net2, 1-hidden layer and increase width)
class MyNetwork2(tf.keras.Model):
    def __init__(self, units_num2):
        super().__init__()
        
        #Hidden layer
        self.Dense1 = tf.keras.layers.Dense(units = units_num2,activation = "sigmoid",kernel_initializer = tf.zeros_initializer(),
                                            bias_initializer = tf.zeros_initializer(),input_shape = (units_num2,))
        
        #Output layer
        self.Output = tf.keras.layers.Dense(units = units_num2)

        #add Softmax layer
        self.softmax = tf.keras.layers.Softmax()
        
    def call(self,input):
        x = input
        x = self.Dense1(x)
        x = self.Output(x)
        output = self.softmax(x)
        return output



#Neural network model(VR-Net3, 2-hidden layers)
class MyNetwork3(tf.keras.Model):
    def __init__(self, units_num):
        super().__init__()
        
        #全连接层(隐藏层)
        self.Dense1 = tf.keras.layers.Dense(units = units_num,activation = "tanh",
                                            kernel_initializer = tf.zeros_initializer(),
                                            bias_initializer = tf.zeros_initializer(),
                                            input_shape = (units_num,))
        
        self.Dense2 = tf.keras.layers.Dense(units = units_num,activation = "sigmoid")
        
        #Output layer
        self.Output = tf.keras.layers.Dense(units = units_num)

        #add Softmax layer
        self.softmax = tf.keras.layers.Softmax()
        
    def call(self,input):
        x = input
        x = self.Dense1(x)
        x = self.Dense2(x)
        x = self.Output(x)
        output = self.softmax(x)
        return output



if __name__ == "__main__":
    
    #generate 30 circular MS-VRs
    Dic = np.load('30_MS_VRs.npz')
    centers = Dic['data_centers']
    r = Dic['data_r']
    
    
    #generate MS-VRs image
    fig = plt.figure(figsize=(8,8),dpi=1000) 
    axes = fig.add_subplot(111) 
    plot_circle(centers,r, axes)
    plt.grid(alpha=0.2)
    plt.xlim(0,300)
    plt.ylim(0,300)
    plt.xticks(range(0,305,50))
    plt.yticks(range(0,305,50))
    plt.tick_params(labelsize=18)
    labels = axes.get_xticklabels() + axes.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    font2 = {'family':'Times New Roman', 'weight':'normal', 'size':'20'}
    plt.legend(loc='upper left',prop=font2)
    #plt.savefig('VR-region.svg', bbox_inches='tight')
    plt.show()
    
    
    '''
    #Number of simulations(can be set by yourself)
    Epochs = 1
    voronoi_test_list = []
    test_list = []
    test2_list = []
    test3_list = []
    for T in range(Epochs):
        x_axis = list(range(1000,20000,1000))
        
        #Generate Location-VR dataset
        Known_labeled_pts_list,Known_pts_list,Known_np_labels_list,Test_pts_list,Test_np_labels_list,vr_num = Labeled_points(centers,r,x_axis,
                                                                                                                             30,20000, 1)
        
        
        #generate training point dataset and test point dataset
        train_data_list, test_data_list = generate_dataset(Known_labeled_pts_list,Known_pts_list,Known_np_labels_list,
                                                           Test_pts_list,Test_np_labels_list)
        
        
        #Accuracy of Voronoi Cell Partition
        Accuracy = Test_acc(Test_pts_list,Test_np_labels_list,Known_pts_list,Known_np_labels_list)
        voronoi_test_list.append(Accuracy)
        
        
        
        #Number of neurons in each layer of neural network
        units_num = vr_num 
        units_num2 = units_num + 50
        units_num3 = units_num
        
        
        #Data pre-processing
        load_train_data_list = np.array(train_data_list)
        load_test_data_list = np.array(test_data_list)
        
        train_data_list,train_labels_list = pre_process(load_train_data_list,units_num)
        test_data_list,test_labels_list = pre_process(load_test_data_list,units_num)
        train_data_list2,train_labels_list2 = pre_process(load_train_data_list,units_num2)
        test_data_list2,test_labels_list2 = pre_process(load_test_data_list,units_num2)
           
        
        
        #VR-Net1    
        model = MyNetwork(units_num)    
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)      
        #Compile the model
        model.compile(optimizer = optimizer,
                      loss = 'categorical_crossentropy',
                      metrics = ['accuracy'])
        
        #VR-Net2    
        model2 = MyNetwork2(units_num2)
        optimizer2 = tf.keras.optimizers.Adam(learning_rate=0.01)
        #Compile the model
        model2.compile(optimizer = optimizer2,
                       loss = 'categorical_crossentropy',
                       metrics = ['accuracy'])
        
        #VR-Net3
        model3 = MyNetwork3(units_num)
        optimizer3 = tf.keras.optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
        #Compile the model
        model3.compile(optimizer = optimizer3,
                       loss = 'categorical_crossentropy',
                       metrics = ['accuracy'])
        
        
        
        #Training times of neural network
        epoch_num = 1000
        test_acc_list = []
        test_acc2_list = []
        test_acc3_list = []
        test_loss_list = []
        test_loss2_list = []
        test_loss3_list = []
        
        #train_loss_list = []
        #train_loss2_list = []
        #train_loss3_list = []
        
        for T in range(len(train_data_list)):
            #Train the model
            #History.history:{'loss':xxx, 'accuracy':xxx, 'val_loss':xxx, 'val_accuracy':xxx}
            History = model.fit(train_data_list[T], train_labels_list[T],
                                epochs = epoch_num,
                                batch_size = 200,
                                verbose = 0)         
            #train_loss = History.history['loss']
            #train_loss_list.append(train_loss)
            #train_acc = History.history['accuracy']
            #Evaluate the model
            test_loss, test_acc = model.evaluate(test_data_list[T], test_labels_list[T], verbose=0)
            test_acc_list.append(test_acc)
            test_loss_list.append(test_loss)
            
            
            
            #Train the model
            History2 = model2.fit(train_data_list2[T], train_labels_list2[T],
                                  epochs = epoch_num,
                                  batch_size = 200,
                                  verbose = 0)
            #train_loss2 = History2.history['loss']
            #train_loss2_list.append(train_loss2)
            #train_acc2 = History2.history['accuracy']
            #Evaluate the model
            test_loss2, test_acc2 = model2.evaluate(test_data_list2[T], test_labels_list2[T], verbose=0)
            test_acc2_list.append(test_acc2)
            test_loss2_list.append(test_loss2)
            
            
            
            #Train the model
            History3 = model3.fit(train_data_list[T], train_labels_list[T],
                                  epochs = epoch_num,
                                  batch_size = 200,
                                  verbose = 0)
            #train_loss3 = History3.history['loss']
            #train_loss3_list.append(train_loss3)
            #train_acc3 = History3.history['accuracy']
            #Evaluate the model
            test_loss3, test_acc3 = model3.evaluate(test_data_list[T], test_labels_list[T], verbose=0)
            test_acc3_list.append(test_acc3)
            test_loss3_list.append(test_loss3)
            
        np_test_acc = np.array(test_acc_list)*100
        test_list.append(np_test_acc)
        
        np_test_acc2 = np.array(test_acc2_list)*100
        test2_list.append(np_test_acc2)
        
        np_test_acc3 = np.array(test_acc3_list)*100
        test3_list.append(np_test_acc3)
    
    np_voronoi_test = np.array(voronoi_test_list)
    np_test = np.array(test_list)
    np_test2 = np.array(test2_list)
    np_test3 = np.array(test3_list)
    print(np_test.shape)
    np_voronoi_test_mean = np.mean(np_voronoi_test, axis = 0)
    np_test_mean = np.mean(np_test, axis = 0)
    np_test2_mean = np.mean(np_test2, axis= 0)
    np_test3_mean = np.mean(np_test3, axis= 0)
    
    print('Voronoi_test_accuracy =', np_voronoi_test_mean.tolist())
    print('Test_accuracy =',np_test_mean.tolist())
    print('Test_accuracy2 =',np_test2_mean.tolist())
    print('Test_accuracy3 =',np_test3_mean.tolist())

    
    #Test accuracy curve
    axis_x = list(range(1000,20000,1000))
    
    fig, ax_acc = plt.subplots(figsize=(10,8),dpi=800)
    ax_acc.xaxis.set_major_locator(MultipleLocator(1000))
    #ax_acc.xaxis.set_minor_locator(MultipleLocator(5))
    ax_acc.set_xlim(1000, 20001)
    ax_acc.yaxis.set_major_locator(MultipleLocator(1))
    ax_acc.yaxis.set_minor_locator(MultipleLocator(0.5))
    
    ax_acc.plot(axis_x, np_voronoi_test_mean, linestyle='-',marker='o', ms=10,
                markerfacecolor='red',label = 'Voronoi Cell Partition')
    
    ax_acc.plot(axis_x, np_test_mean, linestyle='-.',marker='^', ms=12,
                markerfacecolor='red',label = 'VR_Net(1-hidden layer)')    
    
    ax_acc.plot(axis_x, np_test2_mean, linestyle='--',marker='*',ms=14,
                mfc='red', label = 'VR-Net(increase width)')
    
    ax_acc.plot(axis_x, np_test3_mean, linestyle=':', marker='d', ms=10,
                markerfacecolor='red',label = 'VR_Net(2-hidden layers)')
    
    ax_acc.grid(alpha=0.2)       
    
    #font1 = {'family':'Songti Sc', 'weight':'normal', 'size':'24'}
    font2 = {'family':'Times New Roman', 'weight':'normal', 'size':'24'}
    
    plt.tick_params(labelsize=20)
    labels = ax_acc.get_xticklabels() + ax_acc.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    
    plt.xticks(rotation=45)
    
    ax_acc.set_xlabel("Number of training samples", font2)
    ax_acc.set_ylabel("Test accuracy(%)", font2)
    ax_acc.legend(loc = 'lower right',  prop=font2)
    #plt.savefig('Fig12.pdf', bbox_inches='tight')
    plt.savefig('VR-Net-acc.pdf', bbox_inches='tight')
    plt.show()
    '''
    
    end = time.perf_counter()
    print('running time: %s Seconds' %(end - start))




























