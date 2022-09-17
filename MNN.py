import numpy as np
from sklearn import preprocessing

# defining two required methods
def logistic(p):
    return 1.0/(1+np.exp(-p))

def log_derivative(q):
    return logistic(q)*(1-logistic(q))

ip_nodes = 3  # number of input layer nodes
hid_nodes = 4 # number of hidden layer nodes
lr = 0.1      # learning rate

# reading and normalizing the data
data = np.genfromtxt('haberman.data', delimiter=',')
np.random.shuffle(data)
min_max_scaler = preprocessing.MinMaxScaler()
data = min_max_scaler.fit_transform(data)

num_obs = data.shape[0]

# sum of errors and f-scores for 10 folds
sum_err = sum_fs = 0

# 10 fold cross validation
for i in range(1,11):
    
    # counts of correct predictions, TP, FP, FN, TN 
    correct = 0
    tp = fp = fn = tn = 0

    # train test splitting
    train = np.delete(data, slice(num_obs-int(num_obs/10)*i,num_obs-int(num_obs/10)*(i-1),1), axis=0)
    test = data[num_obs-int(num_obs/10)*i:num_obs-int(num_obs/10)*(i-1),:]

    # adding a column for the predicted class label
    test = np.append(test,np.zeros([test.shape[0],1]),1)

    # weights initialization etc    
    weights_ItoH = np.random.uniform(-1, 1, (ip_nodes, hid_nodes))
    weights_HtoO = np.random.uniform(-1, 1, hid_nodes)
    preActivation_H = np.zeros(hid_nodes)
    postActivation_H = np.zeros(hid_nodes)
    training_data = np.delete(train, 3, axis=1)
    target_output = train[:,3]
    test_data = np.delete(test, 4, axis=1)
    test_data = np.delete(test_data, 3, axis=1)

    # training the neural network
    for e in range(10):
        for s in range(train.shape[0]):
            for n in range(hid_nodes):
                preActivation_H[n] = np.dot(training_data[s,:], weights_ItoH[:,n])
                postActivation_H[n] = logistic(preActivation_H[n])
            preActivation_O = np.dot(postActivation_H, weights_HtoO)
            postActivation_O = logistic(preActivation_O)
            FE = postActivation_O - target_output[s]
            for H_node in range(hid_nodes):
                S_error = FE * log_derivative(preActivation_O)
                gradient_HtoO = S_error * postActivation_H[H_node]      
                for I_node in range(ip_nodes):
                    input_value = training_data[s,I_node]
                    gradient_ItoH = S_error*weights_HtoO[H_node]*log_derivative(preActivation_H[H_node])*input_value               
                    weights_ItoH[I_node, H_node] -= lr * gradient_ItoH                
                weights_HtoO[H_node] -= lr * gradient_HtoO

    # predicting the class labels for the test data
    for s in range(test.shape[0]):
        for n in range(hid_nodes):
            preActivation_H[n] = np.dot(test_data[s,:],weights_ItoH[:,n])
            postActivation_H[n] = logistic(preActivation_H[n])            
        preActivation_O = np.dot(postActivation_H, weights_HtoO)
        postActivation_O = logistic(preActivation_O)
        if postActivation_O > 0.5:
            test[s][4] = 1
        else:
            test[s][4] = 0     
        
    # accuracy evaluation
    for n in range(test.shape[0]):
        if test[n][3] == test[n][4]:
            correct +=1
        if test[n][3] == 0:
            if test[n][4] == 0:
                tp += 1
            else:
                fn += 1
        else:
            if test[n][4] == 0:
                fp += 1
            else:
                tn += 1

    sum_err += 100-correct/test.shape[0]*100
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f_score = 2*precision*recall/(precision+recall) 
    sum_fs += f_score

# printing the average results over 10 folds
print("F-Score: " + str(sum_fs/10))
print("Percentage error: " + str(sum_err/10))
