import numpy as np
from sklearn import preprocessing

lr = 0.05 # learning rate

# reading and normalizing the data
data = np.genfromtxt('haberman.data', delimiter=',')
np.random.shuffle(data)
min_max_scaler = preprocessing.MinMaxScaler()
data = min_max_scaler.fit_transform(data)

num_obs = data.shape[0]

# changing the 0 class labels to -1
for i in range(num_obs):
    if data[i][3] == 0:
        data[i][3] = -1

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
    
    # weights initialization
    weights = [np.random.uniform(0,1),np.random.uniform(0,1),np.random.uniform(0,1)]

    # training episodes
    for t in range(1000):

        # computing the gradient and updating the weights
        gradient = [0,0,0]
        for m in range (train.shape[0]):
            temp = train[m][3]/(1+np.exp(train[m][3]*(weights[0]*train[m][0]+weights[1]*train[m][1]+weights[2]*train[m][2])))
            temp_vec = [temp*train[m][0],temp*train[m][1],temp*train[m][2]]
            for l in range(3):
                gradient[l] += temp_vec[l]
                
        for l in range(3):
            gradient[l] /= (-train.shape[0])
            weights[l] -= lr*gradient[l]
        
    # predicting the class labels for the test data
    for j in range(test.shape[0]):
        s = weights[0]*test[j][0]+weights[1]*test[j][1]+weights[2]*test[j][2]
        value = 1/(1+np.exp(-s))
        if value < 0.5:
            test[j][4]=-1
        else:
            test[j][4]=1

    # accuracy evaluation        
    for n in range(test.shape[0]):
        if test[n][3] == test[n][4]:
            correct +=1
        if test[n][3] == -1:
            if test[n][4] == -1:
                tp += 1
            else:
                fn +=1
        else:
            if test[n][4] == -1:
                fp += 1
            else:
                tn +=1

    sum_err += 100-correct/test.shape[0]*100
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f_score = 2*precision*recall/(precision+recall) 
    sum_fs += f_score

# printing the average results over 10 folds
print("F-Score: " + str(sum_fs/10))
print("Percentage error: " + str(sum_err/10))
