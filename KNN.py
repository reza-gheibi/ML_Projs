import numpy as np
from sklearn import preprocessing

K = 11      # value of k

# function for computing euclidean distance between two points in 3D
def eucl_dist(p1, p2):
    s = 0
    for i in range(3):
        s = s + (p1[i]-p2[i])*(p1[i]-p2[i])
    return np.sqrt(s)

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
    
    for j in range(test.shape[0]):

        # computing k_NNs
        dist = np.append(train,np.zeros([train.shape[0],1]),1)
        for k in range(train.shape[0]):
            dist[k][4] = eucl_dist(test[j],dist[k])
        dist = dist[np.argsort(dist[:, 4])]
        cls_0 = 0
        cls_1 = 0
        for m in range(K):
            if dist[m][3] == 0:
                cls_0 += 1
            else:
                cls_1 += 1
                
        # assigning class label based on majority 
        if cls_0 > cls_1:
            test[j][4] = 0
        else:
            test[j][4] = 1
            
    # accuracy evaluation
    for n in range(test.shape[0]):
        if test[n][3] == test[n][4]:
            correct +=1
        if test[n][3] == 0:
            if test[n][4] == 0:
                tp += 1
            else:
                fn +=1
        else:
            if test[n][4] == 0:
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
