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

        # splitting train data into classes to select k-NNs from each class
        train_0 = []
        train_1 = []
        for k in range(train.shape[0]):
            if train[k][3] == 0:
                train_0.append(train[k])
            else:
                train_1.append(train[k])
        
        # finding the k-NNs in each class
        dist_0 = np.append(train_0,np.zeros([len(train_0),1]),1)
        for k in range(len(train_0)):
            dist_0[k][4] = eucl_dist(test[j],dist_0[k])
        dist_0 = dist_0[np.argsort(dist_0[:, 4])]
        knn_0 = []
        for m in range(K):
            knn_0.append(dist_0[m])

        dist_1 = np.append(train_1,np.zeros([len(train_1),1]),1)
        for k in range(len(train_1)):
            dist_1[k][4] = eucl_dist(test[j],dist_1[k])
        dist_1 = dist_1[np.argsort(dist_1[:, 4])]
        knn_1 = []
        for m in range(K):
            knn_1.append(dist_1[m])

        # computing the mean vectors of k-NNs for each class and distances to the mean vectors
        mean_0 = np.mean(knn_0, axis = 0)
        mean_1 = np.mean(knn_1, axis = 0)
        dist_mean_0 = eucl_dist (test[j], mean_0)
        dist_mean_1 = eucl_dist (test[j], mean_1)

        # assigning the class label based on whichever mean vector is closer        
        if dist_mean_0 < dist_mean_1:
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
