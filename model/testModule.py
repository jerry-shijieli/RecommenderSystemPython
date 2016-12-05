import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
sys.path.append("../util/")
import dataloader as dl
import UserNearestNeighbor as unn_model
import MultiNeuralNetwork as mnn_model
import MatrixFactorization as mf_model


dl.test()
datafilepath = '../data/ml-latest-small/ratings_x.csv'
# ratings = dl.get_rating_matrix_from_csv('../data/ml-latest-small/ratings_x.csv')
# print np.mean(ratings.get_rating_byItem('31'))
# print ratings.get_rating('310', '31')
# print ratings.get_item_rating_bias('31')

dataset = dl.get_rating_table_from_csv(datafilepath)
#dataset.drop('timestamp', 1)
test_split_ratio = 0.1
trainset, testset = train_test_split(dataset, test_size=0.1, random_state=0)
print trainset.size

trainset_feature = trainset[['userId','itemId', 'timestamp']].values
trainset_target = trainset['rating'].values

testset_feature = testset[['userId','itemId', 'timestamp']].values
testset_target = testset['rating'].values

# unn = unn_model.UserNearestNeighbor(topk=30, sim_method='Pearson')
# unn.fit(trainset_feature, trainset_target)
# # kf = KFold(n_splits=5)
# # scores = list()
# # for train_index, valid_index in kf.split(trainset_feature):
# #     print("New validation start...")
# #     X_train, X_valid = trainset_feature[train_index], trainset_feature[train_index]
# #     y_train, y_valid = trainset_target[train_index], trainset_target[train_index]
# #     unn.fit(X_train, y_train)
# #     scores.append(unn.score(X_valid, y_valid))
# # print scores
# # print np.mean(scores), np.std(scores)
# predictions = unn.predict(testset_feature)
# print zip(predictions, testset_target)
# rmse = unn.score(testset_feature, testset_target)
# print rmse

# mnn = mnn_model.MultiNeuralNetwork(threshold_rating_num=2000) # set the least number of rated values
# mnn.fit(trainset_feature, trainset_target)
# predictions = mnn.predict(testset_feature)
# print zip(predictions, testset_target)
# rmse = mnn.score(testset_feature, testset_target)
# print rmse

mf = mf_model.MatrixFactorization(dim_of_factor=30, max_iter=100)
mf.fit(trainset_feature, trainset_target)
predictions = mf.predict(testset_feature)
print zip(predictions, testset_target)
rmse = mf.score(testset_feature, testset_target)
print rmse