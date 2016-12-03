import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
sys.path.append("../util/")
import dataloader as dl
import UserNearestNeighbor as unnx


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

unn = unnx.UserNearestNeighbor(topk=10, sim_method='Pearson')
unn.fit(trainset_feature, trainset_target)
# kf = KFold(n_splits=5)
# print scores
rmse = unn.score(testset_feature, testset_target)
print rmse
