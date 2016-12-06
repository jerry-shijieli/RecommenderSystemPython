import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
sys.path.append("../util/")
import dataloader as dl
import UserNearestNeighbor as unn_model
import MatrixFactorization as mf_model


datafilepath = '../data/ml-latest-small/ratings_x.csv'
dataset = dl.get_rating_table_from_csv(datafilepath)

test_split_ratio = 0.1
trainset, testset = train_test_split(dataset, test_size=0.1, random_state=0)

trainset_feature = trainset[['userId','itemId', 'timestamp']].values
trainset_target = trainset['rating'].values

testset_feature = testset[['userId','itemId', 'timestamp']].values
testset_target = testset['rating'].values

all_models = {'UserNearestNeighbor': unn_model.UserNearestNeighbor(topk=80, sim_method='Pearson'),
              'MatrixFactorization': mf_model.MatrixFactorization(dim_of_factor=40, max_iter=100)
              }

for model_name in all_models.keys():
    print("Test new model "+model_name+":")
    model = all_models[model_name]
    kf = KFold(n_splits=5)
    scores = list()
    model.display_model_setting()
    for train_index, valid_index in kf.split(trainset_feature):
        print("New validation start...")
        X_train, X_valid = trainset_feature[train_index], trainset_feature[train_index]
        y_train, y_valid = trainset_target[train_index], trainset_target[train_index]
        model.fit(X_train, y_train)
        scores.append(model.score(X_valid, y_valid))
    print scores
    print np.mean(scores), np.std(scores)
    rmse = model.score(testset_feature, testset_target)
    print rmse
    print('\n\n')