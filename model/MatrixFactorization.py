import sys
import numpy as np
import pandas as pd
sys.path.append("../util/")
import dataloader as dl
from sklearn.metrics import mean_squared_error

class MatrixFactorization:
    def __init__(self, dim_of_factor=10, learning_rate=1e-5, regularization=0.015, max_iter=100, tolerance=0.1):
        self.dim_of_factor = dim_of_factor # dimension of factor matrix
        self.learning_rate = learning_rate # learning rate in training process
        self.max_iter = max_iter # max number of training iterations
        self.regularization = regularization # L2 regularization to avoid overfitting for both user and item factor matrices
        self.tolerance = tolerance  # tolerance value to determine convergence
        self.ratings = None # rating instance of Ratings
        self.user_factor_matrix = None # matrix of user factors
        self.item_factor_matrix = None # matrix of item factors
        self.complete_rating_matrix = None # final rating matrix with missing rating filled by the factor matrix production

    # training process: fit the user factor matrix and item factor matrix using least square optimization
    def fit(self, trainset_feature, trainset_target):
        trainset_df = pd.DataFrame(trainset_feature, columns=['userId', 'itemId', 'timestamp'])
        trainset_df['rating'] = pd.Series(trainset_target)
        self.ratings = dl.Ratings(trainset_df) # load rating data into user-item rating matrix
        self.user_factor_matrix =

    # given test feature, predict the rating
    def predict(self, testset_feature):
        results = list()
        for x in testset_feature:
            userId = x[0]
            itemId = x[1]
            find_user = self.ratings.check_user(userId)
            find_item = self.ratings.check_item(itemId)
            if find_user and find_item:
                userIndex = self.ratings.users_indices[userId]
                itemIndex = self.ratings.items_indices[itemId]
                overall_rating = self.complete_rating_matrix[userIndex, itemIndex]
                results.append(overall_rating)
            elif find_user:
                results.append(self.ratings.get_user_rating_bias(userId))
            elif find_item:
                results.append(self.ratings.get_item_rating_bias(itemId))
            else:
                results.append(self.ratings.get_rating_average())
        return results

    # evaluate the model
    def score(self, testset_feature, testset_target):
        predictions = self.predict(testset_feature)
        return np.sqrt(mean_squared_error(testset_target, predictions))