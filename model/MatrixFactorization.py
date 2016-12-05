import sys
import numpy as np
import pandas as pd
sys.path.append("../util/")
import dataloader as dl
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

class MatrixFactorization:
    def __init__(self, dim_of_factor=10, learning_rate=1e-5, regularization=0.015, max_iter=100, tolerance=1e-5):
        self.dim_of_factor = dim_of_factor # dimension of factor matrix
        self.learning_rate = learning_rate # learning rate in training process
        self.max_iter = max_iter # max number of training iterations
        self.regularization = regularization # L2 regularization to avoid overfitting for both user and item factor matrices
        self.tolerance = tolerance  # tolerance value to determine convergence
        self.ratings = None # rating instance of Ratings
        self.user_factor_matrix = None # matrix of user factors
        self.item_factor_matrix = None # matrix of item factors
        self.complete_rating_matrix = None # final rating matrix with missing rating filled by the factor matrix production
        self.loss_val = np.inf # value of loss function for each iteration

    # training process: fit the user factor matrix and item factor matrix using least square optimization
    def fit(self, trainset_feature, trainset_target):
        self.fit_by_raw_matrix_factorization(trainset_feature, trainset_target)

    # apply matrix factorization to the raw rating matrix
    def fit_by_raw_matrix_factorization(self, trainset_feature, trainset_target):
        trainset_df = pd.DataFrame(trainset_feature, columns=['userId', 'itemId', 'timestamp'])
        trainset_df['rating'] = pd.Series(trainset_target)
        self.ratings = dl.Ratings(trainset_df) # load rating data into user-item rating matrix
        self.user_factor_matrix = np.matrix(np.random.rand(self.ratings.num_of_users, self.dim_of_factor))
        self.item_factor_matrix = np.matrix(np.random.rand(self.ratings.num_of_items, self.dim_of_factor))
        max_rating = self.ratings.max_rating
        min_rating = self.ratings.min_rating
        observe_matrix = (self.ratings.rating_matrix == 0) # initialize and select unrated values to set zero
        count_iter = 0 # iteration counter
        self.complete_rating_matrix = self.user_factor_matrix * self.item_factor_matrix.transpose()
        #self.complete_rating_matrix[self.complete_rating_matrix > max_rating] = max_rating # rescale predicted ratings by top and bottom limits
        #self.complete_rating_matrix[self.complete_rating_matrix < min_rating] = min_rating
        err_matrix = self.ratings.rating_matrix - self.complete_rating_matrix
        err_matrix[observe_matrix] = 0
        self.loss_val = np.sum(np.power(err_matrix, 2)) \
                        + self.regularization / 2 * np.sum(np.power(self.user_factor_matrix, 2)) \
                        + self.regularization / 2 * np.sum(np.power(self.item_factor_matrix, 2))
        while (count_iter<=self.max_iter and self.loss_val>self.tolerance):
            count_iter += 1
            user_factor_matrix_update = self.learning_rate * (err_matrix * self.item_factor_matrix
                                                              - self.regularization * self.user_factor_matrix)
            item_factor_matrix_update = self.learning_rate * (err_matrix.transpose() * self.user_factor_matrix
                                                              - self.regularization * self.item_factor_matrix)
            self.user_factor_matrix += user_factor_matrix_update
            self.item_factor_matrix += item_factor_matrix_update
            self.complete_rating_matrix = self.user_factor_matrix * self.item_factor_matrix.transpose()
            #self.complete_rating_matrix[self.complete_rating_matrix > max_rating] = max_rating # rescale predicted ratings
            #self.complete_rating_matrix[self.complete_rating_matrix < min_rating] = min_rating
            err_matrix = self.ratings.rating_matrix - self.complete_rating_matrix
            err_matrix[observe_matrix] = 0
            self.loss_val = np.sum(np.power(err_matrix, 2)) \
                            + self.regularization / 2 * np.sum(np.power(self.user_factor_matrix, 2)) \
                            + self.regularization / 2 * np.sum(np.power(self.item_factor_matrix, 2))

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

# main function to test module
def main():
    datafilepath = '../data/ml-latest-small/ratings_x.csv'
    dataset = dl.get_rating_table_from_csv(datafilepath)

    trainset, testset = train_test_split(dataset, test_size=0.1, random_state=0)
    print "training and test size:"
    print trainset.size, testset.size

    trainset_feature = trainset[['userId', 'itemId', 'timestamp']].values
    trainset_target = trainset['rating'].values

    testset_feature = testset[['userId', 'itemId', 'timestamp']].values
    testset_target = testset['rating'].values

    mf = MatrixFactorization(dim_of_factor=20, max_iter=100)
    mf.fit(trainset_feature, trainset_target)
    predictions = mf.predict(testset_feature)
    print "Prediction vs Actual Value"
    print zip(predictions, testset_target)
    rmse = mf.score(testset_feature, testset_target)
    print "RMSE:"
    print rmse

if __name__ == "__main__":
    main()