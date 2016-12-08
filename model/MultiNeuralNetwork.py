import sys
import numpy as np
import pandas as pd
sys.path.append("../util/")
import dataloader as dl
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

class MultiNeuralNetwork:
    def __init__(self, threshold_rating_num=10):
        self.complete_rating_matrix = None  # rating matrix with missing value predicted
        self.threshold_rating_num = threshold_rating_num    # threshold number of ratings per user to be used as training
        self.ratings = None # rating instance of Rating
        self.models = None # multilayer perceptrons for each qualified item training set

    # training process
    def fit(self, trainset_feature, trainset_target):
        trainset_df = pd.DataFrame(trainset_feature, columns=['userId', 'itemId', 'timestamp'])
        trainset_df['rating'] = pd.Series(trainset_target)
        self.ratings = dl.Ratings(trainset_df)
        original_rating_matrix = np.matrix(self.ratings.rating_matrix)
        #self.complete_rating_matrix = np.matrix(self.ratings.rating_matrix)
        self.models = [None for _ in range(self.ratings.num_of_items)]

        for itemIndex in xrange(self.ratings.num_of_items):
            mlpr = MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(3,), random_state=1, activation='relu', max_iter=50)
            qualified_training_feature = list()
            qualified_training_target = list()
            for userId in trainset_df['userId']:
                userIndex = self.ratings.users_indices[userId]
                vec = original_rating_matrix[userIndex, :].tolist()[0]
                bias = self.ratings.get_user_rating_bias(userId)
                #print np.asarray(vec)!=0 # debug
                if np.sum(np.asarray(vec) != 0) >= self.threshold_rating_num and vec[itemIndex]!=0:
                    #print "qualified"  # debug
                    vec = map(lambda x: x-bias if x!=0.0 else 0, vec)
                    #print vec # debug
                    qualified_training_target.append(vec.pop(itemIndex))
                    qualified_training_feature.append(vec)
            if len(qualified_training_target)>3:
                #print "qualified"  # debug
                mlpr.fit(qualified_training_feature, qualified_training_target)
                self.models[itemIndex] = mlpr
        # for itemIndex in xrange(self.ratings.num_of_items):
        #     for userIndex in xrange(self.ratings.num_of_users):
        #         if original_rating_matrix[userIndex, itemIndex] == 0:
        #             vec = original_rating_matrix[userIndex].tolist()[0]
        #             vec.pop(itemIndex)
        #             model = models[itemIndex]
        #             if model is not None:
        #                 self.complete_rating_matrix[userIndex, itemIndex] = model.predict(vec) \
        #                                                                 + self.ratings.get_user_rating_bias(self.user_ids[userIndex])
        #             else:
        #                 self.complete_rating_matrix[userIndex, itemIndex] = self.ratings.get_item_rating_bias(self.item_ids[itemIndex])


    # given test feature, predict the rating using the complete rating matrix
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
                prediction = self.ratings.rating_matrix[userIndex, itemIndex]
                if (prediction == 0):
                    bias = self.ratings.get_user_rating_bias(userId)
                    vec = self.ratings.rating_matrix[userIndex].tolist()[0]
                    vec = map(lambda x: x - bias if x != 0.0 else 0, vec)
                    vec.pop(itemIndex)
                    model = self.models[itemIndex]
                    if model is not None:
                        #print vec # debug
                        prediction = model.predict([vec]) + self.ratings.get_user_rating_bias(self.ratings.user_ids[userIndex])
                        prediction = prediction[0]
                    else:
                        prediction = self.ratings.get_item_rating_bias(self.ratings.item_ids[itemIndex])
                results.append(prediction)
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

    # display the setting of the model
    def display_model_setting(self):
        print "Model Setting:"
        print "\tthreshold number of observed ratings: %d" % self.threshold_rating_num

# main function to test module
def main():
    datafilepath = '../data/ml-latest-small/ratings_dp.csv'
    dataset = dl.get_rating_table_from_csv(datafilepath)

    trainset, testset = train_test_split(dataset, test_size=0.1, random_state=0)
    print "training and test size:"
    print trainset.size, testset.size

    trainset_feature = trainset[['userId', 'itemId', 'timestamp']].values
    trainset_target = trainset['rating'].values

    testset_feature = testset[['userId', 'itemId', 'timestamp']].values
    testset_target = testset['rating'].values

    mnn = MultiNeuralNetwork(10)
    mnn.fit(trainset_feature, trainset_target)
    predictions = mnn.predict(testset_feature)
    print "Prediction vs Actual Value"
    print zip(predictions, testset_target)
    rmse = mnn.score(testset_feature, testset_target)
    print "RMSE:"
    print rmse

if __name__ == "__main__":
    main()