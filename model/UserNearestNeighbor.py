import sys
import numpy as np
import pandas as pd
sys.path.append("../util/")
import dataloader as dl
import similarity as sm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

class UserNearestNeighbor:
    def __init__(self, topk=0, sim_method='Cosine'):
        self.topk = topk # max number of nearest neighbors used for prediction
        self.sim_method = sim_method # method name of similarity calculation
        self.user_sim_matrix = None # matrix of user-user similarity
        self.ratings = None # rating instance of Ratings

    # training process: build user-user similarity matrix
    def fit(self, trainset_feature, trainset_target):
        trainset_df = pd.DataFrame(trainset_feature, columns=['userId', 'itemId', 'timestamp'])
        trainset_df['rating'] = pd.Series(trainset_target)
        self.ratings = dl.Ratings(trainset_df)
        self.user_sim_matrix = np.matrix([[0.0 for _ in range(self.ratings.num_of_users)]
                                     for _ in range(self.ratings.num_of_users)])
        sim = sm.Similarity(self.sim_method)
        for usr1 in xrange(self.ratings.num_of_users):
            for usr2 in xrange(usr1, self.ratings.num_of_users):
                if usr1 == usr2:
                    continue
                usr1_ratings = self.ratings.rating_matrix[usr1].tolist()[0]
                usr2_ratings = self.ratings.rating_matrix[usr2].tolist()[0]
                self.user_sim_matrix[usr1, usr2] = sim.similarity(usr1_ratings, usr2_ratings)
                self.user_sim_matrix[usr2, usr1] = self.user_sim_matrix[usr1, usr2]
        del trainset_df

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
                user_similarities = self.user_sim_matrix[userIndex, :].tolist()[0]
                user_similarities = zip(range(len(user_similarities)), user_similarities)
                sorted(user_similarities, key=lambda x: x[1], reverse=True)
                neighbors = map(lambda x: x[0], user_similarities)
                if self.topk != 0:
                    neighbors = neighbors[:self.topk]
                overall_rating = 0.0
                total_sim_val = 0.0
                for nb in neighbors:
                    sim_val = user_similarities[nb][1]
                    if sim_val == 0:
                        continue
                    overall_rating += sim_val * self.ratings.rating_matrix[nb, itemIndex]
                    total_sim_val += np.abs(sim_val)
                overall_rating /= (total_sim_val+(1e-8))
                overall_rating += self.ratings.get_user_rating_bias(userId)
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

    # display the setting of the model
    def display_model_setting(self):
        print "Model Setting:"
        print "\tnumber of max neighbors: %d" % self.topk
        print "\tsimilarity metrics: %s" % self.sim_method

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

    unn = UserNearestNeighbor(topk=30, sim_method='Pearson')
    unn.fit(trainset_feature, trainset_target)
    predictions = unn.predict(testset_feature)
    print "Prediction vs Actual Value"
    print zip(predictions, testset_target)
    rmse = unn.score(testset_feature, testset_target)
    print "RMSE:"
    print rmse

if __name__ == "__main__":
    main()