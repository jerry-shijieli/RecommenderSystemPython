import pandas as pd
import numpy as np
import os
import sys

def test():
    print "It works!"
    pass

# load the rating data
# return rating data embedded in Ratings instance
def get_rating_data(csv_file_path):
    if csv_file_path.endswith('.csv') and os.path.exists(csv_file_path):
        data = pd.read_csv(csv_file_path, header=0,
            dtype={'userId':str, 'itemId':str, 'rating':np.float64, 'timestamp':np.int64})
        return Ratings(data)
    else:
        sys.stderr("Data file either not exists or cannot be loaded!")
        return None

# Ratings class contains the DataFrame of user-item rating table with timestamp, including
# useful methods to select rating data by given conditions.
class Ratings:
    # initialize the rating matrix and relevant properties
    def __init__(self, rating_data, missing_rating_default=0.0):
        self.ratings = rating_data # data frame containing (userId, itemId, rating, timestamp)
        self.users_indices = None  # dictionary of {userId: userIndex}
        self.items_indices = None  # dictionary of {itemId: itemIndex}
        self.num_of_users = 0   # number of total users
        self.num_of_items = 0   # number of total items
        self.rating_matrix = None   # matrix of user-item rating by user and item indices
        self.timestamp_matrix =  None   # matrix of user-item rating timestamp
        self.missing_rating_default_value = missing_rating_default # default value for missing rating
        self._convert_to_matrix()

    # helper function for initialization. Convert the rating data frame to rating and timestamp matrices
    def _convert_to_matrix(self):
        all_user_id  = self.ratings['userId'].values
        users = list(set(all_user_id))
        users.sort()
        all_item_id = self.ratings['itemId'].values
        items = list(set(all_item_id))
        items.sort()
        self.users_indices = dict()
        self.items_indices = dict()
        for index, userId in enumerate(users):
            self.users_indices[userId] = index
        for index, itemId in enumerate(items):
            self.items_indices[itemId] = index
        self.num_of_users = len(self.users_indices)
        self.num_of_items = len(self.items_indices)
        rating_values = [[self.missing_rating_default_value for _ in range(self.num_of_items)]
                         for _ in range(self.num_of_users)]
        self.rating_matrix = np.matrix(rating_values)
        timestamp_values = [[0 for _ in range(self.num_of_items)]
                         for _ in range(self.num_of_users)]
        self.timestamp_matrix = np.matrix(timestamp_values)
        for _, row in self.ratings.iterrows():
            userId = row['userId']
            itemId = row['itemId']
            rating = row['rating']
            timestamp = row['timestamp']
            userIndex = self.users_indices[userId]
            itemIndex = self.items_indices[itemId]
            self.rating_matrix[userIndex, itemIndex] = rating
            self.timestamp_matrix[userIndex, itemIndex] = timestamp
        self.ratings = None

    # check if given userId is within the rating data set
    def check_user(self, userId):
        return True if userId in self.users_indices.keys() else False

    # check if given itemId is within the rating data set
    def check_item(self, itemId):
        return True if itemId in self.items_indices.keys() else False

    # calculate the number of rated entries in the rating matrix
    def get_num_of_ratings(self):
        check_matrix = (self.rating_matrix != self.missing_rating_default_value)
        return np.sum(check_matrix)

    # compute the sparsity of the rating matrix
    def get_sparsity(self):
        return 1 - self.get_num_of_ratings() / np.float64(self.num_of_users * self.num_of_items)

    # compute the global average of the rating matrix (For simplicity, only consider the 0.0 default missing value)
    def get_rating_average(self):
        return np.sum(self.rating_matrix) / self.get_num_of_ratings()

    # retrieve rating value given a (userId, itemId) pair
    def get_rating(self, userId, itemId):
        if self.check_user(userId) and self.check_item(itemId):
            return self.rating_matrix[self.users_indices[userId], self.items_indices[itemId]]
        else:
            sys.stderr("userId or itemId is invalid!")
            return None

    # retrieve all rating values as a list given userId
    def get_rating_byUser(self, userId):
        if self.check_user(userId):
            return self.rating_matrix[self.users_indices[userId], :].tolist()[0]
        else:
            sys.stderr("userId is invalid!")
            return None

    # retrieve all rating values as a list given itemId
    def get_rating_byItem(self, itemId):
        if self.check_item(itemId):
            return self.rating_matrix[:, self.items_indices[itemId]].transpose().tolist()[0]
        else:
            sys.stderr("itemId is invalid!")
            return None

    # calculate the number of rated entried given a userId
    def get_num_of_ratings_byUser(self, userId):
        user_ratings = np.asarray(self.get_rating_byUser(userId))
        if user_ratings is not None:
            return np.sum(user_ratings != 0.0)
        else:
            sys.stderr("userId is invalid!")
            return None

    # calculate the number of rated entried given a itemId
    def get_num_of_ratings_byItem(self, itemId):
        item_ratings = np.asarray(self.get_rating_byItem(itemId))
        if item_ratings is not None:
            return np.sum(item_ratings != 0.0)
        else:
            sys.stderr("itemId is invalid!")
            return None

    # compute the rating bias (mean value) given a userId
    def get_user_rating_bias(self, userId):
        user_ratings = np.asarray(self.get_rating_byUser(userId))
        if user_ratings is not None:
            num_of_ratings = np.sum(user_ratings != 0.0)
            rating_total = np.sum(user_ratings[user_ratings != 0.0])
            rating_bias = 0 if num_of_ratings==0 else rating_total/num_of_ratings
            return rating_bias
        else:
            sys.stderr("userId is invalid!")
            return None

    # compute the rating bias (mean value) given a itemId
    def get_item_rating_bias(self, itemId):
        item_ratings = np.asarray(self.get_rating_byItem(itemId))
        if item_ratings is not None:
            num_of_ratings = np.sum(item_ratings != 0.0)
            rating_total = np.sum(item_ratings[item_ratings != 0.0])
            rating_bias = 0 if num_of_ratings==0 else rating_total/num_of_ratings
            return rating_bias
        else:
            sys.stderr("itemId is invalid!")
            return None

    # set or update the rating value for a given (userId, itemId) pair
    def set_rating(self, userId, itemId, rating):
        if self.check_user(userId) and self.check_item(itemId):
            self.rating_matrix[self.users_indices[userId], self.items_indices[itemId]] = rating
            return True
        else:
            sys.stderr("userId or itemId is invalid!")
            return False