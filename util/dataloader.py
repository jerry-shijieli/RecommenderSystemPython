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
        return True if userId in self.users_indices else False

    # check if given itemId is within the rating data set
    def check_item(self, itemId):
        return True if itemId in self.items_indices else False



