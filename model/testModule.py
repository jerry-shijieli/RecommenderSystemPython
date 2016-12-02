import sys
sys.path.append("../util/")

import dataloader as dl
import numpy as np

dl.test()
ratings = dl.get_rating_data('../data/ml-latest-small/ratings_x.csv')
print ratings.num_of_users