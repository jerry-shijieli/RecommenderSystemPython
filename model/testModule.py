import sys
sys.path.append("../util/")

import dataloader as dl
import numpy as np

dl.test()
ratings = dl.get_rating_data('../data/ml-latest-small/ratings_x.csv')
print np.mean(ratings.get_rating_byItem('31'))
print ratings.get_rating('310', '31')
print ratings.get_item_rating_bias('31')