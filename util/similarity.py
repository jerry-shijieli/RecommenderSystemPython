import numpy as np

class Similarity:
    def __init__(self, sim_method='Cosine'):
        self.missing_value = 0.0
        self.sim_method = sim_method # name of similarity evalution method

    # compute the Pearson correlation coefficients given two array/vectors
    def pearson_coef(self, vec1, vec2):
        v1 = np.asarray(vec1)
        v2 = np.asarray(vec2)
        v1b = (v1 != self.missing_value) # boolean vector to select rated values
        v2b = (v2 != self.missing_value)
        vc = v1b * v2b # commonly rated values in both vectors
        v1r = v1[vc] # selected ratings
        v2r = v2[vc]
        if np.sum(vc)==0:
            return 0
        v1mu = np.mean(v1r) # mean value
        v2mu = np.mean(v2r)
        v1s = v1r - v1mu # remove user bias
        v2s = v2r - v2mu
        v1sqrt = np.sqrt(np.sum(v1s**2)) # variance
        v2sqrt = np.sqrt(np.sum(v2s**2))
        if v1sqrt==0 or v2sqrt==0:
            similarity = 0
        else:
            similarity = np.sum(v1s*v2s) / v1sqrt / v2sqrt
        return similarity

    # compute the cosine value given two array/vectors
    def cosine(self, vec1, vec2):
        v1 = np.asarray(vec1)
        v2 = np.asarray(vec2)
        v1b = (v1 != self.missing_value)  # boolean vector to select rated values
        v2b = (v2 != self.missing_value)
        vc = v1b * v2b  # commonly rated values in both vectors
        v1r = v1[vc]  # selected ratings
        v2r = v2[vc]
        return np.sum(v1r*v2r) / np.sqrt(np.sum(v1r**2)) / np.sqrt(np.sum(v2r**2))

    # return similarity value based on the method option
    def similarity(self, vec1, vec2):
        if self.sim_method == "Pearson":
            return self.pearson_coef(vec1, vec2)
        else:
            return self.cosine(vec1, vec2)