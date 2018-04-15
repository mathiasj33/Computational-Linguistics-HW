import random
from nltk import word_tokenize, FreqDist
from collections import defaultdict


def dot(dictA, dictB):
    sum = 0
    for k in dictA:
        sum += dictA[k] * dictB.get(k, 0)
    return sum


def normalized_tokens(text):
    return [token.lower() for token in word_tokenize(text)]


class DataInstance:
    def __init__(self, feature_counts, label):
        """ A data instance consists of a dictionary with feature counts (string -> int) and a label (True or False)."""
        self.feature_counts = feature_counts
        self.label = label

    @classmethod
    def from_list_of_feature_occurrences(cls, feature_list, label):
        """ Creates feature counts for all features in the list."""
        feature_counts = defaultdict(int)
        for f in feature_list:
            feature_counts[f] += 1
        return cls(feature_counts, label)

    @classmethod
    def from_text_file(cls, filename, label):
        with open(filename, 'r') as myfile:
            token_list = normalized_tokens(myfile.read().strip())
        return cls.from_list_of_feature_occurrences(token_list, label)


class Dataset:
    def __init__(self, instance_list):
        """ A data set is defined by a list of instances """
        self.instance_list = instance_list
        self.feature_set = set.union(*[set(inst.feature_counts.keys()) for inst in instance_list])

    def get_topn_features(self, n):
        """ This returns a set with the n most frequently occurring features (i.e. the features that are contained in most instances)."""
        iid = dict()
        for f in self.feature_set:
            count = 0
            for i in self.instance_list:
                if f in i.feature_counts.keys():
                    count += 1
            iid[f] = count
        sorted_list = sorted([(f,c) for f,c in iid.items()], key=lambda x:x[1], reverse=True)
        return set([f for f,c in sorted_list[:n]])

    def set_feature_set(self, feature_set):
        """
        This restrics the feature set. Only features in the specified set all retained. All other feature are removed
        from all instances in the dataset AND from the feature set."""
        self.feature_set = feature_set
        for inst in self.instance_list:
            keys = set(inst.feature_counts.keys()).intersection(feature_set)
            inst.feature_counts = {k: inst.feature_counts[k] for k in keys}

    def most_frequent_sense_accuracy(self):
        """ Computes the accuracy of always predicting the overall most frequent sense for all instances in the dataset. """
        label_list = [inst.label for inst in self.instance_list]
        freq_dist = FreqDist(label_list)
        return freq_dist[freq_dist.max()] / len(label_list)


    def shuffle(self):
        """ Shuffles the dataset. Beneficial for some learning algorithms."""
        random.shuffle(self.instance_list)
