import sys
import pprint
import json
import numpy as np
import string
import time
import operator
import collections
import random

class Perceptron:

    def __init__(self):
        self.__weight_vector__ = dict()
        self.__cached_weights__ = dict()
        self.__features__ = dict()
        self.__bias__ = 0
        self.__cached_bias__ = 0

    def readfile(self, filename):
        fp = open(filename)
        data = fp.read()
        return data

    def process_data(self, data):
        lines = data.split("\n")
        train_data = collections.OrderedDict()
        for l in lines:
            if l == "":
                continue
            else:
                id_and_text = l.split(" ", 1)
                label1_and_text = id_and_text[1].split(" ", 1)
                label2_and_text = label1_and_text[1].split(" ", 1)
                if id_and_text[0] not in train_data:
                    train_data[id_and_text[0]] = dict()

                    # replace punctuation with space
                    """label2_and_text[1] = re.sub("[^\w\d'\s]+", '', label2_and_text[1])"""
                    label2_and_text[1] = label2_and_text[1].translate(str.maketrans(" ", " ", string.punctuation))
                    """replace_punctuation = str.maketrans(string.punctuation, ' ' * len(string.punctuation))"""
                    """label2_and_text[1] = label2_and_text[1].translate(replace_punctuation)"""

                    # replace numbers with space
                    label2_and_text[1] = label2_and_text[1].translate(str.maketrans(' ', ' ', string.digits))

                    """replace_digits = str.maketrans(string.digits, ' ' * len(string.digits))"""
                    """label2_and_text[1] = label2_and_text[1].translate(replace_digits)"""

                    train_data[id_and_text[0]][str("Review")] = label2_and_text[1].lower()
                    train_data[id_and_text[0]][str("Label 1")] = label1_and_text[0]
                    train_data[id_and_text[0]][str("Label 2")] = label2_and_text[0]
                else:
                    continue
        # pprint.pprint(train_data)
        # a = input()
        # train_data = collections.OrderedDict(sorted(train_data.items()))
        #pprint.pprint(train_data)
        #exit()
        return train_data

    def remove_stopwords(self, sentence):
        stopwords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing",
                     "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is",
                     "it", "its", "itself", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "should", "so", "some", "such", "than",
                     "that", "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "were", "what", "when", "where", "which", "while",
                     "who", "whom", "why", "with", "would", "you", "your", "yours", "yourself", "yourselves"]
        new_sentence = []
        for s in sentence:
            if s not in stopwords:
                new_sentence.append(s)
        return new_sentence

    def assignY(self, train_data):
        y1 = collections.OrderedDict()
        y2 = collections.OrderedDict()

        for td in train_data:
            # print(train_data[td]["Label 1"]+" "+train_data[td]["Label 2"])
            if train_data[td]["Label 1"] == "Fake":
                y1[td] = -1
                # print(-1)
            elif train_data[td]["Label 1"] == "True":
                y1[td] = 1
                # print(1)

            if train_data[td]["Label 2"] == "Neg":
                y2[td] = -1
                # print(-1)
            elif train_data[td]["Label 2"] == "Pos":
                y2[td] = 1
                # print(1)

        return y1, y2

    def sentences(self, train_data):
        # x = []
        # for id in train_data:
        #     sentence = train_data[id]["Review"]
        #     x.append(sentence)
        #
        # return x
        x = collections.OrderedDict()
        for id in train_data:
            sentence = train_data[id]["Review"]
            x[id] = sentence
        return x

    def feature_vector(self, sentence, words):
        sentence = sentence.split(" ")
        sentence = self.remove_stopwords(sentence)
        fv = dict()
        for s in sentence:
            if s in words:
                continue
            if s in self.__weight_vector__:
                if s not in fv:
                    fv[s] = 1
                else:
                    fv[s] += 1
        return fv

    def train_vanilla(self, D, maxIter, x, y, words):
        # pprint.pprint(self.__weight_vector__)
        for i in range(maxIter):
            count = 0
            # x = dict(collections.OrderedDict(sorted(x.items())))

            for id in x:
                # print(sentence)
                sentence = x[id]
                fv = self.feature_vector(sentence, words)
                # pprint.pprint(fv)
                activation = 0
                # print(self.__weight_vector__[f])
                for f in fv:
                    activation += fv[f]*self.__weight_vector__[f]
                activation += self.__bias__

                if activation*y[id] <= 0:
                    # print("hi")
                    self.__bias__ += y[id]
                    for f in fv:
                        self.__weight_vector__[f] += y[id]*fv[f]
                count += 1
        # pprint.pprint(self.__weight_vector__)
        # exit()

    def train_averaged(self, D, maxIter, x, y, words):
        # pprint.pprint(self.__weight_vector__)
        c = 0
        for i in range(maxIter):
            count = 0

            for id in x:
                # print(sentence)
                sentence = x[id]
                fv = self.feature_vector(sentence, words)
                # pprint.pprint(fv)
                activation = 0
                # print(self.__weight_vector__[f])
                for f in fv:
                    activation += fv[f]*self.__weight_vector__[f]
                activation += self.__bias__

                if activation*y[id] <= 0:
                    # print("hi")
                    self.__bias__ += y[id]
                    self.__cached_bias__ += y[id]*c
                    for f in fv:
                        self.__weight_vector__[f] += y[id]*fv[f]
                        self.__cached_weights__[f] += y[id]*fv[f]*c
                count += 1
                c += 1

        for f in self.__weight_vector__:
            self.__weight_vector__[f] -= (self.__cached_weights__[f]/c)
        self.__bias__ -= self.__cached_bias__/c

    def process_results(self, w1, b1, w2, b2, words):

        results = dict()
        results["label1"] = dict()
        results["label1"]["weights"] = w1
        results["label1"]["bias"] = b1
        results["label2"] = dict()
        results["label2"]["weights"] = w2
        results["label2"]["bias"] = b2
        results["escape"] = words
        # pprint.pprint(results)
        return results

    def set_features(self, train, n):
        words = dict()
        for id in train:
            sentence = train[id]["Review"].split(" ")
            for s in sentence:
                if s in self.__features__:
                    self.__features__[s] += 1
                else:
                    self.__features__[s] = 1
        # self.__features__ = sorted(self.__features__.items(), key=operator.itemgetter(1), reverse=True)
        features = dict(sorted(self.__features__.items(), key=operator.itemgetter(1), reverse=True))
        # for f in features:
        #     if features[f] == 1:
        #         words.append(f)
        # words = {k : v for k, v in features.items() if v == 1}
        print(len(words))
        top = sorted(self.__features__.items(), key=operator.itemgetter(1), reverse=True)[:11]
        for k in top:
            words[k[0]] = k[1]
        # print(words)

        # for f in features:
        #     orderedFeatures[f[0]] = f[1]
        self.__features__ = features
        # print(self.__features__)
        return words


    def extract_weight_vector(self, train):

        weights = dict()
        for t in train:
            sentence = train[t]["Review"]

            sentence = sentence.split(" ")
            # print(sentence)
            sentence = self.remove_stopwords(sentence)

            for s in sentence:
                if s == '':
                    continue
                if s not in weights:
                    weights[s] = 0

        return weights

    def write_data(self, data, filename):
        with open(filename, 'w') as outfile:
            json.dump(data, outfile, indent=4)


if __name__ == "__main__":
    t1 = time.time()
    inputfile = sys.argv[1]
    vanilla = "vanillamodel.txt"
    averaged = "averagedmodel.txt"
    # inputfile = "../data/train-labeled.txt"
    # vanilla = "../data/vanillamodel.txt"
    # averaged = "../data/averagedmodel.txt"

    model = Perceptron()
    data = model.readfile(inputfile)
    training_data = model.process_data(data)
    escape_words = model.set_features(training_data, 10)

    y1, y2 = model.assignY(training_data)

    # exit()
    x = model.sentences(training_data)  # a list of reviews
    maxIter = 30

    # vanilla model
    model.__bias__ = 0
    model.__weight_vector__ = model.extract_weight_vector(training_data)
    D = len(model.__weight_vector__)
    # pprint.pprint(model.__weight_vector__)
    # print(model.__bias__)
    model.train_vanilla(D, maxIter, x, y1, escape_words)

    # label 1 results
    weights1 = model.__weight_vector__
    bias1 = model.__bias__

    model = Perceptron()
    model.__bias__ = 0
    model.__weight_vector__ = model.extract_weight_vector(training_data)
    # pprint.pprint(model.__weight_vector__)
    # print(model.__bias__)

    model.train_vanilla(D, maxIter, x, y2, escape_words)

    # label 2 results
    weights2 = model.__weight_vector__
    bias2 = model.__bias__
    # pprint.pprint(weights2)
    # print(bias2)

    model_data1 = model.process_results(weights1, bias1, weights2, bias2, escape_words)
    model.write_data(model_data1, vanilla)

    # averaged model
    model = Perceptron()
    model.__weight_vector__ = model.extract_weight_vector(training_data)
    model.__cached_weights__ = model.extract_weight_vector(training_data)
    model.__bias__ = 0
    model.____cached_bias__ = 0
    # print(list(x))

    model.train_averaged(D, maxIter, x, y1, escape_words)
    # label 1
    weights1 = model.__weight_vector__
    bias1 = model.__bias__
    model = Perceptron()
    model.__weight_vector__ = model.extract_weight_vector(training_data)
    model.__cached_weights__ = model.extract_weight_vector(training_data)
    model.__bias__ = 0
    model.____cached_bias__ = 0

    model.train_averaged(D, maxIter, x, y2, escape_words)
    # label 2
    weights2 = model.__weight_vector__
    bias2 = model.__bias__

    model_data2 = model.process_results(weights1, bias1, weights2, bias2, escape_words)
    model.write_data(model_data2, averaged)

    print("time: ", (time.time() - t1))
