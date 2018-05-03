import sys
import pprint
import json
import numpy as np
import string
import time
import operator
import collections
class Perceptron:

    def __init__(self):
        self.__weight_vector_1__ = dict()
        self.__bias_1__ = 0
        self.__weight_vector_2__ = dict()
        self.__bias_2__ = 0

    def readfile(self, filename):
        fp = open(filename)
        data = fp.read()
        return data

    def process_data(self, data):
        lines = data.split("\n")
        test_data = collections.OrderedDict()
        ids = []
        for l in lines:
            if l == "":
                continue
            else:
                id_and_text = l.split(" ", 1)
                if id_and_text[0] not in test_data:
                    review = id_and_text[1].translate(str.maketrans(" ", " ", string.punctuation))

                    # replace punctuation with space
                    """label2_and_text[1] = re.sub("[^\w\d'\s]+", '', label2_and_text[1])"""
                    review = review.translate(str.maketrans(" ", " ", string.punctuation))
                    """replace_punctuation = str.maketrans(string.punctuation, ' ' * len(string.punctuation))"""
                    """label2_and_text[1] = label2_and_text[1].translate(replace_punctuation)"""

                    # replace numbers with space
                    review = review.translate(str.maketrans(' ', ' ', string.digits))

                    # replace_digits = str.maketrans(string.digits, ' ' * len(string.digits))
                    """label2_and_text[1] = label2_and_text[1].translate(replace_digits)"""
                    test_data[id_and_text[0]] = dict()
                    test_data[id_and_text[0]][str("Review")] = review.lower()
                    test_data[id_and_text[0]][str("Label 1")] = ""
                    test_data[id_and_text[0]][str("Label 2")] = ""
                    ids.append(id_and_text[0])

                else:
                    continue

        return test_data, ids

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

    def test_label1(self, x):
        activation = 0
        # print(x)
        for f in x:
            # print(x[f])
            activation += self.__weight_vector_1__[f]*x[f]
        activation += self.__bias_1__
        if activation >= 0:
            return 1
        else:
            return -1

    def test_label2(self, x):
        activation = 0
        for f in x:
            activation += self.__weight_vector_2__[f]*x[f]
        activation += self.__bias_2__
        if activation >= 0:
            return 1
        else:
            return -1

    def extract_features(self, review, escape):
        x = dict()
        for r in review:
            if r in escape:
                continue
            if r in self.__weight_vector_1__:
                if r in x:
                    x[r] += 1
                else:
                    x[r] = 2
        return x

    def process_tests(self, test_data, escape):
        for id in test_data:
            review = test_data[id]["Review"]
            review = review.split(" ")
            review = self.remove_stopwords(review)
            x = self.extract_features(review, escape)
            label1 = self.test_label1(x)
            label2 = self.test_label2(x)

            if label1 == -1:
                test_data[id]["Label 1"] = "Fake"
            else:
                test_data[id]["Label 1"] = "True"

            if label2 == -1:
                test_data[id]["Label 2"] = "Neg"
            else:
                test_data[id]["Label 2"] = "Pos"

        return test_data

    def write(self, data, ids, outputfile):
        with open(outputfile,"a") as fp:
            for id in ids:
                result = str(id) + " " + data[id]["Label 1"] + " " + data[id]["Label 2"] + "\n"
                fp.write(result)


if __name__ == "__main__":
    t1 = time.time()
    modelfile = sys.argv[1]
    inputfile = sys.argv[2]
    outputfile = "percepoutput.txt"
    # inputfile = "../data/dev-text.txt"
    # modelfile1 = "../data/vanillamodel.txt"
    # modelfile2 = "../data/averagedmodel.txt"
    # vanilla = "../data/vanilla.txt"
    # averaged = "../data/averaged.txt"
    # outputfile = "../data/percepoutput.txt"

    model = Perceptron()
    # read test data
    data = model.readfile(inputfile)
    test_data, ids = model.process_data(data)

    parameters = json.load(open(modelfile, 'r'))

    model.__bias_1__ = parameters["label1"]["bias"]
    model.__weight_vector_1__ = parameters["label1"]["weights"]
    model.__bias_2__ = parameters["label2"]["bias"]
    model.__weight_vector_2__ = parameters["label2"]["weights"]
    escape = parameters["escape"]
    results = model.process_tests(test_data, escape)
    open(outputfile, "w").write("")
    model.write(results, ids, outputfile)

    # # # vanilla
    # parameters = json.load(open(modelfile1, 'r'))
    #
    # model.__bias_1__ = parameters["label1"]["bias"]
    # model.__weight_vector_1__ = parameters["label1"]["weights"]
    # model.__bias_2__ = parameters["label2"]["bias"]
    # model.__weight_vector_2__ = parameters["label2"]["weights"]
    #
    # results = model.process_tests(test_data)
    # open(vanilla, "w").write("")
    # model.write(results, vanilla)
    #
    # #averaged
    # parameters = json.load(open(modelfile2, 'r'))
    #
    # model.__bias_1__ = parameters["label1"]["bias"]
    # model.__weight_vector_1__ = parameters["label1"]["weights"]
    # model.__bias_2__ = parameters["label2"]["bias"]
    # model.__weight_vector_2__ = parameters["label2"]["weights"]
    #
    # results = model.process_tests(test_data)
    # open(averaged, "w").write("")
    # model.write(results, averaged)
    print("time: ", (time.time() - t1))
