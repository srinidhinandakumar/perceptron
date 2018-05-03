import time
from sklearn.metrics import f1_score


def evaluate(data1, data2):
    data1 = data1.split("\n")
    data2 = data2.split("\n")

    n = len(data1)
    count = 0
    y_true_1 = []
    y_true_2 = []
    y_out_1 = []
    y_out_2 = []
    for text1, text2 in zip(data1, data2):
        if text1 == "" or text2 == "":
            continue
        else:
            text1 = text1.split(" ")
            text2 = text2.split(" ")
            y_true_1.append(text2[1])
            y_true_2.append(text2[2])
            y_out_1.append(text1[1])
            y_out_2.append(text1[2])
            if text1[1] == text2[1] and text1[2] == text2[2]:
                count += 1

    # print(y_out_1)
    # print(y_true_1)
    # print(y_out_2)
    # print(y_true_2)

    print("Label 1 F1: "+str(f1_score(y_true_1, y_out_1, average="macro")))
    print("Label 2 F1: "+str(f1_score(y_true_2, y_out_2, average="macro")))

    print("Averaged F1: "+str((f1_score(y_true_1, y_out_1, average="macro") + f1_score(y_true_2, y_out_2, average="macro")) / 2))

    print("accuracy : ", str(count * 100 / n))
    print("time: ", str(time.time() - t1))



t1 = time.time()
file1 = "percepoutput.txt"
file4 = "../data/dev-key.txt"
# file4 = "../data/test_key.txt"

fp = open(file1, "r")
fr = open(file4, "r")

data1 = fp.read()
data3 = fr.read()
#print("Vanilla")
evaluate(data1, data3)
#valuate(data2, data3)



