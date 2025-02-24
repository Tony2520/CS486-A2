from collections import Counter
import math
from queue import PriorityQueue
import matplotlib.pyplot as plt


def loadWords():
    words = []
    with open("words.txt", "r") as f:
        for line in f:
            words.append(line.strip())
    return words


WORDS = loadWords()


def loadData(data_file, label_file):

    res = []
    for i in range(1500):
        res.append([set(), 0])

    i = 0

    # label
    with open(label_file, "r") as f:
        for line in f:
            res[i][1] = int(line)
            i += 1

    # word features
    with open(data_file, "r") as f:
        for line in f:
            docId, wordId = line.split()
            res[int(docId) - 1][0].add(int(wordId))

    return res


def entropy(dataSet):
    if not dataSet:
        return 1

    probA = Counter(d[1] for d in dataSet)[1] / len(dataSet)
    probB = 1 - probA

    eA = -1 * probA * (0 if probA == 0 else math.log2(probA))
    eB = -1 * probB * (0 if probB == 0 else math.log2(probB))

    return eA + eB


def gain(dataSet, feature, v=1):
    if not dataSet:
        return 0
    t = entropy([d for d in dataSet if feature in d[0]])
    f = entropy([d for d in dataSet if feature not in d[0]])

    g = 0

    if v == 1:
        n1, n2, total = 0, 0, 0
        for d in dataSet:
            total += 1
            if feature in d[0]:
                n1 += 1
            else:
                n2 += 1

        g = n1 / total * t + n2 / total * f

    else:
        g = t / 2 + f / 2

    return max(0, g)


class Node:
    def __init__(self, data, version=1, existingFeatures=set()):
        self.data = data
        self.left = None  # True
        self.right = None  # False
        self.entropy = entropy(data)
        self.version = version
        self.existingFeatures = existingFeatures
        self.targetFeature, self.maxGain = self.findTargetFeature()

    def getPointEstimate(self):
        return Counter(d[1] for d in self.data).most_common(1)[0][0]

    def findTargetFeature(self):
        maxGain = 0
        targetFeature = None
        for i, w in enumerate(WORDS):
            if i in self.existingFeatures:
                continue
            g = gain(self.data, i, self.version)
            e = self.entropy - g
            if e > maxGain:
                maxGain = e
                targetFeature = i

        return targetFeature, maxGain

    def split(self):
        leftData = [d for d in self.data if self.targetFeature in d[0]]
        rightData = [d for d in self.data if self.targetFeature not in d[0]]

        newLeft = Node(
            leftData,
            self.version,
            self.existingFeatures | {self.targetFeature},
        )
        newRight = Node(
            rightData,
            self.version,
            self.existingFeatures | {self.targetFeature},
        )

        self.left = newLeft
        self.right = newRight

        return newLeft, newRight

    def __lt__(self, other):
        return self.maxGain < other.maxGain

    def printTree(self, level=0):
        if not self.left and not self.right:
            print(" " * 4 * level + f"Point Estimate: {self.getPointEstimate()}")
            return
        if self.targetFeature:
            print(
                " " * 4 * level
                + f"{WORDS[self.targetFeature - 1]}: I(E) = {self.maxGain}"
            )
        if self.left:
            print(" " * 4 * level + "  " + "True:")
            self.left.printTree(level + 1)
        if self.right:
            print(" " * 4 * level + "  " + "False:")
            self.right.printTree(level + 1)


def buildDecisionTree(trainData, maxNodes=100, version=1):
    pq = PriorityQueue()
    root = Node(trainData, version)
    pq.put((-root.maxGain, root))

    count = 0
    while not pq.empty() and count < maxNodes:
        node = pq.get()[1]

        leftChild, rightChild = node.split()

        pq.put((-leftChild.maxGain, leftChild))
        pq.put((-rightChild.maxGain, rightChild))

        count += 1

    return root


def main():
    # Load data
    trainData = loadData("trainData.txt", "trainLabel.txt")
    weighted = buildDecisionTree(trainData, 10, 1)
    avg = buildDecisionTree(trainData, 10, 2)

    weighted.printTree()

    # Load test data
    testData = loadData("testData.txt", "testLabel.txt")


if __name__ == "__main__":
    main()
