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


def calculateEntropy(data, dataSet):
    if not dataSet:
        return 1

    probA = Counter(data[d][1] for d in dataSet)[1] / len(dataSet)
    probB = 1 - probA

    eA = -1 * probA * (0 if probA == 0 else math.log2(probA))
    eB = -1 * probB * (0 if probB == 0 else math.log2(probB))

    return eA + eB


def calculateGain(data, dataSet, feature, v=1):
    if not dataSet:
        return 0
    t = calculateEntropy(data, [d for d in dataSet if feature in data[d][0]])
    f = calculateEntropy(data, [d for d in dataSet if feature not in data[d][0]])

    g = 0

    if v == 1:
        n1, n2, total = 0, 0, 0
        for d in dataSet:
            total += 1
            if feature in data[d][0]:
                n1 += 1
            else:
                n2 += 1

        g = n1 / total * t + n2 / total * f

    else:
        g = t / 2 + f / 2

    return max(0, g)


class Node:
    def __init__(self, data, dataSet, version=1, existingFeatures=set()):
        self.data = data
        self.dataSet = dataSet
        self.left = None  # True
        self.right = None  # False
        self.entropy = calculateEntropy(data, dataSet)
        self.version = version
        self.existingFeatures = existingFeatures
        self.targetFeature, self.maxGain = self.findTargetFeature()

    def getPointEstimate(self):
        return Counter(self.data[d][1] for d in self.dataSet).most_common(1)[0][0]

    def findTargetFeature(self):
        maxGain = 0
        targetFeature = None
        for i, w in enumerate(WORDS):
            if i in self.existingFeatures:
                continue
            g = calculateGain(self.data, self.dataSet, i, self.version)
            e = self.entropy - g
            if e > maxGain:
                maxGain = e
                targetFeature = i

        return targetFeature, maxGain

    def split(self):
        leftData = [d for d in self.dataSet if self.targetFeature in self.data[d][0]]
        rightData = [
            d for d in self.dataSet if self.targetFeature not in self.data[d][0]
        ]

        newLeft = Node(
            self.data,
            leftData,
            self.version,
            self.existingFeatures | {self.targetFeature},
        )
        newRight = Node(
            self.data,
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
                + f"{WORDS[self.targetFeature - 1]}: Information Gain = {self.maxGain}"
            )
        if self.left:
            print(" " * 4 * level + "  " + "True:")
            self.left.printTree(level + 1)
        if self.right:
            print(" " * 4 * level + "  " + "False:")
            self.right.printTree(level + 1)


def predictExample(node, example):
    if not node.left and not node.right:
        return node.getPointEstimate()

    if node.targetFeature in example[0]:
        return predictExample(node.left, example)
    return predictExample(node.right, example)


def calculateAccuracy(root, data):
    correct = 0
    for example in data:
        prediction = predictExample(root, example)
        if prediction == example[1]:
            correct += 1
    return (correct / len(data)) * 100


def buildDecisionTree(trainData, testData, maxNodes=100, version=1):
    pq = PriorityQueue()
    root = Node(trainData, [i for i in range(len(trainData))], version)
    pq.put((-root.maxGain, root))

    trainAccuracies = []
    testAccuracies = []

    count = 0
    while not pq.empty() and count < maxNodes:
        node = pq.get()[1]
        leftChild, rightChild = node.split()

        pq.put((-leftChild.maxGain, leftChild))
        pq.put((-rightChild.maxGain, rightChild))

        count += 1

        trainAcc = calculateAccuracy(root, trainData)
        testAcc = calculateAccuracy(root, testData)
        trainAccuracies.append(trainAcc)
        testAccuracies.append(testAcc)

    return root, trainAccuracies, testAccuracies


def plotAccuracies(trainAcc1, testAcc1, trainAcc2, testAcc2):
    nodes = range(1, len(trainAcc1) + 1)

    # Create two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot weighted version (v=1)
    ax1.plot(nodes, trainAcc1, label="Training Accuracy", color="blue")
    ax1.plot(nodes, testAcc1, label="Testing Accuracy", color="red")
    ax1.set_title("Weighted Information Gain")
    ax1.set_xlabel("# of Internal Nodes")
    ax1.set_ylabel("Accuracy (%)")
    ax1.legend()
    ax1.grid(True)

    # Plot average version (v=2)
    ax2.plot(nodes, trainAcc2, label="Training Accuracy", color="blue")
    ax2.plot(nodes, testAcc2, label="Testing Accuracy", color="red")
    ax2.set_title("Average Information Gain")
    ax2.set_xlabel("# of Internal Nodes")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("accuracy_plots.png")
    plt.close()


def main():
    # Load data
    trainData = loadData("trainData.txt", "trainLabel.txt")
    testData = loadData("testData.txt", "testLabel.txt")

    # part 2
    # weightedTree, weightedTrainAcc, weightedTestAcc = buildDecisionTree(
    #     trainData, testData, 10, 1
    # )
    # avgTree, avgTrainAcc, avgTestAcc = buildDecisionTree(trainData, testData, 10, 2)
    # print("Weighted Tree:")
    # weightedTree.printTree()
    # print("\nAverage Tree:")
    # avgTree.printTree()

    # part 3
    # Build trees and get accuracies
    weightedTree, weightedTrainAcc, weightedTestAcc = buildDecisionTree(
        trainData, testData, 100, 1
    )
    avgTree, avgTrainAcc, avgTestAcc = buildDecisionTree(trainData, testData, 100, 2)

    # Plot results
    plotAccuracies(weightedTrainAcc, weightedTestAcc, avgTrainAcc, avgTestAcc)

    print("Plots have been saved to 'accuracy_plots.png'")


if __name__ == "__main__":
    main()
