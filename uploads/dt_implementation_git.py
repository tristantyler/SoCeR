import re
import math
from collections import deque

# x is examples in training set
# y is set of attributes
# label is target attributes
# Node is a class which has properties values, childs, and next
# root is top node in the decision tree
import pandas as pd
training_data = [
({'level':'Senior', 'lang':'Java', 'tweets':'no', 'phd':'no'}, False),
({'level':'Senior', 'lang':'Java', 'tweets':'no', 'phd':'yes'}, False),
({'level':'Mid', 'lang':'Python', 'tweets':'no', 'phd':'no'}, True),
({'level':'Junior', 'lang':'Python', 'tweets':'no', 'phd':'no'}, True),
({'level':'Junior', 'lang':'R', 'tweets':'yes', 'phd':'no'}, True),
({'level':'Junior', 'lang':'R', 'tweets':'yes', 'phd':'yes'}, False),
({'level':'Mid', 'lang':'R', 'tweets':'yes', 'phd':'yes'}, True),
({'level':'Senior', 'lang':'Python', 'tweets':'no', 'phd':'no'}, False),
({'level':'Senior', 'lang':'R', 'tweets':'yes', 'phd':'no'}, True),
({'level':'Junior', 'lang':'Python', 'tweets':'yes', 'phd':'no'}, True),
({'level':'Senior', 'lang':'Python', 'tweets':'yes', 'phd':'yes'}, True),
({'level':'Mid', 'lang':'Python', 'tweets':'no', 'phd':'yes'}, True),
({'level':'Mid', 'lang':'Java', 'tweets':'yes', 'phd':'no'}, True),
({'level':'Junior', 'lang':'Python', 'tweets':'no', 'phd':'yes'}, False)
] 


pd.DataFrame(training_data) 
df=pd.DataFrame([row[0] for row in training_data])
df["Result"]=[row[1] for row in training_data]
dataset=df.values.tolist()
class Node(object):
    def __init__(self):
        self.value = None
        self.next = None
        self.childs = None

# Simple class of Decision Tree
# Aimed for who want to learn Decision Tree, so it is not optimized
class DecisionTree(object):
    def __init__(self, sample, attributes, labels):
        self.sample = sample
        self.attributes = attributes
        self.labels = labels
        self.labelCodes = None
        self.labelCodesCount = None
        self.initLabelCodes()
        # print(self.labelCodes)
        self.root = None
        self.entropy = self.getEntropy([x for x in range(len(self.labels))])

    def initLabelCodes(self):
        self.labelCodes = []
        self.labelCodesCount = []
        for l in self.labels:
            if l not in self.labelCodes:
                self.labelCodes.append(l)
                self.labelCodesCount.append(0)
            self.labelCodesCount[self.labelCodes.index(l)] += 1

    def getLabelCodeId(self, sampleId):
        return self.labelCodes.index(self.labels[sampleId])

    def getAttributeValues(self, sampleIds, attributeId):
        vals = []
        for sid in sampleIds:
            val = self.sample[sid][attributeId]
            if val not in vals:
                vals.append(val)
        # print(vals)
        return vals

    def getEntropy(self, sampleIds):
        entropy = 0
        labelCount = [0] * len(self.labelCodes)
        for sid in sampleIds:
            labelCount[self.getLabelCodeId(sid)] += 1
        # print("-ge", labelCount)
        for lv in labelCount:
            # print(lv)
            if lv != 0:
                entropy += -lv/len(sampleIds) * math.log(lv/len(sampleIds), 2)
            else:
                entropy += 0
        return entropy

    def getDominantLabel(self, sampleIds):
        labelCodesCount = [0] * len(self.labelCodes)
        for sid in sampleIds:
            labelCodesCount[self.labelCodes.index(self.labels[sid])] += 1
        return self.labelCodes[labelCodesCount.index(max(labelCodesCount))]

    def getInformationGain(self, sampleIds, attributeId):
        gain = self.getEntropy(sampleIds)
        attributeVals = []
        attributeValsCount = []
        attributeValsIds = []
        for sid in sampleIds:
            val = self.sample[sid][attributeId]
            if val not in attributeVals:
                attributeVals.append(val)
                attributeValsCount.append(0)
                attributeValsIds.append([])
            vid = attributeVals.index(val)
            attributeValsCount[vid] += 1
            attributeValsIds[vid].append(sid)
        # print("-gig", self.attributes[attributeId])
        for vc, vids in zip(attributeValsCount, attributeValsIds):
            # print("-gig", vids)
            gain -= vc/len(sampleIds) * self.getEntropy(vids)
        return gain

    def getAttributeMaxInformationGain(self, sampleIds, attributeIds):
        attributesEntropy = [0] * len(attributeIds)
        for i, attId in zip(range(len(attributeIds)), attributeIds):
            attributesEntropy[i] = self.getInformationGain(sampleIds, attId)
        maxId = attributeIds[attributesEntropy.index(max(attributesEntropy))]
        return self.attributes[maxId], maxId

    def isSingleLabeled(self, sampleIds):
        label = self.labels[sampleIds[0]]
        for sid in sampleIds:
            if self.labels[sid] != label:
                return False
        return True

    def getLabel(self, sampleId):
        return self.labels[sampleId]

    def id3(self):
        sampleIds = [x for x in range(len(self.sample))]
        attributeIds = [x for x in range(len(self.attributes))]
        self.root = self.id3Recv(sampleIds, attributeIds, self.root)

    def id3Recv(self, sampleIds, attributeIds, root):
        root = Node() # Initialize current root
        if self.isSingleLabeled(sampleIds):
            root.value = self.labels[sampleIds[0]]
            return root
        # print(attributeIds)
        if len(attributeIds) == 0:
            root.value = self.getDominantLabel(sampleIds)
            return root
        bestAttrName, bestAttrId = self.getAttributeMaxInformationGain(
            sampleIds, attributeIds)
        # print(bestAttrName)
        root.value = bestAttrName
        root.childs = []  # Create list of children
        for value in self.getAttributeValues(sampleIds, bestAttrId):
            # print(value)
            child = Node()
            child.value = value
            root.childs.append(child)  # Append new child node to current
                                       # root
            childSampleIds = []
            for sid in sampleIds:
                if self.sample[sid][bestAttrId] == value:
                    childSampleIds.append(sid)
            if len(childSampleIds) == 0:
                child.next = self.getDominantLabel(sampleIds)
            else:
                # print(bestAttrName, bestAttrId)
                # print(attributeIds)
                if len(attributeIds) > 0 and bestAttrId in attributeIds:
                    toRemove = attributeIds.index(bestAttrId)
                    attributeIds.pop(toRemove)
                child.next = self.id3Recv(
                    childSampleIds, attributeIds, child.next)
        return root

    def printTree(self):
        if self.root:
            roots = deque()
            roots.append(self.root)
            while len(roots) > 0:
                root = roots.popleft()
                print("{")
                print(root.value)
                if root.childs:
                    for child in root.childs:
                        print('({})'.format(child.value))
                        roots.append(child.next)
                elif root.next:
                    print(root.next)
                    

def test():
##    f = open('playtennis.csv')
##    attributes = f.readline().split(',')
##    attributes = attributes[1:len(attributes)-1]
##    print(attributes)
    attributes=["level","lang","phd","tweets"]
##    print(attributes)
##    sample = f.readlines()
##    f.close()
    sample=[]
    labels=[]
    
        
##    for i in range(len(sample)):
##        sample[i] = re.sub('\d+,', '', sample[i])
##        sample[i] = sample[i].strip().split(',')
##    labels = []
##    for s in sample:
##        labels.append(s.pop())
##    print(dataset)
    
    ##-------- assigning new samples and labels
    for tup in dataset:
        size=len(tup)
        sample.append(tup[:size-1])
        labels.append(tup[size-1])

    print(sample)
    print(labels)

    
    decisionTree = DecisionTree(sample, attributes, labels)
    print("System entropy {}".format(decisionTree.entropy))
    decisionTree.id3()
    decisionTree.printTree()


if __name__ == '__main__':
    test()
