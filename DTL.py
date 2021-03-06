import pandas as pd
import math as math

from collections import Counter
from treelib import Node, Tree

def read_file(namefile) :
    data = pd.read_csv(namefile)
    # Menentukan Attribute
    header = list(data.head(0))
    targetvalue = [header[len(header)-1]]
    header.pop(len(header)-1)
    print(header)

    # Menentukan Examples
    row = []
    for i in data.iterrows():
        row.append(list(i))
    
    i = len(row)
    examplesize = 0.8*len(row)
    while (i > examplesize) : 
        row.pop(len(row)-1)
        i = i - 1

    index, name = header_labelling(header)
    
    # Mengambil target value
    for i in range (0,len(row)-1):
        exist = False
        for j in targetvalue :
            if (row[i][1][len(row[0][1])-1] == j) :
                exist = True
        if (not exist) :
            targetvalue.append(row[i][1][len(row[0][1])-1])

    tennisData=data.iloc[:, :5]
    tennisData = fixMissingValue(tennisData)
    countTrainingData = math.floor(len(tennisData)*0.8)
    tennisData = tennisData[0:countTrainingData]
    tennisSet = []
    for a in tennisData:
        tennisSet.append(list(set(tennisData[a])))
    

    data = {
        'header' : header,
        'targetvalue' : targetvalue,
        'example' : row,
        'node' : tennisSet,
        'name' : name,
        'index' : index
    }
 
    return data,tennisData

def fixMissingValue(tennisData):
    print(tennisData)
    for i in tennisData.head(0) :
        counterAttribute = Counter(tennisData[i])
        labels = tennisData[i].unique()
        max = [labels[0],counterAttribute[labels[0]]]
        for label in labels :
            if (counterAttribute[label] > max[1]):    
                max = [label,counterAttribute[label]]
                
        tennisData[i].fillna(max[0],inplace=True)
    
    print(tennisData)
                
    
    # tennisData['outlook'].fillna("sunny",inplace=True)
    # print(tennisData)
    return tennisData

def ID3(examples,target,attributes,tree_parent,tree,usedLabels):
    dataCount=Counter(examples['play'])
    for targetValue in target:
        if(dataCount[targetValue]==len(examples)):
            tree.create_node(targetValue,parent=tree_parent)
            return

    if(len(attributes)==0):     
        max=-999        
        dataReturn=''
        for sumData in dataCount:
            if(dataCount[sumData]>max):
                max=dataCount[sumData]
                dataReturn=sumData
        tree.create_node(sumData,sumData,parent=tree_parent)
    max=-999
    for att in attributes :
        gainTemp=information_gain(examples,att,target)
        if(max<gainTemp):
            max=gainTemp
            selectedAttributes=att    
    selectedAttributesValues=examples[selectedAttributes].unique()
    for values in selectedAttributesValues:
        tree.show()
        if (values not in usedLabels):
            tree.create_node(selectedAttributes+" = "+values,values,parent=tree_parent)
            usedLabels.append(values)
        newData=examples.loc[examples[selectedAttributes]==values]
        if(len(newData)==0):
            max=-999
            dataReturn=''
            for sumData in dataCount:
                if(dataCount[sumData]>max):
                    max=dataCount[sumData]
                    dataReturn=sumData
            tree.create_node(sumData,sumData,parent=tree_parent)
            
        else:
            print(attributes)
            print(selectedAttributes)
            while (selectedAttributes in attributes) :   
                attributes.remove(selectedAttributes)
            print(attributes)
            tree.show()
            ID3(newData,target,attributes,values,tree,usedLabels)
            attributes.append(selectedAttributes)
        attributes.append(selectedAttributes)

def header_labelling(header):
    name = {}
    index = {}
    for i in range(0,len(header)) :
        index[header[i]] = i
        name[i] = header[i]
    
    return index,name


def entropy(arrentropy):
    count_entropy = 0
    for i in range (0, len(arrentropy)):
        if (arrentropy[i] != 0) :
            count_entropy +=(-1*(arrentropy[i]*math.log(arrentropy[i],2)))
    return count_entropy

def information_gain(example,att,target):
    # Hitung Entropy semua
    countAll = Counter(example[target[0]])
    total=len(example)
    arrentropy=[]
    for i in range(1,len(target)): 
        arrentropy.append(countAll[target[i]]/total)
    entropyAll = entropy(arrentropy)

    labels = example[att].unique()
    entropyLabel = 0
    for label in labels :
        countTarget = []
        totalLabel = 0
        subexample = example.loc[example[att]==label]
        countTarget = Counter(subexample[target[0]])
        totalLabel=len(subexample)
        arrentropyLabel = []
        for i in range (1,len(target)) :
            arrentropyLabel.append(countTarget[target[i]]/totalLabel)
        entropyLabel += totalLabel/total * entropy(arrentropyLabel)


    return (entropyAll - entropyLabel)
    
         
if __name__ == '__main__':
    data,tennisData=read_file("play-tennis2.csv")
    tree=Tree()
    tree.create_node("Root","root")
    ID3(tennisData,data['targetvalue'],data['header'],"root",tree,[])
    tree.show()