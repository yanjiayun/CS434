from numpy import *
import numpy as np
import operator

dataNum = 8000
testNum = 2000
dimensions = 85

def read():
    with open("train.csv", "r") as file1:
        train = [i[:-1].split(',') for i in file1.readlines()]
    with open("test_pub.csv", "r") as file2:
        test = [i[:-1].split(',') for i in file2.readlines()]
    label = []     
    for i in range(1,dataNum+1):
        label.append(int(train[i][86]))
    return train, test, label

def kNNClassify(test, train, labels, k):
    train_final = np.asarray(train, dtype=np.float64, order='C')
    test_final = np.asarray(test, dtype=np.float64, order='C')
    
    different = tile(test_final, (dataNum,1)) - train_final #!!!
    #different = tile(test_final, (6000,1)) - train_final
    differebt_square = different  ** 2  
    distance_square = sum(differebt_square, axis = 1)   
    distance = distance_square ** 0.5
    sortedDistIndex = argsort(distance)
    
    greater = 0
    less = 0
    j = 0
    for j in range(0,k-1):
        if (labels[sortedDistIndex[j]] == 1):
            greater = greater + 1 
        else:
            less = less + 1
    
    if greater > less:
        return 1
    else:
        return 0

def main():
    train, test, label = read()
    k = 99

    test_data = np.delete(test, 0, axis=0)
    test_data2 = np.delete(test_data, 0, axis=1)
    
    
    train_data = np.delete(train, 0, axis=0)
    train_data2 = np.delete(train_data, 86, axis=1)
    train_data3 = np.delete(train_data2, 0, axis=1) #!!!
    
    '''
    train_final = np.delete(train_data2, 0, axis=1)
    set1 = []
    set2 = []
    set3 = []
    set4 = []
    other1 = []
    other2 = []
    other3 = []
    other4 = []
    result1 = []
    result2 = []
    result3 = []
    result4 = []
    for x in range(0,2000):
        set1.append(train_final[x])
    for x in range(2000,4000):
        set2.append(train_final[x])
    for x in range(4000,6000):
        set3.append(train_final[x])
    for x in range(6000,8000):
        set4.append(train_final[x])

    for x in range(2000,8000):
        other1.append(train_final[x])
    
    for x in range(0,8000):
        if x<2000:
            other2.append(set1[x])
        elif x>=4000 and x<6000:
            other2.append(set3[x-4000])
        elif x>=6000:
            other2.append(set4[x-6000])

    for x in range(0,8000):
        if x<2000:
            other3.append(set1[x])
        elif x>=2000 and x<4000:
            other3.append(set2[x-2000])
        elif x>=6000:
            other3.append(set4[x-6000])

    for x in range(0,6000):
        other4.append(train_final[x])

    t1=0
    for i in range(0,2000):
        result1.append(kNNClassify(set1[i], other1, label, k))
        if result1[i] == label[i]:
            t1 = t1+1

    t2=0     
    for i in range(0,2000):
        result2.append(kNNClassify(set2[i], other2, label, k))
        if result2[i] == label[i+2000]:
            t2 = t2+1

    t3=0
    for i in range(0,2000):
        result3.append(kNNClassify(set3[i], other3, label, k))
        if result3[i] == label[i+4000]:
            t3 = t3+1
    
    t4=0
    for i in range(0,2000):
        result4.append(kNNClassify(set4[i], other4, label, k))
        if result4[i] == label[i+6000]:
            t4 = t4+1

    p1 = t1/2000.0
    p2 = t2/2000.0
    p3 = t3/2000.0
    p4 = t4/2000.0
    print("t1: ", t1, ",p1: ", p1)
    print("t2: ", t2, ",p2: ", p2)
    print("t3: ", t3, ",p3: ", p3)
    print("t4: ", t4, ",p4: ", p4)
    print("p_final:", (p1+p2+p3+p4)/4.0)
    '''
    outputFile = open("output.csv", "w") #!!!
    outputFile.write('%s,' % "id") #!!!
    outputFile.write('%s\n' % "income") #!!!
    for i in range(0,testNum): #!!!
        outputFile.write('%d, ' % i) #!!!
        outputFile.write('%d\n' % kNNClassify(test_data2[i], train_data3, label, k)) #!!!

main()