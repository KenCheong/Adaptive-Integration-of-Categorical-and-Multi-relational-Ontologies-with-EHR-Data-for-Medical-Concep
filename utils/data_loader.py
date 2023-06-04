import numpy as np
import random
import math
import pickle
import torch
from torch.autograd import Variable

_TEST_RATIO = 0.1
_TRAIN_RATIO = 0.8
_VALIDATION_RATIO = 0.1
#has_die='y'
has_die=None
def filter_repeat_visit(dataset,task='y'):
    dxseq=dataset[0]
    rxseq=dataset[1]
    dp=dataset[2]
    times=dataset[3]
    mo=dataset[4]

    newdxseq=[]
    newrxseq=[]
    newdp=[]
    newmo=[]
    newtime=[]
    for p in range(len(dxseq)):
        newdxseq.append([])
        newrxseq.append([])
        newdp.append([])
        newmo.append([])
        newtime.append([])
        for v in range(len(dxseq[p])):
            if len(dxseq[p][v])==0:continue
            if v<len(dxseq[p])-1:
                if set(dxseq[p][v])==set(dxseq[p][v+1]):continue
            newdxseq[-1].append(dxseq[p][v])
            newrxseq[-1].append(rxseq[p][v])
            newdp[-1].append(dp[p][v])
            if has_die!=None:
                newmo[-1].append(mo[p][v])
            newtime[-1].append(times[p][v])
        if len(newdxseq[-1])==0:
            newdxseq=newdxseq[:-1]
            newrxseq=newrxseq[:-1]
            newdp=newdp[:-1]
            if has_die!=None:
                newmo=newmo[:-1]
            newtime=newtime[:-1]
    newdataset = (newdxseq, newrxseq, newdp,newtime,newmo)
    return newdataset
def dx(dxSeqFile, dpLabelFile):
    dxSeqs = np.array(pickle.load(open(dxSeqFile, 'rb')))
    dpLabels = np.array(pickle.load(open(dpLabelFile, 'rb')))

    np.random.seed(0)
    dataSize = len(dxSeqs)
    ################## Random ##################
    ind = np.random.permutation(dataSize)
    ################## END ##################
    # ind = np.arange(dataSize)
    # print('ind1:', ind.shape, ind)

    # train: 0~80%, valid: 80~90%, test: 90~100%
    nTest = int(_TEST_RATIO * dataSize)
    nValid = int(_VALIDATION_RATIO * dataSize)
    nTrain = int((1-_TEST_RATIO-_VALIDATION_RATIO)*dataSize)
    # nTrain = int(_TRAIN_RATIO * dataSize)

    test_indices = ind[(dataSize-nTest):]
    valid_indices = ind[(dataSize-nTest-nValid):(dataSize-nTest)]
    train_indices = ind[:(dataSize-nTest-nValid)]

    train_set_x = dxSeqs[train_indices]
    train_set_dp_y = dpLabels[train_indices]
    test_set_x = dxSeqs[test_indices]
    test_set_dp_y = dpLabels[test_indices]
    valid_set_x = dxSeqs[valid_indices]
    valid_set_dp_y = dpLabels[valid_indices]

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    train_sorted_index = len_argsort(train_set_x)
    train_set_x = [train_set_x[i] for i in train_sorted_index]
    train_set_dp_y = [train_set_dp_y[i] for i in train_sorted_index]

    valid_sorted_index = len_argsort(valid_set_x)
    valid_set_x = [valid_set_x[i] for i in valid_sorted_index]
    valid_set_dp_y = [valid_set_dp_y[i] for i in valid_sorted_index] 

    test_sorted_index = len_argsort(test_set_x)
    test_set_x = [test_set_x[i] for i in test_sorted_index]
    test_set_dp_y = [test_set_dp_y[i] for i in test_sorted_index]

    train_set = (train_set_x, train_set_dp_y)
    valid_set = (valid_set_x, valid_set_dp_y)
    test_set = (test_set_x, test_set_dp_y)

    return train_set, valid_set, test_set

def dxrx(dxSeqsFile, drugSeqsFile, dpLabelFile,moLabelFile,timeSeqFile=None,datasource=None,task='a'):
    drugSeqs = np.array(pickle.load(open(drugSeqsFile, 'rb')))
    dxSeqs = np.array(pickle.load(open(dxSeqsFile, 'rb')))
    dpLabels = np.array(pickle.load(open(dpLabelFile, 'rb')))
    if has_die!=None:
        moLabels = np.array(pickle.load(open(moLabelFile, 'rb')))
    if datasource=='eicu':
        for i in range(len(moLabels)):
            for j in range(len(moLabels[i])):
                if moLabels[i][j]=='Alive':
                    moLabels[i][j]=1.0
                else:
                    moLabels[i][j]=0.0
    accu_counts=np.array([0]*15000)
    if timeSeqFile!=None:
        timeseq = np.array(pickle.load(open(timeSeqFile, 'rb')))
        newdrugSeqs=[]
        newdxSeqs=[]
        newdpSeqs=[]
        newmoSeqs=[]
        for p in range(len(timeseq)):
            timedefseq.append([])
            #if len(timeseq[p])<2:continue
            newdrugSeqs.append([drugSeqs[p][0]])
            newdxSeqs.append([dxSeqs[p][0]])
            newdpSeqs.append([dpLabels[p][0]])
            if has_die!=None:
                newmoSeqs.append([moLabels[p][0]])
            for t in range(len(timeseq[p])-1):

                delta = timeseq[p][t+1]-timeseq[p][t]
                accu_counts[:delta.days]+=1
                timedefseq[-1].append(delta.days)
#                print(delta)
#                if delta.days>=1500:
 #                   print(delta.days)
                if delta.days<=30000:
                    newdrugSeqs[-1].append(drugSeqs[p][t+1])
                    newdxSeqs[-1].append(dxSeqs[p][t+1])
                    newdpSeqs[-1].append(dpLabels[p][t+1])

                    if task!=None:
                        newmoSeqs[-1].append(moLabels[p][t+1])
                else:break
            timedefseq[-1].append(10000000)
            if len(newdrugSeqs[-1])==1:
                newdrugSeqs=newdrugSeqs[:-1]
                newdxSeqs=newdxSeqs[:-1]
                newdpSeqs=newdpSeqs[:-1]
                newmoSeqs=newmoSeqs[:-1]
        timedefseq=np.array(timedefseq)
        drugSeqs=np.array(newdrugSeqs)
        dxSeqs=np.array(newdxSeqs)
        dpLabels=np.array(newdpSeqs)

        if has_die!=None:
            moLabels=np.array(newmoSeqs)
    else:

        timedefseq=dxSeqs
#        print(','.join([str(c) for c in accu_counts[0]-accu_counts[:4000]]))
#        1/0
#    print(drugSeqs[:2])
 #   print(dxSeqs[:2])
  #  print(dpLabels[:2])
    #1/0

    #print(dxSeqs,dpLabels,dxSeqs.shape)
    #1/0
    train_set_mort=[]
    valid_set_mort=[]
    test_set_mort=[]
    np.random.seed(0)
    dataSize = len(dxSeqs)
    ################## Random ##################
    ind = np.random.permutation(dataSize)
    ################## END ##################
    # ind = np.arange(dataSize)
    # print('ind1:', ind.shape, ind)

    # train: 0~80%, valid: 80~90%, test: 90~100%
    nTest = int(_TEST_RATIO * dataSize)
    nValid = int(_VALIDATION_RATIO * dataSize)
    nTrain = int((1-_TEST_RATIO-_VALIDATION_RATIO)*dataSize)
    # nTrain = int(_TRAIN_RATIO * dataSize)

    test_indices = ind[(dataSize-nTest):]
    valid_indices = ind[(dataSize-nTest-nValid):(dataSize-nTest)]
    train_indices = ind[:(dataSize-nTest-nValid)]

    train_set_dx_x = dxSeqs[train_indices]
    train_set_drug_x = drugSeqs[train_indices]
    train_set_dp_y = dpLabels[train_indices]
    train_set_time = timedefseq[train_indices]

    if has_die!=None:
        train_set_mort = moLabels[train_indices]

    test_set_dx_x = dxSeqs[test_indices]
    test_set_drug_x = drugSeqs[test_indices]
    test_set_dp_y = dpLabels[test_indices]
    test_set_time = timedefseq[test_indices]

    if has_die!=None:
        test_set_mort = moLabels[test_indices]

    valid_set_dx_x = dxSeqs[valid_indices]
    valid_set_drug_x = drugSeqs[valid_indices]
    valid_set_dp_y = dpLabels[valid_indices]
    valid_set_time = timedefseq[valid_indices]

   

    if has_die!=None:
        valid_set_mort = moLabels[valid_indices]

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    train_sorted_index = len_argsort(train_set_dx_x)
    train_set_dx_x = [train_set_dx_x[i] for i in train_sorted_index]
    train_set_drug_x = [train_set_drug_x[i] for i in train_sorted_index]
    train_set_dp_y = [train_set_dp_y[i] for i in train_sorted_index]
    train_set_time = [train_set_time[i] for i in train_sorted_index]
    if has_die!=None:
        train_set_mort = [train_set_mort[i] for i in train_sorted_index]

    valid_sorted_index = len_argsort(valid_set_dx_x)
    valid_set_dx_x = [valid_set_dx_x[i] for i in valid_sorted_index]
    valid_set_drug_x = [valid_set_drug_x[i] for i in valid_sorted_index]
    valid_set_dp_y = [valid_set_dp_y[i] for i in valid_sorted_index]
    valid_set_time = [valid_set_time[i] for i in valid_sorted_index]

    if has_die!=None:
        valid_set_mort = [valid_set_mort[i] for i in valid_sorted_index]

    test_sorted_index = len_argsort(test_set_dx_x)
    test_set_dx_x = [test_set_dx_x[i] for i in test_sorted_index]
    test_set_drug_x = [test_set_drug_x[i] for i in test_sorted_index]
    test_set_dp_y = [test_set_dp_y[i] for i in test_sorted_index]
    test_set_time = [test_set_time[i] for i in test_sorted_index]

    if has_die!=None:
        test_set_mort = [test_set_mort[i] for i in test_sorted_index]
    
    train_set = (train_set_dx_x, train_set_drug_x, train_set_dp_y,train_set_time,train_set_mort)
    valid_set = (valid_set_dx_x, valid_set_drug_x, valid_set_dp_y,valid_set_time,valid_set_mort)
    test_set = (test_set_dx_x, test_set_drug_x, test_set_dp_y,test_set_time,test_set_mort)

    return train_set, valid_set, test_set
