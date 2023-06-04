import random
import numpy as np
import torch
import copy

from utils import const

def get_mo_acc( crit, preds,targets ,mo_mask):
    loss = crit(preds, targets)
    correct_dx_num = total_dx_num = 0
    patient_num = preds.size()[0]
    visit_num = preds.size()[1]
    pos_corr=0.0
    true_pos=0.0
    true_neg=0.0
    neg_corr=0.0
    Y=[]
    preY=[]
#    preds = preds.view(patient_num*visit_num, -1)
 #   targets = targets.view(patient_num*visit_num, -1)
    tt=0.0
    ttt=0.0
  #  print(targets.)
    for i in range(patient_num):
        for j in range(visit_num):
            if mo_mask[i][j][0]==0:continue
            total_dx_num+=1
#            print(j,visit_num)
            Y.append(targets.detach().cpu().numpy()[i][j])
            preY.append(preds[i][j].detach().cpu().numpy()[0])
            if targets[i][j]==1:
                tt+=1
                ttt+=preds[i][j]
                if preds[i][j]>0.4:
                    pos_corr+=1
                    correct_dx_num+=1
                true_pos+=1
            elif targets[i][j]==0:
                if preds[i][j]<0.4:
                    correct_dx_num+=1
                    neg_corr+=1
                true_neg+=1
#    print('tr',pos_corr/true_pos, 'nr',neg_corr/true_neg)
#                break
#    print(preds,Y)
 #   1/0
#    print('rem',Y[:10],'\n',targets.detach().cpu().numpy()[:5],np.sum(np.array(re_mask)),len(Y))
 #   1/0
    return loss, correct_dx_num, total_dx_num,Y,preY,pos_corr,true_pos,neg_corr,true_neg
def get_re_acc( crit, preds, targets,re_mask):
    loss = crit(preds, targets)
    correct_dx_num = total_dx_num = 0
    patient_num = preds.size()[0]
    visit_num = preds.size()[1]
    pos_corr=0.0
    true_pos=0.0
    true_neg=0.0
    neg_corr=0.0
    Y=[]
    preY=[]
#    preds = preds.view(patient_num*visit_num, -1)
 #   targets = targets.view(patient_num*visit_num, -1)
    tt=0.0
    ttt=0.0
  #  print(targets.)
    for i in range(patient_num):
        for j in range(visit_num):
            if re_mask[i][j][0]==0:continue
            total_dx_num+=1
#            print(j,visit_num)
            Y.append(targets.detach().cpu().numpy()[i][j])
            preY.append(preds[i][j].detach().cpu().numpy()[0])
            if targets[i][j]==1:
                tt+=1
                ttt+=preds[i][j]
                if preds[i][j]>0.4:
                    pos_corr+=1
                    correct_dx_num+=1
                true_pos+=1
            elif targets[i][j]==0:
                if preds[i][j]<0.4:
                    correct_dx_num+=1
                    neg_corr+=1
                true_neg+=1
#    print('tr',pos_corr/true_pos, 'nr',neg_corr/true_neg)
#                break
#    print(preds,Y)
 #   1/0
#    print('rem',Y[:10],'\n',targets.detach().cpu().numpy()[:5],np.sum(np.array(re_mask)),len(Y))
 #   1/0
    return loss, correct_dx_num, total_dx_num,Y,preY,pos_corr,true_pos,neg_corr,true_neg


def get_mo_mask(labels):
    max_visit_num = np.max(np.array([len(p) for p in labels]))
    new_labels = []
    mask = []
    for p in labels:
        mask_p = []
        label_p = []
    #    print(p)
#        print(visit)
        for visit in p:
            mask_p.append(np.array([1]))
#        new_v =label 
        label_p=copy.copy(p)
        if len(mask_p) < max_visit_num:
            mask_p.extend([np.array([0])] * (max_visit_num-len(label_p)))
            label_p.extend([0] * (max_visit_num-len(label_p)))
        mask.append(np.array(mask_p))
        new_labels.append(np.array(label_p))
        '''
        if label_p[-2]==0 and mask_p[-1]==1:
            print(label_p,mask_p,p,labels,max_visit_num)
            1/0
        '''
        #print(max_visit_num,label_p,new_labels,labels)
        #print(new_labels)
        #1/0
    #print('grm',mask[:5],'\n',new_labels[:5],max_visit_num)
    return torch.FloatTensor(new_labels), torch.FloatTensor(mask)
def get_re_mask(labels):
    max_visit_num = np.max(np.array([len(p) for p in labels]))
    new_labels = []
    mask = []
    for p in labels:
        mask_p = []
        label_p = []
    #    print(p)
#        print(visit)
        for visit in p:
            mask_p.append(np.array([1]))
#        new_v =label 
        label_p=copy.copy(p)
        if len(mask_p) < max_visit_num:
            mask_p.extend([np.array([0])] * (max_visit_num-len(label_p)))
            label_p.extend([0] * (max_visit_num-len(label_p)))
        mask.append(np.array(mask_p))
        new_labels.append(np.array(label_p))
        '''
        if label_p[-2]==0 and mask_p[-1]==1:
            print(label_p,mask_p,p,labels,max_visit_num)
            1/0
        '''
        #print(max_visit_num,label_p,new_labels,labels)
        #print(new_labels)
        #1/0
    #print('grm',mask[:5],'\n',new_labels[:5],max_visit_num)
    return torch.FloatTensor(new_labels), torch.FloatTensor(mask)
def get_dp_mask(labels, labelSize):
    max_visit_num = np.max(np.array([len(p) for p in labels]))
    new_labels = []
    mask = []
    for p in labels:
        mask_p = []
        label_p = []
        for visit in p:
            mask_p.append(np.array([1]*labelSize))
            new_v = np.array([0]*labelSize)
            for label in visit:
                new_v[label] = 1
            label_p.append(new_v)
        if len(mask_p) < max_visit_num:
            mask_p.extend([np.array([0]*labelSize)] * (max_visit_num-len(label_p)))
            label_p.extend([np.array([0]*labelSize)] * (max_visit_num-len(label_p)))
        mask.append(np.array(mask_p[1:]))
        new_labels.append(np.array(label_p[1:]))
    return torch.FloatTensor(new_labels), torch.FloatTensor(mask)

def get_seqs(seqs, args, codetype,task='nextdx'):
    if codetype == 'dx':
        #padid = const.PAD_DXID
        padid = args.dxVocabSize
        vocabSize = args.dxVocabSize
    elif codetype == 'drug':
        #padid = const.PAD_DRUGID
        padid = args.drugVocabSize
        vocabSize =args.drugVocabSize 
        #1/0
    else:
        padid = const.PAD_ID
    visit_num = np.array([len(p) for p in seqs])
    max_visit_num = np.max(visit_num)     
    code_num = []
    for p in seqs:
        max_dx_num = np.max(np.array([len(v) for v in p]))
        code_num.append(max_dx_num)
    max_code_num = np.max(np.array(code_num))
    new_seqs = []
#    print(seqs,max_code_num,padid,max_visit_num, args.batchSize, vocabSize)
    for p in seqs:
        new_p = []
        for v in p:
            new_v = v[:]
            if len(v) < max_code_num: 
                new_v.extend([padid]*(max_code_num-len(v)))
            new_p.append(new_v)
        if len(p) < max_visit_num:
            new_p.extend([[padid]*max_code_num]*(max_visit_num-len(p)))
        if max_visit_num > 1:
            if task=='nextdx':
                new_seqs.append(new_p[:-1])
            elif task=='readm':
                #new_seqs.append(new_p[:-1])
                new_seqs.append(new_p)
            elif task=='mortality':
                new_seqs.append(new_p)

    if task=='nextdx':
        lengths = np.array([len(seq) for seq in seqs]) - 1
    elif task=='readm':
        #lengths = np.array([len(seq) for seq in seqs]) -1
        lengths = np.array([len(seq) for seq in seqs]) 
    elif task=='mortality':
        lengths = np.array([len(seq) for seq in seqs]) 
    max_visit_num = np.max(lengths)
    if max_visit_num != 0:
        onehot = np.zeros((max_visit_num, args.batchSize, vocabSize))
        for idx, seq in enumerate(seqs):
            if task=='nextdx':
                for xvec, subseq in zip(onehot[:,idx,:], seq[:-1]): 
                    xvec[subseq] = 1.
            elif task=='readm':
                for xvec, subseq in zip(onehot[:,idx,:], seq): 
                    xvec[subseq] = 1.
            elif task=='mortality':
                for xvec, subseq in zip(onehot[:,idx,:], seq): 
                    xvec[subseq] = 1.
    else:
        new_seqs.append(new_p)
        onehot = np.zeros((1, args.batchSize, vocabSize))
        for idx, seq in enumerate(seqs):
            for xvec, subseq in zip(onehot[:,idx,:], seq): 
                xvec[subseq] = 1.
    return torch.LongTensor(new_seqs), torch.FloatTensor(onehot)

    
