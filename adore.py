

import argparse, random, time, pickle, math
import numpy as np
import copy

from scipy.special import softmax
from scipy.stats import entropy
from sklearn import metrics

from sklearn.manifold import TSNE
import torch
import torch.nn.functional as F
from operator import mul

import pandas as pd
from model import models

#from model import att_fusion_models 


from utils import data_loader
from utils import data_helper

global ctd_d
ctd_d=False
#task='readm'
task='nextdx'
#task='mortality'
#use_data='eicu'
use_data='mimic'
use_ontology=True
auto_split=True
use_icd=True
use_snomed=True
use_ctd=False


def get_readm_labels(visits,time_seqs=None):
    relabels=[]
    cc=0
    oc=0
    for i in range(len(visits)):
#        relabels.append([])
#        if len(visits)!=len(time_seqs):1/0
 #       if len(visits[i])!=len(time_seqs[i]):1/0

        oc+=1
        cc+=len(visits[i])
#        print(len(visits[i]))
#        for j in range(len(visits[i])):
        relabels.append([1]*len(visits[i]))
        relabels[-1][-1]=0
   # print(relabels)
  #  1/0

  #  print(time_seqs,relabels)
#    print(oc,cc,1.0*oc/cc)
 #   1/0
    return relabels

def calculate_vocabSize(file):
    sequences = pickle.load(open(file, 'rb'))
    codeDict = {}
    max_w=1
    for patient_title in sequences:
        for visit_word in patient_title:
            for icd_char in visit_word:
                if icd_char>max_w: 
                    max_w=icd_char+7
                codeDict[icd_char] = ''
#    print(max_w,len(codeDict))
 #   1/0
    #return len(codeDict)#change
    if use_data=='mimic':
        vsize=len(codeDict)
    elif use_data=='eicu':
        vsize=max_w
    print(file,vsize)
    return vsize
#    return len(codeDict)#change
#    print(sequences,max_w)
#    1/0
    return max_w

def get_rootCode(treeFile):
    tree = pickle.load(open(treeFile, 'rb'))
    #tree = pickle.load(open('../eicu_inputs/dx.level2.pk', 'rb'))

    
    #print(treeFile,tree)
    #1/0
    rootCode = list(tree.values())[0][1]
    #print(tree,rootCode)
    #1/0
    return rootCode
def build_tree(treeFile):
    treeMap = pickle.load(open(treeFile, 'rb'))
    ancestors = np.array(list(treeMap.values())) 
    ancSize = ancestors.shape[1]
    leaves = []
    for k in treeMap.keys():
        leaves.append([k] * ancSize)
    leaves = np.array(leaves)
#    leaves = torch.LongTensor(leaves)
#    ancestors = torch.LongTensor(ancestors)
    return leaves, ancestors
eicu_folder='../eicu_inputs/'
mimic_folder='../mimic_inputs/'
#mimic_folder='../mimic_inputs_withdie/'
#mimic_folder='../mimic_inputs_withtime/'
if use_data=='eicu':
    folder=eicu_folder
elif use_data=='mimic':
    folder=mimic_folder
parser = argparse.ArgumentParser()
parser.add_argument('--dxlabel', type=str, default=folder+'dx.types')
parser.add_argument('--druglabel', type=str, default=folder+'rx.types')
parser.add_argument('--dxSeqsFile', type=str, default=folder+'dx.seqs')
#parser.add_argument('--dxSeqsFile', type=str, default=folder+'dx.seq')
parser.add_argument('--drugSeqsFile', type=str, default=folder+'rx.seqs')
#parser.add_argument('--drugSeqsFile', type=str, default=folder+'rx.seq')
parser.add_argument('--dxtreeFile', type=str, default=folder+'dx')
parser.add_argument('--drugtreeFile', type=str, default=folder+'rx')
parser.add_argument('--relationFile', type=str, default=folder+'snomed')
parser.add_argument('--dpLabelFile', type=str, default=folder+'dp.labels')
parser.add_argument('--EHREmbDim', type=int, default=350)##change ! should be 400
parser.add_argument('--ontoEmbDim', type=int, default=350)
parser.add_argument('--ontoattnDim', type=int, default=100)
parser.add_argument('--ptattnDim', type=int, default=100)
parser.add_argument('--batchSize', type=int, default=100)
parser.add_argument('--topk', type=int, default=20)
parser.add_argument('--epochs', type=int, default=60)
parser.add_argument('--gpu_core', type=str, default='cuda:2')
parser.add_argument('--LR', type=float, default=1)
parser.add_argument('--use_gpu', action='store_true', default=True)
parser.add_argument('--seed', type=int, default=7)
parser.add_argument('--save', type=str, default='./outputs/dxrx/80.pt')


args = parser.parse_args()
args.dxVocabSize = calculate_vocabSize(args.dxSeqsFile)
args.drugVocabSize = calculate_vocabSize(args.drugSeqsFile)
args.dpLabelSize = calculate_vocabSize(args.dpLabelFile)
if use_ontology==True:
    args.dxnumAncestors = get_rootCode(args.dxtreeFile+'.level2.pk')-args.dxVocabSize+1
    #args.dxnumAncestors = get_rootCode(args.dxtreeFile+'.level2.pk')+1
    #print(args.dxnumAncestors)
    #1/0
    
    #args.drugnumAncestors = get_rootCode(args.drugtreeFile+'.level3.pk')-args.drugVocabSize+1
    args.drugnumAncestors =0 
    args.split_recodes={i+args.dxVocabSize:1 for i in range( args.dxnumAncestors)}
#ontology_data={(('dx','ctd','rx'):[[],[]]),(('dx','icd','extra_dx'):[[],[]]),(('rx','atc','extra_rx'):None),(('dx','snomed','extra'):None)}
#ontology_data={('dx','icd','extra_dx'):None}
#rels=[('dx','icd','extra_dx'),('rx','atc','extra_rx')]
rels=[('extra_dx','icd','dx')]
#ontology_data={ r:(torch.tensor([],dtype=torch.int32),torch.tensor([],dtype=torch.int32)) for r in rels}
def load_dgi_icd(ontology_data):
    dxLeavesList = []
    dxAncestorsList = []
    drugLeavesList = []
    drugAncestorsList = []

    for i in range(5, 1, -1): # An ICD9 diagnosis code can have at most five ancestors (including the artificial root) when using CCS multi-level grouper. 
        leaves, ancestors = build_tree(args.dxtreeFile+'.level'+str(i)+'.pk')
        dxLeavesList.append(leaves)
        dxAncestorsList.append(ancestors)
    '''
    for i in range(5, 0, -4): 
        leaves, ancestors = build_tree(args.drugtreeFile+'.level'+str(i)+'.pk')
        drugLeavesList.append(leaves)
        drugAncestorsList.append(ancestors)
    '''
    rid=('extra_dx','icd','dx')
    src_list=[]
    des_list=[]
    for i in range(len(dxLeavesList)):  
        for j in range(len(dxLeavesList[i])):  
         #   des_list.append(dxLeavesList[i][j][0])
          #  src_list.append(dxLeavesList[i][j][0])

            for k in range(1,len(dxLeavesList[i][j])):  
                #print(dxAncestorsList)
                if dxAncestorsList[i][j][k]==3508:continue
                des_list.append(dxLeavesList[i][j][k])
                src_list.append(dxAncestorsList[i][j][k])
               # src_list.append(dxLeavesList[i][j][k])
                #des_list.append(dxAncestorsList[i][j][k])


                #des_list.append(dxLeavesList[i][j][k])
                #src_list.append(dxLeavesList[i][j][k])
#    print(len(set(des_list)),max(des_list),min(des_list),args.dxnumAncestors)
 #   1/0
    ontology_data=(torch.tensor(src_list).to(args.device),torch.tensor(des_list).to(args.device))
#    print(src_list,max(src_list),args.dxnumAncestors,args.dxVocabSize)
 #   1/0
        

    return ontology_data


def load_dgi_CTD(ontology_data,dxlabels,rxlabels):
    rid=('extra_dx','icd','dx')
    dx_num=len(dxlabels)
    rx_num=len(rxlabels)
    ctd_rels=pickle.load(open('../CTD_rels.pk','rb'))
    print(ctd_rels,dxlabels,rxlabels)
    final_rels=[]
    x=None
    c=None
    for row in ctd_rels:
        if row[0] not in rxlabels:continue
        if row[1] not in dxlabels:continue
        if row[0]!=x:
            x=row[0]
            c=0
#        if c>15:continue
        final_rels.append([rxlabels[row[0]],dxlabels[row[1]]])
        c+=1
#    print(final_rels,len(final_rels))
    final_rels=np.array(final_rels)
#    print(len(list(set(final_rels[:,0]))),len(list(set(final_rels[:,1]))))
#    return final_rels[:,0],final_rels[:,1]

    #train_set, valid_set, test_set = data_loader.dxrx('./inputs/dx.seqs', './inputs/rx.seqs', './inputs/dp.labels')
    train_set, valid_set, test_set = data_loader.dxrx(folder+'dx.seqs', folder+'rx.seqs', folder+'dp.labels',folder+'die.labels',datasource=use_data)

    rxseqs = train_set[1]
    dxseqs = train_set[0]
    
    final_rels=np.array(count_ctd_freq(dxseqs[:int(len(dxseqs)*0.8)],rxseqs[:int(len(dxseqs)*0.8)],final_rels))
    ctd_rx,ctd_dx=final_rels[:,0],final_rels[:,1]
    rx_list=range(args.drugVocabSize)
    dx_list=range(args.dxVocabSize)
    src_list=np.concatenate((ctd_rx+args.dxVocabSize,ctd_dx))
    des_list=np.concatenate((ctd_dx,ctd_rx+args.dxVocabSize))
    ontology_data=(torch.tensor(src_list).to(args.device),torch.tensor(des_list).to(args.device))
    return ontology_data
def count_ctd_freq(dxseqs,rxseqs,final_rels):
    print(dxseqs[0],len(dxseqs))
    dx_rx_count={}
    for i in range(len(dxseqs)):
        for j in range(len(dxseqs[i])):
            for d in range(len(dxseqs[i][j])):
                if dxseqs[i][j][d] not in dx_rx_count:
                    dx_rx_count[dxseqs[i][j][d]]={}
                for r in range(len(rxseqs[i][j])):
                    if rxseqs[i][j][r] not in dx_rx_count[dxseqs[i][j][d]]:
                        dx_rx_count[dxseqs[i][j][d]][rxseqs[i][j][r]]=0
                    dx_rx_count[dxseqs[i][j][d]][rxseqs[i][j][r]]+=1
    print(dx_rx_count)
    new_final_rels=[]
    dn=0.0
    nn=0.0
    for i in range(len(final_rels)):
        dn+=1
        if final_rels[i][1] not in dx_rx_count:
#            print('d')
            continue
        if final_rels[i][0] not in dx_rx_count[final_rels[i][1]]:
 #           print('r')
            continue
        if dx_rx_count[final_rels[i][1]][final_rels[i][0]]<1:
            continue
        new_final_rels.append(final_rels[i])
        nn+=1 
        print(final_rels[i][1],final_rels[i][0],dx_rx_count[final_rels[i][1]][final_rels[i][0]],nn/dn,len(new_final_rels),len(final_rels))
#    1/0
    return new_final_rels
def dcmap_to_input(extra_num,rel_map):
    max_lv=max([len(it[1][0]) for it in rel_map.items()])
    ctd_dx_leaves_list=[]
    ctd_dx_ancesster_list=[]
    ctd_dx_rel_list=[]
    permute_list=[]
    for i in range(max_lv+1):
        ctd_dx_leaves_list.append([])
        ctd_dx_ancesster_list.append([])
        ctd_dx_rel_list.append([])
    
    for it in rel_map.items():
        lv=len(it[1][0])
#        a=[dx_rel_extra_entitys.index(i) for i in it[1][0]]
        ctd_dx_ancesster_list[lv].append([i for i in it[1][0]]+[extra_num])
        ctd_dx_rel_list[lv].append([i for i in it[1][1]]+[1])
        ctd_dx_leaves_list[lv].append([it[0]]*(lv+1))
    for i in range(max_lv+1):
        ctd_dx_ancesster_list[i]=torch.LongTensor(ctd_dx_ancesster_list[i])
        ctd_dx_leaves_list[i]=torch.LongTensor(ctd_dx_leaves_list[i])
        ctd_dx_rel_list[i]=torch.LongTensor(ctd_dx_rel_list[i])
    for l in ctd_dx_leaves_list:
        for i in l:
            permute_list.append(int(i[0]))
    permute_list.append(len(permute_list))
#    print(ctd_dx_ancesster_list[6])
    return ctd_dx_leaves_list,ctd_dx_ancesster_list,ctd_dx_rel_list,permute_list
def load_CTD(dxlabels,rxlabels):
    dx_num=len(dxlabels)
    rx_num=len(rxlabels)
    ctd_rels=pickle.load(open('../CTD_rels.pk','rb'))
    print(ctd_rels,dxlabels,rxlabels)
    final_rels=[]
    x=None
    c=None
    for row in ctd_rels:
        if row[0] not in rxlabels:continue
        if row[1] not in dxlabels:continue
        if row[0]!=x:
            x=row[0]
            c=0
#        if c>15:continue
        final_rels.append([rxlabels[row[0]],dxlabels[row[1]]])
        c+=1
#    print(final_rels,len(final_rels))
    final_rels=np.array(final_rels)
#    print(len(list(set(final_rels[:,0]))),len(list(set(final_rels[:,1]))))
#    return final_rels[:,0],final_rels[:,1]

    train_set, valid_set, test_set = data_loader.dxrx('./inputs/dx.seqs', './inputs/rx.seqs', './inputs/dp.labels',datasource=use_data)
    rxseqs = train_set[1]
    dxseqs = train_set[0]
    final_rels=np.array(count_ctd_freq(dxseqs[:int(len(dxseqs)*0.5)],rxseqs[:int(len(dxseqs)*0.5)],final_rels))
    ctd_rx,ctd_dx=final_rels[:,0],final_rels[:,1]
    dx_ctd_map={}
    rx_ctd_map={}
    
    for i in range(dx_num):
        dx_ctd_map[i]=[[],[]]
    for i in range(rx_num):
        rx_ctd_map[i]=[[],[]]
    for i in range(len(ctd_dx)):
        dx_ctd_map[ctd_dx[i]][0].append(ctd_rx[i])
        dx_ctd_map[ctd_dx[i]][1].append(0)
        rx_ctd_map[ctd_rx[i]][0].append(ctd_dx[i])
        rx_ctd_map[ctd_rx[i]][1].append(0)

    ctd_dx_leaves_list,ctd_dx_ancesster_list,ctd_dx_rel_list,ctd_dx_permute_list=dcmap_to_input(rx_num,dx_ctd_map)
    ctd_rx_leaves_list,ctd_rx_ancesster_list,ctd_rx_rel_list,ctd_rx_permute_list=dcmap_to_input(dx_num,rx_ctd_map)
    return ctd_dx,ctd_rx,ctd_dx_leaves_list,ctd_dx_ancesster_list,ctd_dx_rel_list,ctd_dx_permute_list,ctd_rx_leaves_list,ctd_rx_ancesster_list,ctd_rx_rel_list,ctd_rx_permute_list
    
    1/0
    '''
    umls_to_snomed={}
    mesh_to_umls={}
    rx_dx_rel=[]
    '''

dxlabels = pickle.load(open(folder+'dx.types', 'rb'))
rxlabels = pickle.load(open(folder+'rx.types', 'rb'))
#ctd_rx,ctd_dx=load_CTD(dxlabels,rxlabels)
#ctd_dx,ctd_rx,ctd_dx_leaves_list,ctd_dx_ancesster_list,ctd_dx_rel_list,ctd_dx_permute_list,ctd_rx_leaves_list,ctd_rx_ancesster_list,ctd_rx_rel_list,ctd_rx_permute_list=load_CTD(dxlabels,rxlabels)
#1/0

def load_snomed(fn,labels,dxseqs):
    newlabels={}
    if use_data=='eicu':
        for it in labels.items():
            ind=it[0].strip('D_').strip('.')
            ind=ind.replace('.','')
            newlabels[ind]=it[1]
        labels=newlabels
            

    f_name='preprocessed_snomed_full2/'
    dx_rel_f=open(f_name+fn)
    dx_rel_list=[]
    dx_list=[]
    rel_map={}
    st=set()
    dx_rel_extra_entitys=set()
    dx_relationships=set()
    dx_num=len(labels.keys())
    #dx_num=max(labels.values())
    for i in range(dx_num):
        rel_map[i]=[[],[],[]]
        
    for line in dx_rel_f.readlines():#d=2 mean both dx code

        h,t,r,d=line.split(' ')
        st.add(h)
        if d.strip('\n')!='2':
            dx_rel_extra_entitys.add(t)
        dx_relationships.add(r)
        #print(labels,h)
      #  print(labels[h])
        if h not in labels:continue
#        print(labels[h],dx_num)
        rel_map[labels[h]][0].append(t)
        rel_map[labels[h]][1].append(r)
        rel_map[labels[h]][2].append(d.strip('\n'))
#        print(h,t,r)
        dx_rel_list.append([h,t,r])
#    print(st.intersection(dxlabels),len(list(st.intersection(dxlabels))))
 #   print(len(dxlabels.keys()))
#    print(rel_map)
#    print(len(dx_rel_list))

    dx_rel_extra_entitys=list(dx_rel_extra_entitys)
    dx_relationships=list(dx_relationships)
    ##testing
    extra_to_dx_map={}
    snomed_to_name={}
    extra_f=open('./preprocessed_snomed_full2/snomedId_to_names.txt')
    for line in extra_f.readlines():  
        ls=line.split(' ')
        snomed_to_name[ls[0]]=' '.join(ls[1:]).strip()
#    print(snomed_to_name)

    dx_rel_f=open(f_name+fn)
    for line in dx_rel_f.readlines():  
    #    print(line)
        h,t,r,d=line.split(' ')
        if t not in extra_to_dx_map:
            extra_to_dx_map[t]=[[],[],[]]
        extra_to_dx_map[t][0].append(h)
        extra_to_dx_map[t][1].append(r)
        extra_to_dx_map[t][2].append(d.strip())
#    print(extra_to_dx_map,len(extra_to_dx_map))
    coocur_graph=np.zeros(shape=(dx_num,dx_num))
    freq=np.zeros(shape=(dx_num))
    for p in range(len(dxseqs)):
        for v in range(len(dxseqs[p])):
            for i in range(len(dxseqs[p][v])):
                freq[dxseqs[p][v][i]]+=1
                for j in range(i+1,len(dxseqs[p][v])):
                    coocur_graph[dxseqs[p][v][i]][dxseqs[p][v][j]]+=1
    #print(coocur_graph)
    print(freq)
    tc=0.0
    icd_match=0.0
    #dxlabels = pickle.load(open('./inputs/dx.types', 'rb'))
    dxlabels = labels
    for it in extra_to_dx_map.items():##just for printing
        
        #if it[0] in snomed_to_name:continue
        for i in range(len(it[1][0])):
            if it[1][2][i]=='2':continue
            print(snomed_to_name[it[0]],it[1][0][i],snomed_to_name[it[1][1][i]])
            #print(it)

            
#        print('\n')
#    print(dxname_to_id)
#    print(dxlabels)
   # print(dxseqs)
   # avgit=0.0
   # for it in extra_to_dx_map.items():
   #     print(len(it[1][0]))
    #    avgit+=len(it[1][0])
   # print(avgit/len(extra_to_dx_map.items()))
        
    ###
    '''
    return extra_to_dx_map,snomed_to_name
    print(icd_match/tc)
    1/0
    '''

    max_lv=max([len(it[1][0]) for it in rel_map.items()])
    rel_leaves_list=[]
    rel_ancesster_list=[]
    rel_rel_list=[]
    '''
    for i in range(max_lv+1):
        rel_leaves_list.append([])
        rel_ancesster_list.append([])
        rel_rel_list.append([])
    '''
    # I add dummy extra entity so lv+1!! 
    '''
    for it in rel_map.items():
        lv=len(it[1][0])
#        a=[dx_rel_extra_entitys.index(i) for i in it[1][0]]
        rel_ancesster_list.extend([dx_rel_extra_entitys.index(i) for i in it[1][0]]+[len(list(dx_rel_extra_entitys))])
        rel_rel_list.extend([dx_relationships.index(i) for i in it[1][1]]+[len(list(dx_relationships))])
        rel_leaves_list.extend([it[0]]*(lv+1))
#    for i in range(max_lv):
 #       print(rel_ancesster_list[i]) 
  #  1/0
    '''
    rel_leaves_list,rel_ancesster_list,rel_rel_list,dxdirectionList,dxrelationnumAncestors,dxrelationnum=[],[],[],[],0,0
    for it in rel_map.items():
        lv=len(it[1][0])
        rel_leaves_list.append([it[0]]*(lv+1))
        rel_rel_list.append([dx_relationships.index(i) for i in it[1][1]]+[len(list(dx_relationships))])
        dxdirectionList.append([i for i in it[1][2]]+[0])
        rel_ancesster_list.append([])
        for i in range(lv):
            if it[1][2][i]=='2':
                rel_ancesster_list[-1].append(dxlabels[it[1][0][i]])
            else:

                rel_ancesster_list[-1].append(dx_rel_extra_entitys.index(it[1][0][i])+args.dxVocabSize )
        #rel_ancesster_list[-1].append(len(list(dx_rel_extra_entitys)))
        rel_ancesster_list[-1].append(it[0])#self loop
    permute_list=[]
    for l in rel_leaves_list:
        permute_list.append(l[0])
    permute_list.append(len(permute_list))
    #for i in range(max_lv+1):
    for i in range(len(rel_leaves_list)):
        rel_ancesster_list[i]=torch.LongTensor(rel_ancesster_list[i])
        rel_leaves_list[i]=torch.LongTensor(rel_leaves_list[i])
        rel_rel_list[i]=torch.LongTensor(rel_rel_list[i])
    
    print('per',permute_list,len(permute_list))

    return rel_leaves_list,rel_ancesster_list,rel_rel_list,list(permute_list)    ,len(list(dx_rel_extra_entitys))+1,len(list(dx_relationships))+1,dx_rel_extra_entitys,dx_relationships,snomed_to_name


def load_dgi_snomed(ontology_data,dxlabels,dxseqs):
    dxlabels = pickle.load(open(folder+'dx.types', 'rb'))
    newlabels={}
    if use_data=='eicu':
        for it in dxlabels.items():
            ind=it[0].strip('D_').strip('.')
            ind=ind.replace('.','')
            newlabels[ind]=it[1]
        dxlabels=newlabels
    #rxlabels = pickle.load(open('./inputs/rx.types', 'rb'))
    f_name='preprocessed_snomed_full2/'
    dx_rel_f=open(f_name+'Diagnosis_relations.txt')
    dx_rel_list=[]
    dx_list=[]
    rel_map={}
    st=set()
    dx_rel_extra_entitys=set()
    dx_relationships=set()
    dx_num=len(dxlabels.keys())+1
    #dx_num=max(labels.values())
    for i in range(dx_num):
        rel_map[i]=[[],[],[]]

    count=0
    for line in dx_rel_f.readlines():
        if count>9000:break
        count+=1

        h,t,r,d=line.strip('\n').split(' ')
        st.add(h)
        if d!='2':
            dx_rel_extra_entitys.add(t)

        dx_relationships.add(r)
        #print(labels,h)
      #  print(labels[h])
        if h not in dxlabels:continue
#        print(labels[h],dx_num)
        rel_map[dxlabels[h]][0].append(t)
        rel_map[dxlabels[h]][1].append(r)
        rel_map[dxlabels[h]][2].append(d)
#        print(h,t,r)
        dx_rel_list.append([h,t,r])
#    print(st.intersection(dxlabels),len(list(st.intersection(dxlabels))))
 #   print(len(dxlabels.keys()))
#    print(rel_map)
#    print(len(dx_rel_list))
 #   1/0

    dx_rel_extra_entitys=list(dx_rel_extra_entitys)
    dx_relationships=list(dx_relationships)
    ##testing
    extra_to_dx_map={}
    snomed_to_name={}
    extra_f=open('./preprocessed_snomed_full2/snomedId_to_names.txt')
    for line in extra_f.readlines():  
        ls=line.split(' ')
        snomed_to_name[ls[0]]=' '.join(ls[1:]).strip()

    dxrelationLeavesList,dxrelationAncestorsList,dxrelationList,dxdirectionList,dxrelationnumAncestors,dxrelationnum=[],[],[],[],0,0
    for it in rel_map.items():
        lv=len(it[1][0])
        dxrelationLeavesList.extend([it[0]]*(lv+1))
        dxrelationList.extend([dx_relationships.index(i) for i in it[1][1]]+[len(list(dx_relationships))])
        dxdirectionList.extend([i for i in it[1][2]]+[0])
        for i in range(lv):
            if it[1][2][i]=='2':
                dxrelationAncestorsList.append(dxlabels[it[1][0][i]])
            else:

                dxrelationAncestorsList.append(dx_rel_extra_entitys.index(it[1][0][i])+args.dxVocabSize )
        dxrelationAncestorsList.append(len(list(dx_rel_extra_entitys)))
    ###################################################

#    dxrelationLeavesList,dxrelationAncestorsList,dxrelationList,dxrelation_permuteindex,dxrelationnumAncestors,dxrelationnum,dx_rel_extra_entitys,snomed_to_name=load_snomed('Diagnosis_relations.txt',dxlabels,dxseqs)
 #   for i in range(len(dxrelationLeavesList)):
#        dxrelationAncestorsList[i]+=args.dxVocabSize
#    print(dxrelationAncestorsList)
 #   1/0
 #   print(len(dxrelationAncestorsList),len(dxrelationLeavesList),len(dxrelationList))
#    1/0
    dxrelationnumAncestors,dxrelationnum=len(list(dx_rel_extra_entitys))+1,len(list(dx_relationships))+1
    ontology_data=(torch.tensor(dxrelationAncestorsList).to(args.device),torch.tensor(dxrelationLeavesList).to(args.device))
        
    etypes=torch.tensor(dxrelationList).to(args.device)
    args.snomed_rel_num=dxrelationnum
    print(dxrelationAncestorsList,dxrelationLeavesList,dxrelationList,len(dxrelationLeavesList),len(dxrelationList))
    #1/0

    return ontology_data,etypes,dx_rel_extra_entitys,dxdirectionList,snomed_to_name,dxrelationnumAncestors

def load_snomed_paths(extra_to_dx_map,extra_to_rx_map,snomed_to_name):
    f_name='preprocessed_snomed/Snomed_entity_relations.txt'
    rel_f=open(f_name)
    rel_list=[]
    
    dxext=[it[0] for it in extra_to_dx_map.items()]
    rxext=[it[0] for it in extra_to_rx_map.items()]
    ext_to_ext={}
#    print(set(dxext).intersection(rxext))
    hset={}
    for line in rel_f.readlines():  
    #    print(line)
        h,t,r,_=line.split(' ')
        if h not in hset:
            hset[h]=set()
        else:
            if t in hset[h]:continue
        hset[h].add(t)
#        if h not in extra_to_dx_map and t not in extra_to_rx_map:continue
 #       if h not in extra_to_dx_map and t not in extra_to_rx_map:continue
        
#        print(h,t,r)
        rel_list.append([h,t,r])
    for rel  in rel_list:  
    #    print(line)
        h,t,r=rel
        if h not in ext_to_ext:
            ext_to_ext[h]=[[],[]]
        ext_to_ext[h][0].append(t)
        ext_to_ext[h][1].append(r)
        if t not in ext_to_ext:
            ext_to_ext[t]=[[],[]]
        ext_to_ext[t][0].append(h)
        ext_to_ext[t][1].append(r)
    print(ext_to_ext)
    cc=0
    for it in extra_to_dx_map.items():
        if it[0] not in ext_to_ext:continue
        for i in range(len(ext_to_ext[it[0]][0])):
            for j in range(len(ext_to_ext[ext_to_ext[it[0]][0][i]][0] )):
                #print(snomed_to_name[it[0]],snomed_to_name[ext_to_ext[it[0]][0][i]])
            #if ext_to_ext[it[0]][0][i] in extra_to_rx_map:
                if  ext_to_ext[ext_to_ext[it[0]][0][i]][0][j] in extra_to_rx_map:
                    print(it[1][0][0],snomed_to_name[it[1][1][0]],snomed_to_name[it[0]],snomed_to_name[ext_to_ext[it[0]][1][i]],snomed_to_name[ext_to_ext[it[0]][0][i]],snomed_to_name[ext_to_ext[ext_to_ext[it[0]][0][i]][1][j]],snomed_to_name[ext_to_ext[ext_to_ext[it[0]][0][i]][0][j]],snomed_to_name[extra_to_rx_map[ext_to_ext[ext_to_ext[it[0]][0][i]][0][j]][1][0]],extra_to_rx_map[ext_to_ext[ext_to_ext[it[0]][0][i]][0][j]][0][0])
                    print('\n')
                    cc+=1
    print(cc,len(ext_to_ext))
    1/0
#dxlabels = pickle.load(open('./inputs/dx.types', 'rb'))
#dxlabels = pickle.load(open('./inputs/dx.types', 'rb'))
#rxlabels = pickle.load(open('./inputs/rx.types', 'rb'))
#ctd_rx,ctd_dx=load_CTD(dxlabels,rxlabels)
#1/0
train_set, valid_set, test_set = data_loader.dxrx(folder+'dx.seqs', folder+'rx.seqs', folder+'dp.labels', folder+'die.labels',datasource=use_data)

train_set=data_loader.filter_repeat_visit(train_set)
valid_set=data_loader.filter_repeat_visit(valid_set)
test_set=data_loader.filter_repeat_visit(test_set)

rxseqs = train_set[1]
dxseqs = train_set[0]







args.allDxs = [dx for dx in range(args.dxVocabSize)][1:]
args.use_cuda = torch.cuda.is_available() and args.use_gpu
#args.use_cuda = False
args.device = torch.device(args.gpu_core if args.use_cuda else 'cpu')

torch.cuda.manual_seed_all(args.seed)

dxlabels = pickle.load(open(args.dxlabel, 'rb'))
#print(dxlabels)
#1/0
druglabels = pickle.load(open(args.druglabel, 'rb'))

G={}
ontology_data=[]
dx_to_ancestor_matrix=np.zeros(shape=(args.dxVocabSize,args.dxnumAncestors))
if use_ontology==True:
    if auto_split==False:
        args.split_max_num=2

    else:
        args.split_max_num=4
    if use_icd==True:
        
        ontology_data.append(load_dgi_icd(ontology_data))
        G['icd']=dgl.graph(ontology_data[0],num_nodes=args.dxVocabSize+args.dxnumAncestors*args.split_max_num)
    ans=ontology_data[0][0].detach()
    les=ontology_data[0][1].detach()
    for i in range(len(ans)):
        dx_to_ancestor_matrix[les[i].detach()][ans[i].detach()-args.dxVocabSize]=1.0
    dx_to_ancestor_matrix=torch.Tensor(dx_to_ancestor_matrix).to(args.device)
    
    dxrelationLeavesList,dxrelationAncestorsList,dxrelationList,dxrelation_permuteindex,dxrelationnumAncestors,dxrelationnum,dx_rel_extra_entitys,snomed_to_name=[],[],[],[],0,0,[],[]

    drugrelationLeavesList,drugrelationAncestorsList,drugrelationList,drugrelation_permuteindex,drugrelationnumAncestors,drugrelationnum=0,0,0,0,0,0
    if use_snomed==True:

        dxrelationLeavesList,dxrelationAncestorsList,dxrelationList,dxrelation_permuteindex,dxrelationnumAncestors,dxrelationnum,dx_rel_extra_entitys,dx_relationships,snomed_to_name=load_snomed('Diagnosis_relations.txt',dxlabels,dxseqs)
        args.dxrelationnum =dxrelationnum 
        args.dxrelationnumAncestors =dxrelationnumAncestors
        ##for dgi layer
        '''
        ondata,etypes,dx_rel_extra_entitys,dxdirectionList,snomed_to_name,snomed_ans_num=load_dgi_snomed(ontology_data,dxlabels,dxseqs)

        args.snomed_dx_ans_num=snomed_ans_num
        ontology_data.append(ondata)
        G['snomed']=[dgl.graph(ontology_data[-1],num_nodes=args.dxVocabSize+args.snomed_dx_ans_num),etypes]
        '''
    if use_ctd==True:
        ontology_data.append(load_dgi_CTD(ontology_data,dxlabels,rxlabels))
        G['ctd']=dgl.graph(ontology_data[-1],num_nodes=args.dxVocabSize+args.drugVocabSize)

    #G=dgl.graph(ontology_data[('extra_dx','icd','dx')],num_nodes=args.dxVocabSize+args.dxnumAncestors+args.dxnumAncestors)
    #print(args.dxVocabSize,args.dxnumAncestors)
    #1/0
    #print(args.dxVocabSize+args.dxnumAncestors*args.split_max_num)
    #1/0
    #G.append(dgl.graph(ontology_data[0],num_nodes=args.dxVocabSize+args.dxnumAncestors))
    '''
    G.append(dgl.graph(ontology_data[1],num_nodes=args.dxVocabSize+args.drugVocabSize))
    '''
    
    #print(ontology_data[2][0].size(),ontology_data[2][1].size(),etypes.size())
    #1/0
    for it in G.items():
    #    if i==2:continue##snomed
        if it[0]=='snomed':
            continue
            #G[it[0]][0]=dgl.add_self_loop(G[it[0]][0])
        else:
            G[it[0]]=dgl.add_self_loop(G[it[0]])##add loop for icd or cts
    for it in G.items():
    #for g in range(len(G)):
        if it[0]=='snomed':
            G[it[0]][0]=G[it[0]][0].to(args.device)
        else:
            G[it[0]]=G[it[0]].to(args.device)
#        G[g]=G[g].to(args.device)

    '''
    G[2]=[G[2],etypes]##snomed input
    '''
    #G[i]=dgl.to_bidirected(G[i])

#print(G[2][0].num_edges())
#1/0


#print(len(dxlabels.items()),len(druglabels.items()))
#1/0
# ##############################################################################
# Load data
################################################################################

#train_set, valid_set, test_set = data_loader.dxrx(args.dxSeqsFile, args.drugSeqsFile, args.dpLabelFile,mimic_folder+'patients.dates')
#train_set, valid_set, test_set = data_loader.dxrx(args.dxSeqsFile, args.drugSeqsFile, args.dpLabelFile)
dxLeavesList = []
dxAncestorsList = []
drugLeavesList = []
drugAncestorsList = []

#t=pickle.load(open(args.dxSeqsFile, 'rb'))
#print(np.array(dxLeavesList[1]).shape,np.array(relationLeavesList[1]).shape)
#1/0
# ##############################################################################
# Build model
# ##############################################################################

#mmore_model = models.MMORE(args)## medical graph version!!
#mloss=torch.nn.MSELoss()

#mloss=mloss.to(mmore_model.device)

#mmore_model = models.MMORE_GAT(args)## medical graph version!!
#rel_num=1

#mmore_model = models.DGI_model(args,G,rels,use_ontology=use_ontology,use_icd=use_icd,use_snomed=use_snomed,use_ctd=use_ctd)## medical graph version!!
mmore_model = models.two_layer_model(args,G,dx_to_ancestor_matrix,rels,use_ontology=use_ontology,use_icd=use_icd,use_snomed=use_snomed,use_ctd=use_ctd)## medical graph version!!
#mmore_model = att_fusion_models.DGI_model(args,G,rels,use_ontology=use_ontology,use_icd=use_icd,use_snomed=use_snomed,use_ctd=use_ctd)## medical graph version!!
#print(etypes)
#print(etypes.size())

print('mmore_model:', mmore_model)

#print(mmore_model.parameters())
#for name, param in mmore_model.named_parameters():
 #   if param.requires_grad:
  #      print(name)
#1/0
optimizer = torch.optim.Adadelta([
    {'params': mmore_model.parameters()}
    ], 
    lr=args.LR, rho=0.95, weight_decay=0)
'''
optimizer = torch.optim.Adam([
    {'params': mmore_model.parameters()}
    ], 
    lr=args.LR)
'''
def get_criterion():
    return torch.nn.BCELoss(reduction='sum')

crit = get_criterion()

if args.use_cuda:
    mmore_model = mmore_model.to(args.device)
    crit = crit.to(args.device)
'''
ctd_dx=torch.LongTensor(ctd_dx)
ctd_rx=torch.LongTensor(ctd_rx)
ctd_dx=ctd_dx.to(mmore_model.device)
ctd_rx=ctd_rx.to(mmore_model.device)
'''
#    print(ctd_rx,ctd_dx)
#ctd_loss=100*mloss(mmore_model.ctd_W(mmore_model.EHRdxEmb(ctd_dx)),mmore_model.EHRdrugEmb(ctd_rx))
#ctd_loss=mmore_model.ctd_loss(ctd_dx,ctd_rx)
#ctd_loss=1*torch.sum((mmore_model.ctd_W(mmore_model.EHRdxEmb(ctd_dx))-mmore_model.ctd_W(mmore_model.EHRdrugEmb(ctd_rx)))**2)
#ctd_loss=1*torch.sum(mmore_model.EHRdxEmb(ctd_dx)-mmore_model.EHRdrugEmb(ctd_rx))
# ##############################################################################
# Training
# ##############################################################################

train_loss = []
valid_loss = []
test_loss = []
def change_edges(graph,split_recodes,add_all=False):
    anc_to_leave_atts={}
    anc_to_leave_id={}
    anc_to_leave_mean={}
    icd_edges_attentions=graph.edata['a'].cpu().detach().numpy()
    edges=[[it.item() for it in graph.edges()[0].cpu()],[it.item() for it in graph.edges()[1].cpu()]]
    for i in range(len(edges[0])):
        if edges[0][i]==edges[1][i]:continue
        if edges[0][i]<args.dxVocabSize:continue
        if edges[0][i]>=args.dxVocabSize+args.dxnumAncestors:continue
        if edges[0][i] not in anc_to_leave_atts:
            anc_to_leave_atts[edges[0][i]]=[]
            anc_to_leave_id[edges[0][i]]=[]
        anc_to_leave_atts[edges[0][i]].append(icd_edges_attentions[i][0][0])
        anc_to_leave_id[edges[0][i]].append(edges[1][i])
    for it in anc_to_leave_atts.items():
#        anc_to_leave_mean[it[0]]=np.mean(it[1])
        anc_to_leave_mean[it[0]]=entropy(softmax(it[1]),base=2)
#    print(len(anc_to_leave_mean))
 #   1/0
    src_list=[]
    des_list=[]
    split_anc=[]

    #for it in anc_to_leave_mean.items():
        #if it[1]>0.05:continue
    for it in anc_to_leave_atts.items():
        if add_all==False and max(it[1])<0.8:continue
        if split_recodes[it[0]]>=args.split_max_num:continue
        split_anc.append(it[1])
        for  l in range(len(anc_to_leave_id[it[0]])):
            src_list.append(args.dxnumAncestors*split_recodes[it[0]]+it[0])
            des_list.append(anc_to_leave_id[it[0]][l])
            #des_list.append(args.dxnumAncestors*split_recodes[it[0]]+it[0])
            #src_list.append(anc_to_leave_id[it[0]][l])
        split_recodes[it[0]]=int(split_recodes[it[0]]+int(1))
    if len(src_list)!=0:
        graph.add_edges(torch.tensor(src_list).to(args.device),torch.tensor(des_list).to(args.device))
#    print(anc_to_leave_atts)
    print('add_edges',args.dxnumAncestors,len(split_anc))
  #  1/0

#            print(self.G[0].edges(),self.icd_edges_attentions.shape)
def get_dp_acc_train(args, crit, preds, targets):
    loss = crit(preds, targets)
    return loss

def get_dp_acc(args, crit, preds, targets):
    loss = crit(preds, targets)
    correct_dx_num = total_dx_num = 0
    patient_num = preds.size()[0]
    visit_num = preds.size()[1]
    dpLabelSize = preds.size()[2]
    preds = preds.view(patient_num*visit_num, -1)
    targets = targets.view(patient_num*visit_num, -1)
    pred_topk, pred_idx = torch.topk(preds, k=args.topk, dim=1)
    for v_pred_idx, v_tgt in zip(pred_idx, targets):
        v_tgts_idx = torch.nonzero(v_tgt)
        if list(v_tgts_idx.size()):
                total_dx_num += list(v_tgts_idx.size())[0]
        for idx in v_pred_idx:
            if idx in v_tgts_idx:
                correct_dx_num += 1
    return loss, correct_dx_num, total_dx_num

def evaluate(args, dataSet):
    mmore_model.eval()
    total_loss = patient_num = total_dxnum = correct_dxnum =pos_corr=neg_corr=true_pos=true_neg= 1
    batch_num = int(np.ceil(float(len(dataSet[0])) / float(args.batchSize))) - 1
    tauc=0.0
    Ys=[]
    preYs=[]
    moYs=[]
    mopreYs=[]
    for bidx in random.sample(range(batch_num), batch_num):
    #for bidx in range(batch_num):
        ##3 is time
        patient_num += args.batchSize
        dxseqs = dataSet[0][bidx*args.batchSize:(bidx+1)*args.batchSize]
        drugseqs = dataSet[1][bidx*args.batchSize:(bidx+1)*args.batchSize]
        dplabels = dataSet[2][bidx*args.batchSize:(bidx+1)*args.batchSize]
        molabels = dataSet[4][bidx*args.batchSize:(bidx+1)*args.batchSize]
        relabels = dataSet[5][bidx*args.batchSize:(bidx+1)*args.batchSize]
        #print(relabels)
        #1/0
  #      print(drugseqs[:8])
        dxseqs, dx_onehot = data_helper.get_seqs(dxseqs, args, codetype='dx',task=task)
        drugseqs, drug_onehot = data_helper.get_seqs(drugseqs, args, codetype='drug',task=task)
   #     inputs = (dxseqs, drugseqs, dx_onehot, drug_onehot,
    #        dxLeavesList, dxAncestorsList, drugLeavesList, drugAncestorsList,dxrelationLeavesList,dxrelationAncestorsList,dxrelationList,dxrelation_permuteindex,drugrelationLeavesList,drugrelationAncestorsList,drugrelationList,drugrelation_permuteindex)
        #inputs = (dxseqs, drugseqs, dx_onehot, drug_onehot,dxLeavesList, dxAncestorsList, drugLeavesList, drugAncestorsList)
#        inputs = (dxseqs, drugseqs, dx_onehot, drug_onehot,dxLeavesList, dxAncestorsList, drugLeavesList, drugAncestorsList,ctd_dx_leaves_list,ctd_dx_ancesster_list,ctd_dx_rel_list,ctd_dx_permute_list,ctd_rx_leaves_list,ctd_rx_ancesster_list,ctd_rx_rel_list,ctd_rx_permute_list)
     #   inputs = (dxseqs, drugseqs, dx_onehot, drug_onehot)
        inputs = (dxseqs, drugseqs, dx_onehot, drug_onehot,
            dxLeavesList, dxAncestorsList, drugLeavesList, drugAncestorsList,dxrelationLeavesList,dxrelationAncestorsList,dxrelationList,dxrelation_permuteindex,drugrelationLeavesList,drugrelationAncestorsList,drugrelationList,drugrelation_permuteindex)
        mort_result,read_result,dp_result, cooccur_loss = mmore_model(inputs)
        labels_dp, dp_mask = data_helper.get_dp_mask(dplabels, args.dpLabelSize)
        labels_re, re_mask = data_helper.get_re_mask(relabels)
        labels_mo, mo_mask = data_helper.get_mo_mask(molabels)
       # pred_dp = torch.mul(dp_result, dp_mask.to(args.device))

        auc=0.0
        moauc=0.0
        if task=='nextdx':
            pred_dp = torch.mul(dp_result, dp_mask.to(args.device))
            pred_loss, batch_correct_dxnum, batch_total_dxnum = get_dp_acc(args, crit, pred_dp, labels_dp.to(args.device))
        elif task=='readm':
            pred_re = torch.mul(read_result, re_mask.to(args.device))
#            print(drugseqs[:5])
 #           1/0
#            pred_dp = torch.mul(dp_result, dp_mask.to(args.device))
            
            pred_loss, batch_correct_dxnum, batch_total_dxnum,Y,preY,bat_pos_corr,bat_true_pos,bat_neg_corr,bat_true_neg= data_helper.get_re_acc( crit, pred_re, labels_re.to(args.device),re_mask)
            Ys.extend(Y)
            preYs.extend(preY)
            
            pos_corr += bat_pos_corr
            neg_corr += bat_neg_corr
            true_pos += bat_true_pos
            true_neg += bat_true_neg
        elif task=='mortality':
            pred_mo = torch.mul(mort_result, mo_mask.to(args.device))
            pred_loss, batch_correct_dxnum, batch_total_dxnum,Y,preY,bat_pos_corr,bat_true_pos,bat_neg_corr,bat_true_neg= data_helper.get_mo_acc( crit, pred_mo, labels_mo.to(args.device),mo_mask)
            moYs.extend(Y)
            mopreYs.extend(preY)
#            print(np.array(re_mask)[:3],'\n',pred_re[:3],'\n',labels_re[:3],'\n',Y[:10],'\n',preY[:10])

    





#        pred_loss, batch_correct_dxnum, batch_total_dxnum = get_dp_acc(args, crit, pred_dp, labels_dp.to(args.device))
        total_dxnum += batch_total_dxnum
        correct_dxnum += batch_correct_dxnum
        batch_loss = pred_loss.add(cooccur_loss)
        total_loss += batch_loss.item()
        
    if task=='readm':
#        print(relabels,Ys,len(Ys),sum(Ys))
 #       1/0
   #     1/0
        fpr, tpr, thresholds = metrics.roc_curve(Ys, preYs, pos_label=1)
        auc=metrics.auc(fpr, tpr)
    if task=='mortality':
        fpr, tpr, thresholds = metrics.roc_curve(moYs, mopreYs, pos_label=1)
        moauc=metrics.auc(fpr, tpr)
#        preYs=np.random.uniform(0,1,len(preYs))
        
 #       precision,recall,  _ = metrics.precision_recall_curve(Ys, preYs,pos_label=1)
  #      auc=metrics.auc(recall, precision)
    #print(np.array(re_mask)[:3],'\n',pred_re[:3],'\n',labels_re[:3],'\n',Ys[:10],'\n',preYs[:10])
    #print('tr',pos_corr/true_pos,'nr',neg_corr/true_neg)

    return total_loss/patient_num, correct_dxnum, total_dxnum, correct_dxnum/total_dxnum,auc,moauc
    
#ctd_W = torch.nn.Linear(mmore_model.EHREmbDim, mmore_model.EHREmbDim)
#ctd_W2 = torch.nn.Linear(mmore_model.EHREmbDim, mmore_model.EHREmbDim)
#ctd_W=ctd_W.to(mmore_model.device)
#ctd_W2=ctd_W2.to(mmore_model.device)
def train(args, dataSet,epoch=0):
#    print(ctd_loss)
 #   1/0

    #dxlabels = pickle.load(open('./inputs/dx.types', 'rb'))
    #rxlabels = pickle.load(open('./inputs/rx.types', 'rb'))
#    ctd_rx=[[x] for x in ctd_rx]
 #   ctd_dx=[[x] for x in ctd_dx]
#    print(ctd_rx,ctd_dx)
    mmore_model.train()
    total_loss = patient_num = 0
    batch_num = int(np.ceil(float(len(dataSet[0])) / float(args.batchSize))) - 1
    '''
    global ctd_d
    if ctd_d==False:
        for i in range(30):
            ctd_loss=mmore_model.ctd_loss(ctd_dx,ctd_rx)*1
            optimizer.zero_grad()
            l=ctd_loss.backward( retain_graph=True)
            #print('ll',l)
            optimizer.step()
            print(ctd_loss)
        ctd_d=True
    '''

     
    #ontoloss=mmore_model.onto_loss(ontology_data[0][0],ontology_data[0][1])
    '''
    print('fir')
    for v in dataSet[4]:
        if v[-2]==0:
            print(v)
            1/0
    '''
    for bidx in random.sample(range(batch_num), batch_num):
        patient_num += args.batchSize  
        dxseqs = dataSet[0][bidx*args.batchSize:(bidx+1)*args.batchSize]
        drugseqs = dataSet[1][bidx*args.batchSize:(bidx+1)*args.batchSize]
        dplabels = dataSet[2][bidx*args.batchSize:(bidx+1)*args.batchSize] 
        #print(dplabels[0])
        #print(dxseqs[0])
        #1/0
        molabels = dataSet[4][bidx*args.batchSize:(bidx+1)*args.batchSize]
        relabels = dataSet[5][bidx*args.batchSize:(bidx+1)*args.batchSize]
#        print(dxseqs)
 #       1/0
#        print()
       # print(dplabels,np.array(dplabels).shape)
       # print(dxseqs,np.array(dxseqs).shape)
#        print([len(x) for x in dxseqs[:115]],'\n',[len(x) for x in dplabels[:115]])
 #       1/0
       # print('tre0',len(dxseqs[0]))
    #    print(dxseqs[:5])
     #   1/0
        dxseqs, dx_onehot = data_helper.get_seqs(dxseqs, args, codetype='dx',task=task)
    #    print(dxseqs,np.array(dxseqs).shape)
    #    1/0
        drugseqs, drug_onehot = data_helper.get_seqs(drugseqs, args, codetype='drug',task=task)
        inputs = (dxseqs, drugseqs, dx_onehot, drug_onehot,
            dxLeavesList, dxAncestorsList, drugLeavesList, drugAncestorsList,dxrelationLeavesList,dxrelationAncestorsList,dxrelationList,dxrelation_permuteindex,drugrelationLeavesList,drugrelationAncestorsList,drugrelationList,drugrelation_permuteindex)
#        inputs = (dxseqs, drugseqs, dx_onehot, drug_onehot,dxLeavesList, dxAncestorsList, drugLeavesList, drugAncestorsList)

 #       inputs = (dxseqs, drugseqs, dx_onehot, drug_onehot,dxLeavesList, dxAncestorsList, drugLeavesList, drugAncestorsList,ctd_dx_leaves_list,ctd_dx_ancesster_list,ctd_dx_rel_list,ctd_dx_permute_list,ctd_rx_leaves_list,ctd_rx_ancesster_list,ctd_rx_rel_list,ctd_rx_permute_list)

#        inputs = (dxseqs, drugseqs, dx_onehot, drug_onehot)
        mort_result,read_result,dp_result, cooccur_loss = mmore_model(inputs)
        labels_dp, dp_mask = data_helper.get_dp_mask(dplabels, args.dpLabelSize)
        labels_re, re_mask = data_helper.get_re_mask(relabels)
        labels_mo, mo_mask = data_helper.get_mo_mask(molabels)
    #    print('tre',dxseqs.cpu().detach().numpy().shape,len(relabels[0]),re_mask.cpu().detach().numpy().shape,dp_mask.cpu().detach().numpy().shape)

  #      print('\n',dxseqs.cpu().detach().numpy().shape,dp_result.cpu().detach().numpy().shape,dp_mask.cpu().detach().numpy().shape)
 #       1/0
        #pred_dp = torch.mul(dp_result, dp_mask.to(args.device))
        #pred_loss = get_dp_acc_train(args, crit, pred_dp, labels_dp.to(args.device))

        if task=='nextdx':

            pred_dp = torch.mul(dp_result, dp_mask.to(args.device))
            pred_loss = get_dp_acc_train(args, crit, pred_dp, labels_dp.to(args.device))
            batch_loss = pred_loss.add(cooccur_loss)
        elif task=='readm':
        #    print(read_result.cpu().detach().numpy(),dxseqs.cpu().detach().numpy().shape,read_result.cpu().detach().numpy().shape,re_mask.cpu().detach().numpy().shape,labels_re.cpu().detach().numpy().shape)
         #   1/0
            #align_loss=mmore_model.align_loss()
            #print(re_mask.shape)
            #print(read_result.shape)
#            print(read_result,re_mask)
            pred_re = torch.mul(read_result, re_mask.to(args.device))
            #cc=torch.nn.BCELoss(weight=(labels_re.to(args.device)+1)**4,reduction='sum')
 #           cc=torch.nn.BCELoss(weight=(labels_re.to(args.device)+1)**2,reduction='sum')
            #re_loss = 1000*cc(torch.squeeze(pred_re),labels_re.to(args.device))
            re_loss = 100000*crit(torch.squeeze(pred_re),labels_re.to(args.device))
            #re_loss = 100000*torch.sum((torch.squeeze(pred_re)-labels_re.to(args.device))**2)
            #re_loss = 1000*torch.sum((torch.squeeze(pred_re)-labels_re.to(args.device))**2)
            batch_loss = re_loss.add(cooccur_loss)
        elif task=='mortality':
            pred_mo = torch.mul(mort_result, mo_mask.to(args.device))
            
            mo_loss = 100000*crit(torch.squeeze(pred_mo),labels_mo.to(args.device))
            batch_loss = mo_loss.add(cooccur_loss)

#        batch_loss = pred_loss.add(cooccur_loss)
 #       batch_loss = ctd_loss


#        ctd_loss=mmore_model.ctd_loss(ctd_dx,ctd_rx)*0.001
#        batch_loss = ctd_loss
#        batch_loss = batch_loss.add(ctd_loss)
        optimizer.zero_grad()
       # l=batch_loss.backward( )

        l=batch_loss.backward( retain_graph=True)
        #print('ll',l)
        optimizer.step()
        total_loss += batch_loss.item()
        #print(mmore_model.EHRdxEmb(ctd_dx))
    
    #print('ctd_loss',ctd_loss)
    #print('onto_loss',align_loss)

    if use_ontology==True and auto_split==True and epoch%1==0:
        change_edges(G['icd'],args.split_recodes)
        print(G['icd'])
    '''
    print('sec')
    for v in dataSet[4]:
        if v[-2]==0:
            print(v)
            1/0
    '''
    return total_loss/patient_num
    #return ctd_loss

# ##############################################################################
# Save Model
# ##############################################################################
best_valid_loss = None
best_test_acc = None
best_valid_acc = None
best_valid_auc = None
best_valid_moauc = None
total_start_time = time.time()
train_set=list(train_set)
test_set=list(test_set)
valid_set=list(valid_set)

try:
    
    
    train_set.append(get_readm_labels(train_set[0],train_set[3]))
    test_set.append(get_readm_labels(test_set[0],test_set[3]))
    valid_set.append(get_readm_labels(valid_set[0],valid_set[3]))

    print('tre',len(train_set[0][0]),len(train_set[3][0]))
    print('-' * 70)
    print('train,valid,test',np.array(train_set[0]).shape,np.array(valid_set).shape,np.array(test_set).shape )
    presetinput = (dxLeavesList, dxAncestorsList, drugLeavesList, drugAncestorsList,dxrelationLeavesList,dxrelationAncestorsList,dxrelationList,dxrelation_permuteindex,drugrelationLeavesList,drugrelationAncestorsList,drugrelationList,drugrelation_permuteindex)
    mmore_model.preset(presetinput)
    
    for epoch in range(1, args.epochs+1):
        # Train
        epoch_start_time = time.time()
        trainLoss = train(args=args, dataSet=train_set,epoch=epoch)
        train_loss.append(trainLoss)
        print('| epoch: {:3d} (train) | loss: {:.2f} | time: {:2.0f}s'.format(epoch, trainLoss, time.time() - epoch_start_time))
        print('-' * 70)
        if args.split_max_num==2 and epoch==1:##mmore
        #if epoch==1:##mmore
            change_edges(G['icd'],args.split_recodes,add_all=True)
        if epoch%2 == 0:
            # Validation
            print('train0')
            trainLoss, _, _, _,trainauc,trainmoauc = evaluate(args=args, dataSet=train_set)
            print('train',trainLoss,trainauc,trainmoauc)
            validLoss, correct_dx, total_dx, validdpacc,vaildauc,vaildmoauc = evaluate(args=args, dataSet=valid_set)
            valid_loss.append(validLoss)
            print('| epoch: {:3d} (valid) | loss: {:.2f} | DPACC: {:.3f}% ({}/{})|reauc: {:.4f}|moauc: {:.4f}'.format(epoch, validLoss, validdpacc*100, correct_dx, total_dx,vaildauc,vaildmoauc))
            print('-' * 70)
            # Test
            testLoss, correct_dx, total_dx, testdpacc,testauc,testmoauc  = evaluate(args=args, dataSet=test_set)
            test_loss.append(testLoss)
            print('| epoch: {:3d} (test)  | loss: {:.2f} | DPACC: {:.3f}% ({}/{})|reauc: {:.4f}|moauc: {:.4f}'.format(epoch, testLoss, testdpacc*100, correct_dx, total_dx,testauc,testmoauc))
            
            if task=='nextdx':
                best_mea=best_valid_acc
                mea=validdpacc
            elif task=='readm':
                best_mea=best_valid_auc
                mea=vaildauc
            elif task=='mortality':
                best_mea=best_valid_moauc
                mea=vaildmoauc
            print('-' * 70)
#            if not best_valid_acc or not best_valid_acc > validdpacc or not best_valid_auc > vaildauc :
            if not best_mea or not best_mea > mea :
                best_epoch_num = epoch
                best_valid_acc = validdpacc
                best_valid_auc = vaildauc
                best_valid_moauc = vaildmoauc
                best_test_acc = testdpacc
                best_test_auc = testauc
                model_state_dict = mmore_model.state_dict()
                model_source = {
                    "settings": args,
                    "model": model_state_dict,
                }
                #torch.save(model_source, args.save)
    
except KeyboardInterrupt:

    '''
    t=mmore_model.EHRdxEmb.cpu().weight.detach().numpy()
    t2=mmore_model.EHRdrugEmb.cpu().weight.detach().numpy()
    tx=np.concatenate((t,t2),axis=0)
    print('tx',tx.shape)
    tx = TSNE(n_components=2,n_iter=250).fit_transform(tx)
    t=tx[:len(t)]
    t2=tx[len(t):]
    print('emb',t,t.shape)
    
    np.save('ctd_emb/dx_emb',t)
    np.save('ctd_emb/rx_emb',t2)
    '''


    
#    print(dx_anc_types)
    print("-"*70)
    print("Exiting from training early | cost time: {:5.2f} min".format((time.time() - total_start_time)/60.0))
####save data
visit_attentions=[]
ontology_attention=[]
dataSet=train_set
batch_num = int(np.ceil(float(len(dataSet[0])) / float(args.batchSize))) - 1
for bidx  in range(batch_num):
    dxseqs = dataSet[0][bidx*args.batchSize:(bidx+1)*args.batchSize]
    drugseqs = dataSet[1][bidx*args.batchSize:(bidx+1)*args.batchSize]
    dplabels = dataSet[2][bidx*args.batchSize:(bidx+1)*args.batchSize]
    dxseqs, dx_onehot = data_helper.get_seqs(dxseqs, args, codetype='dx',task=task)
    drugseqs, drug_onehot = data_helper.get_seqs(drugseqs, args, codetype='drug',task=task)
#    inputs = (dxseqs, drugseqs, dx_onehot, drug_onehot)
    inputs = (dxseqs, drugseqs, dx_onehot, drug_onehot,
            dxLeavesList, dxAncestorsList, drugLeavesList, drugAncestorsList,dxrelationLeavesList,dxrelationAncestorsList,dxrelationList,dxrelation_permuteindex,drugrelationLeavesList,drugrelationAncestorsList,drugrelationList,drugrelation_permuteindex)
    mort_result,read_result,dp_result, cooccur_loss = mmore_model(inputs)
    visit_attentions.append(mmore_model.visit_attentions.cpu().detach().numpy())

dx_Y=dxlabels
#print(dx_Y)
#1/0
dx_embs=snomed_extra_embs=0
dx_embs=mmore_model.dxALLontoEmb.cpu().detach().numpy()
dxseqs = train_set[0]
icd_rels=dx_anc_types=0
snomed_rels=snomed_anc_types=snomed_rel_types=snomed_edge_types=0
icd_ontology_attention=np.array(mmore_model.icd_edges_attentions)
if use_icd==True:
    #dx_anc_types = pickle.load(open(folder+'dx.anctypes', 'rb'))
    dx_anc_types = {}
#    print(ontology_attention,dx_embs)
 #   1/0
    icd_rels=mmore_model.G['icd'].edges()
    icd_rels=[icd_rels[0].cpu().detach().numpy(),icd_rels[1].cpu().detach().numpy()]

if use_snomed==True:
#    snomed_edge_types=etypes.cpu().detach().numpy()
    #snomed_rels=mmore_model.G['snomed'][0].edges()
    #snomed_rels=[snomed_rels[0].cpu().detach().numpy()-args.dxVocabSize,snomed_rels[1].cpu().detach().numpy()]
#    print('relnum',len(snomed_rels[0]))

    snomed_rels=[]
    snomed_edge_types=[]
    snomed_anc_types=[]
  #  print(snomed_rels[0])
#    snomed_extra_embs=mmore_model.snomed_extraEmb.cpu().detach().numpy()
    snomed_extra_embs=np.array(mmore_model.snomed_extraEmb.cpu().weight.data)

#    print(mmore_model.G['icd'].edges())
 #   print(snomed_rels,snomed_anc_types,len(set(dx_rel_extra_entitys)),len(set(snomed_rels[0])))
#    print(dx_anc_types)
#   1/0
#pickle.dump([dxseqs,dxlabels,dx_Y,dx_embs,visit_attentions,icd_ontology_attention,dx_anc_types,icd_rels,snomed_extra_embs,snomed_rels,snomed_anc_types,snomed_edge_types],open('../outputs/mimic/output.pk','wb'))

print('Best epoch: {:3d} | DPACC: {:.5f} '.format(best_epoch_num, best_test_acc))
if task=='readm':
    print('Best epoch: {:3d} | DPAuc: {:.5f} '.format(best_epoch_num, best_test_auc))

