"""
Additional layers.
"""
import numpy as np
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda')
#device = torch.device('cpu')
#device = torch.device('cuda')

class OntoEmb(nn.Module):
    """docstring for OntoEmb"""
    def __init__(self, dxVocabSize, dxnumAncestors, dxEmbDim, attnDim):
        super(OntoEmb, self).__init__()
        self.dxVocabSize = dxVocabSize
        self.dxnumAncestors = dxnumAncestors
        self.dxEmbDim = dxEmbDim
        self.attnDim = attnDim
        self.dxEmb = nn.Embedding(self.dxVocabSize+self.dxnumAncestors, self.dxEmbDim)
        self.attn = nn.Linear(2*self.dxEmbDim, self.attnDim)
        self.attnCombine = nn.Linear(self.attnDim, 1)

    def forward(self, ontoInput):
        leavesList, ancestorsList = ontoInput
        tempAllEmb = []
        for leaves, ancestors in zip(leavesList, ancestorsList):
            leavesEmb = self.dxEmb(leaves.to(device))
            ancestorsEmb = self.dxEmb(ancestors.to(device))
            attnInput = torch.cat((leavesEmb, ancestorsEmb), dim=2)
            mlpOutput = torch.tanh(self.attn(attnInput))
            preAttn = self.attnCombine(mlpOutput)
            attn = F.softmax(preAttn, dim=1)
            tempEmb = torch.sum(ancestorsEmb*attn, dim=1)
            tempAllEmb.append(tempEmb)
        tempAllEmb.append(torch.zeros(1, self.dxEmbDim).to(device))
        allEmb = torch.cat([i for i in tempAllEmb], dim=0)
	
        return allEmb
class SnomedEmb(nn.Module):
    """docstring for OntoEmb"""
    def __init__(self, dxVocabSize, dxnumAncestors, dxnumRelations,dxEmbDim, attnDim,dxEmb=None,dxAnEmb=None):
        super(SnomedEmb, self).__init__()
        self.dxVocabSize = dxVocabSize
        self.dxnumAncestors = dxnumAncestors
        self.dxnumRelations = dxnumRelations
        self.dxEmbDim = dxEmbDim
        self.attnDim = attnDim
        if dxEmb==None:
            self.dxEmb = nn.Embedding(self.dxVocabSize, self.dxEmbDim)
        else:
            self.dxEmb = dxEmb
        #self.t=nn.Linear(self.dxEmbDim, self.dxEmbDim)
        #self.t=nn.Embedding(self.dxVocabSize, self.dxEmbDim)
        self.t=nn.Embedding(1, self.dxEmbDim)
       # self.dxEmb = dxEmb
        if dxAnEmb==None:
            self.dxAnEmb = nn.Embedding(self.dxnumAncestors+1, self.dxEmbDim)
        else:
            self.dxAnEmb = dxAnEmb
        self.dxReEmb = nn.Embedding(self.dxnumRelations+1, self.dxEmbDim)
        self.attn = nn.Linear(2*self.dxEmbDim, self.attnDim)
        self.attnCombine = nn.Linear(self.attnDim, 1)
        self.anW = nn.Linear(self.dxEmbDim,self.dxEmbDim )

    def forward(self, ontoInput):
        dxEmb,leavesList, ancestorsList,relationList,permute_index = ontoInput## the leave order from rel_map,eg(29,5,6..), so need permute_index to reorder
        #print(leavesList)
        #1/0
        tempAllEmb = []
        for leaves, ancestors,relations in zip(leavesList, ancestorsList,relationList):
            if len(leaves)==0:continue

            #relations=relations[:5]
            relations=relations.to(device)
            leaves=leaves.to(device)
            ancestors=ancestors.to(device)
            relationsEmb = self.dxReEmb(relations)
            #1/0
            leavesEmb = self.dxEmb(leaves)
            #leavesEmb = dxEmb[leaves.to(device)]
#            print(leaves[0],leaves.size())
            #leavesEmb = dxEmb(leaves)
            ancestorsEmb = self.dxAnEmb(ancestors)
            #print(relations,np.array(leaves).shape,np.array(ancestors).shape,np.array(relations).shape,self.dxnumRelations, self.dxEmbDim)

#            1/0

            #relations=relations.to(device)
            #relationsEmb = self.dxReEmb(relations)
            #1/0
            ll=leavesEmb[:,[0],:]
            #print(ll.size())
            ancestorsEmb = torch.cat(( ancestorsEmb,ll), dim=1)
            leavesEmb = torch.cat(( leavesEmb,ll), dim=1)
            relationsEmb = torch.cat(( relationsEmb,self.t(torch.LongTensor([[0]]*ll.size()[0]).to(device))), dim=1)
            #relationsEmb = torch.cat(( relationsEmb,ll), dim=1)
#            print(np.array(leaves).shape,np.array(ancestors).shape,np.array(relations).shape,self.dxnumRelations, self.dxEmbDim)

            attnInput = torch.cat((leavesEmb, ancestorsEmb+relationsEmb), dim=2)

            #attnInput = torch.cat((leavesEmb, ancestorsEmb), dim=2)
            mlpOutput = torch.tanh(self.attn(attnInput))
            preAttn = self.attnCombine(mlpOutput)
            attn = F.softmax(preAttn, dim=1)
            tempEmb = torch.sum((ancestorsEmb+relationsEmb)*attn, dim=1)
            #tempEmb = torch.sum((ancestorsEmb), dim=1)
            #tempEmb = torch.mean((ancestorsEmb), dim=1)
            #tempEmb = torch.sum(((leavesEmb+relationsEmb))*attn, dim=1)
            #print(tempEmb.size(),leavesEmb[:,0,:].size())
            #tempAllEmb.append(0.5*tempEmb+0.5*leavesEmb[:,0,:])
            tempAllEmb.append(tempEmb)
#        print(permute_index)
        #print(leavesList)
        #print(len(tempAllEmb))
       
        #tempAllEmb=[tempAllEmb[i] for i in permute_index]
        #permute_index.append(len(permute_index))
        tempAllEmb.append(torch.zeros(1, self.dxEmbDim).to(device))
        allEmb = torch.cat([i for i in tempAllEmb], dim=0)
        allEmb=allEmb[permute_index,:]
#        print(np.array(allEmb.cpu().detach().numpy() ))
        #print(allEmb.size(),allEmb[permute_index,:].size())
	
        return allEmb
class class_aware_rgat(nn.Module):
    """docstring for OntoEmb"""
    def __init__(self, dxVocabSize, dxnumAncestors, dxnumRelations,dxEmbDim, attnDim,dxEmb=None,dxAnEmb=None):
        super(class_aware_rgat, self).__init__()
        self.dxVocabSize = dxVocabSize
        self.dxnumAncestors = dxnumAncestors
        self.dxnumRelations = dxnumRelations
        self.dxEmbDim = dxEmbDim
        self.attnDim = attnDim
        if dxEmb==None:
            self.dxEmb = nn.Embedding(self.dxVocabSize+1, self.dxEmbDim)
        else:
            self.dxEmb = dxEmb
        #self.t=nn.Linear(self.dxEmbDim, self.dxEmbDim)
        #self.t=nn.Embedding(self.dxVocabSize, self.dxEmbDim)
        self.t=nn.Embedding(1, self.dxEmbDim)
       # self.dxEmb = dxEmb
        dxAnEmb=None
        if dxAnEmb==None:
            self.dxAnEmb = nn.Embedding(self.dxVocabSize+self.dxnumAncestors+1, self.dxEmbDim)
        else:
            self.dxAnEmb = dxAnEmb
        self.dxReEmb = nn.Embedding(self.dxnumRelations+1, self.dxEmbDim)
        self.attn = nn.Linear(2*self.dxEmbDim, self.attnDim)
        self.attnCombine = nn.Linear(self.attnDim, 1)
        self.anW = nn.Linear(self.dxEmbDim,self.dxEmbDim )
        self.class_coefficient_layer = nn.Linear(self.dxEmbDim, 1)
#        self.selfrelationemb =Variable(torch.randn(1, self.dxEmbDim).type(torch.FloatTensor), requires_grad=True)

    def forward(self, ontoInput):
        dxEmb,classEmb,leavesList, ancestorsList,relationList,permute_index = ontoInput## the leave order from rel_map,eg(29,5,6..), so need permute_index to reorder
        #print(leavesList)
        #1/0
        tempAllEmb = []
        c=0
        for i,leaves, ancestors,relations in zip(range(len(leavesList)),leavesList, ancestorsList,relationList):
#            print(i,leaves,permute_index[i])
            leaves=leaves.to(device)
            ancestors=ancestors.to(device)
            relations=relations.to(device)
            leavesList[i]=leaves
            ancestorsList[i]=ancestors
            relationList[i]=relations
        for leaves, ancestors,relations in zip(leavesList, ancestorsList,relationList):
            '''
            print('cc',c)
            print('cc',relations,self.dxnumRelations)
            print('cc',leaves,self.dxVocabSize)
            print('cc',ancestors,self.dxnumAncestors)
            '''
            if c>=self.dxVocabSize:continue
            c+=1
        #    if len(leaves)==0:continue

            #relations=relations[:5]
#            relations=torch.cat((relations,torch.LongTensor([self.dxnumRelations])),dim=0)
            #leaves=leaves.to(device)
          #  ancestors=ancestors.to(device)
           # relations=relations.to(device)
            relationsEmb = self.dxReEmb(relations)
            #1/0
            leavesEmb = dxEmb[leaves]
            leaveclassEmb = classEmb[leaves[0]]
            class_coe=self.class_coefficient_layer(leaveclassEmb)

            relationsEmb = class_coe*relationsEmb##rescale relation bias by classes coe

            #leavesEmb = dxEmb[leaves.to(device)]
#            print(leaves[0],leaves.size())
            #leavesEmb = dxEmb(leaves)
            ancestorsEmb = self.dxAnEmb(ancestors)
            #print(relations,np.array(leaves).shape,np.array(ancestors).shape,np.array(relations).shape,self.dxnumRelations, self.dxEmbDim)

#            1/0

            #relations=relations.to(device)
            #relationsEmb = self.dxReEmb(relations)
            #1/0
#            print('s',self.dxEmb)
 #           print('a',leavesEmb)
  #          1/0
  #          print(leavesEmb.size())
   #         print('a',ancestorsEmb.size())
            #ll=leavesEmb[:,[0],:]
 #           ll=leavesEmb[0:1,:]
            #rl=self.dxReEmb(torch.tensor([self.dxnumRelations]).to(device))
#            rl=ll*0
#            print('l',ll.size())
 #           1/0
        #    ancestorsEmb = torch.cat(( ancestorsEmb,ll), dim=0)
        #    leavesEmb = torch.cat(( leavesEmb,ll), dim=0)
         #   relationsEmb = torch.cat(( relationsEmb,rl), dim=0)
           # print('r',relationsEmb.size())
            #relationsEmb = torch.cat(( relationsEmb,self.t(torch.LongTensor([0]*ll.size()[1]).to(device))), dim=0)
 #           relationsEmb = torch.cat(( relationsEmb,self.selfrelationemb.to(device)), dim=0)
            #print('r',relationsEmb.size())
            #1/0
            #relationsEmb = torch.cat(( relationsEmb,ll), dim=1)
#            print(np.array(leaves).shape,np.array(ancestors).shape,np.array(relations).shape,self.dxnumRelations, self.dxEmbDim)
            attnInput = torch.cat((leavesEmb, ancestorsEmb+relationsEmb), dim=1)

            #attnInput = torch.cat((leavesEmb, ancestorsEmb), dim=2)
            mlpOutput = torch.tanh(self.attn(attnInput))
            preAttn = self.attnCombine(mlpOutput)
            attn = F.softmax(preAttn, dim=1)
            tempEmb = torch.sum((ancestorsEmb+relationsEmb)*attn, dim=0)
            #tempEmb = torch.sum((ancestorsEmb), dim=1)
            #tempEmb = torch.mean((ancestorsEmb), dim=1)
            #tempEmb = torch.sum(((leavesEmb+relationsEmb))*attn, dim=1)
            #print(tempEmb.size(),leavesEmb[:,0,:].size())
            #tempAllEmb.append(0.5*tempEmb+0.5*leavesEmb[:,0,:])
#            print(tempEmb.size())
            tempAllEmb.append(tempEmb[None,:])
#        print(permute_index)
        #print(leavesList)
        #print(len(tempAllEmb))
       
        #tempAllEmb=[tempAllEmb[i] for i in permute_index]
        #permute_index.append(len(permute_index))
        tempAllEmb.append(torch.zeros(1, self.dxEmbDim).to(device))
        allEmb = torch.cat([i for i in tempAllEmb], dim=0)
        #print(allEmb.size())
        #1/0
        allEmb=allEmb[permute_index,:].to(device)
#        print(np.array(allEmb.cpu().detach().numpy() ))
        #print(allEmb.size(),allEmb[permute_index,:].size())
	
        return allEmb
