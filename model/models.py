import torch
from torch.autograd import Variable
import numpy as np
import dgl
import torch.nn as nn
import torch.nn.functional as F
from model import layers
from model import dgl_layer
from dgl.nn.pytorch import GATConv,RelGraphConv
use_ontology=True
use_ctd=False
def multi_class_cross_entropy_loss(predictions, labels):
    loss = -torch.mean(torch.sum(torch.sum(labels * torch.log(predictions), dim=1), dim=1))
    return loss
class two_layer_model(nn.Module):##snomed first, icd then
    def __init__(self, args,G,dx_to_ancestor_matrix,rel_num,use_ontology=False,use_icd=False,use_snomed=False,use_ctd=False):
        super(two_layer_model, self).__init__()
#        self.wt_dict = nn.ParameterDict()
        self.use_ontology=use_ontology
        self.use_icd=use_icd
        self.use_snomed=use_snomed
        self.use_ctd=use_ctd
        for k, v in args.__dict__.items():
            self.__setattr__(k, v)
        self.G=G
        if self.use_ontology==True:
            self.icdontoEmb =GATConv(self.EHREmbDim, self.EHREmbDim,1 ).to(self.device)
    #       self.snomedontoEmb =GATConv(400, 400,1 ).to(self.device)
 #           if self.use_snomed==True:
            #    self.snomedontoEmb =RelGraphConv(self.EHREmbDim, self.EHREmbDim, self.snomed_rel_num, layer_norm=True)
#                self.snomedAnsDxEmb = nn.Embedding(self.snomed_dx_ans_num, self.EHREmbDim)
            self.ICDancEmb_list=[]
            for i in range(self.split_max_num):
#                self.__dict__['split_icd_emb{0}'.format(i)]=nn.Embedding(self.dxnumAncestors, self.EHREmbDim)
                #self.ICDancEmb_list.append(nn.Embedding(self.dxnumAncestors, self.EHREmbDim).to(self.device))
                setattr(self, 'split_icd_emb{0}'.format(i), nn.Embedding(self.dxnumAncestors, self.EHREmbDim))
                self.ICDancEmb_list.append(getattr(self,'split_icd_emb{0}'.format(i)))
            if self.use_ctd==True:
                #self.ctdontoEmb = dgl_layer.GATLayer(G['ctd'],50,50).to(self.device)
                self.ctdontoEmb =GATConv(self.EHREmbDim, self.EHREmbDim,1 ).to(self.device)
    #        self.ICDancEmb2 = nn.Embedding(self.dxnumAncestors, self.EHREmbDim)
        '''
        self.icdontoEmb = dgl_layer.GATLayer(G[0],50,50).to(self.device)
      #  self.icdontoEmb = dgl.nn.pytorch.GAT(G,in_dim=400,hidden_dim=100,out_dim=400,num_heads=1)
        self.ctdontoEmb = dgl_layer.GATLayer(G[1],50,50).to(self.device)
        '''

#        self.dxontoEmb = layers.OntoEmb(self.dxVocabSize, self.dxnumAncestors, self.ontoEmbDim, self.ontoattnDim)        
 #       self.drugontoEmb = layers.OntoEmb(self.drugVocabSize, self.drugnumAncestors, self.ontoEmbDim, self.ontoattnDim)        

        self.dxALLontoEmb=None
        self.cooccurLinear = nn.Linear(self.EHREmbDim, self.dxVocabSize+self.drugVocabSize)
        self.dxproLinear = nn.Linear(2*self.EHREmbDim, self.EHREmbDim)
        self.ontoLinear = nn.Linear(self.EHREmbDim,self.EHREmbDim,bias=False )
        self.ctdLinear = nn.Linear(self.EHREmbDim,self.EHREmbDim,bias=False )
        self.onto_b = nn.Parameter(data=torch.rand((self.EHREmbDim), dtype=torch.float), requires_grad=True)
#        self.wt_dict['onto_b'] =self.onto_b
        self.register_parameter(name='onto_b', param=self.onto_b)
        self.EHRdxEmb = nn.Embedding(self.dxVocabSize+1, self.EHREmbDim, padding_idx=self.dxVocabSize)
        #self.ICDancEmb = nn.Embedding(self.dxnumAncestors, self.EHREmbDim)
        
        self.EHRdrugEmb = nn.Embedding(self.drugVocabSize+1, self.EHREmbDim, padding_idx=self.drugVocabSize)
        self.EHRdxEmb2 = nn.Embedding(self.dxVocabSize+1, self.EHREmbDim, padding_idx=self.dxVocabSize)
        self.EHRdrugEmb2 = nn.Embedding(self.drugVocabSize+1, self.EHREmbDim, padding_idx=self.drugVocabSize)
        self.gfeatures={}

        #self.gfeatures['extra_rx']=self.ICDancEmb(torch.LongTensor(range(self.drugnumAncestors)))
#        self.gfeatures['extra_rx']=self.ICDancEmb(torch.LongTensor(range(self.drugnumAncestors)))

#        self.dxrelationEmb = layers.SnomedEmb(self.dxVocabSize, self.dxrelationnumAncestors,  self.dxrelationnum,self.ontoEmbDim, self.ontoattnDim)        

        if self.use_snomed==True:
            self.dxrelationEmb = layers.class_aware_rgat(self.dxVocabSize, self.dxrelationnumAncestors,  self.dxrelationnum,self.ontoEmbDim, self.ontoattnDim)        
        #self.dxrelationEmb = layers.SnomedEmb(self.EHRdxEmb,self.dxVocabSize, self.dxrelationnumAncestors,  self.dxrelationnum,self.ontoEmbDim, self.ontoattnDim)        
 #       self.drugrelationEmb = layers.SnomedEmb(self.drugVocabSize, self.drugrelationnumAncestors,  self.drugrelationnum,self.ontoEmbDim, self.ontoattnDim)        
        if self.use_ontology==False:
            self.attn = nn.Linear(2*self.EHREmbDim, self.ptattnDim)
            self.dpPredLinear = nn.Linear(self.EHREmbDim, self.dpLabelSize)
            self.readPredLinear = nn.Linear(self.EHREmbDim, 1)
            self.mortPredLinear = nn.Linear(self.EHREmbDim, 1)
        else:
            self.attn = nn.Linear(2*self.ontoEmbDim+2*self.EHREmbDim, self.ptattnDim)
            self.dpPredLinear = nn.Linear(self.ontoEmbDim+self.EHREmbDim, self.dpLabelSize)
            self.readPredLinear = nn.Linear(self.ontoEmbDim+self.EHREmbDim, 1)
            self.mortPredLinear = nn.Linear(self.ontoEmbDim+self.EHREmbDim, 1)
            self.SingleclassEmb=nn.Parameter(data=torch.rand((self.dxnumAncestors, self.EHREmbDim), dtype=torch.float), requires_grad=True)
            self.register_parameter(name='SingleclassEmb', param=self.SingleclassEmb)
#            self.wt_dict['SingleclassEmb'] =self.SingleclassEmb
        self.attnCombine = nn.Linear(self.ptattnDim, 1)
        self.dx_to_ancestor_matrix=dx_to_ancestor_matrix

    def onto_loss(self,extra_ents,ents):

        #ctd_loss=1*torch.sum((self.ctd_W(self.EHRdxEmb(ctd_dx))+self.ctd_b-self.ctd_W2(self.EHRdrugEmb(ctd_rx)))**2)
        #ctd_loss=1*torch.sum((self.EHRdxEmb(ctd_dx)+self.ctd_b-self.EHRdrugEmb(ctd_rx))**2)
        self.gfeatures['dx']=self.EHRdxEmb(torch.LongTensor(range(self.dxVocabSize)).to(self.device)).to(self.device)
        self.gfeatures['extra_dx']=self.ICDancEmb(torch.LongTensor(range(self.split_max_num*self.dxnumAncestors)).to(self.device)).to(self.device)
        self.gfeatures['rx']=self.EHRdrugEmb(torch.LongTensor(range(self.drugVocabSize)).to(self.device)).to(self.device)
#        icdembds2=self.ICDancEmb2(torch.LongTensor(range(self.dxnumAncestors)).to(self.device)).to(self.device)
            #dxALLontoEmb = self.ontoEmb(self.G,self.gfeatures)['dx']
            #concat_embs=torch.cat((self.gfeatures['dx'],self.gfeatures['extra_dx'],icdembds2))
        concat_embs=torch.cat((self.gfeatures['dx'],self.gfeatures['extra_dx']))
        #concat_embs=torch.cat((self.gfeatures['dx'],self.gfeatures['extra_dx'],icdembds2))
        #ontoloss=1*torch.sum((self.ontoLinear(concat_embs[extra_ents])-concat_embs[ents])**2)/extra_ents.size(0)
        ontoloss=1*torch.sum((concat_embs[extra_ents]-concat_embs[ents])**2)/extra_ents.size(0)
        #ctd_loss=1*torch.sum((self.ctd_W(self.EHRdxEmb(ctd_dx))-self.EHRdxEmb(ctd_dx))**2)
        return ontoloss
    def align_loss(self):
        #l=torch.sum((self.icd_dxAllontoEmb-self.ctd_dxALLontoEmb)**2)
        #l=torch.sum((self.icd_dxAllontoEmb-self.snomed_dxAllontoEmb)**2)
        l=torch.mean((self.icd_dxAllontoEmb-self.gfeatures['dx'])**2)*1000
        return l
    def preset(self,preinput):
        ( dxLeavesList, dxAncestorsList, drugLeavesList, drugAncestorsList,dxrelationLeavesList,dxrelationAncestorsList,dxrelationList,dxrelation_permuteindex,drugrelationLeavesList,drugrelationAncestorsList,drugrelationList,drugrelation_permuteindex) = preinput
        self.classEmb=F.normalize(torch.matmul(self.dx_to_ancestor_matrix,self.SingleclassEmb), p=2, dim=1)
        if self.use_ontology==True:
            self.gfeatures['dx']=self.EHRdxEmb(torch.LongTensor(range(self.dxVocabSize)).to(self.device)).to(self.device)
            self.gfeatures['extra_dx']=[]
            for i in range(self.split_max_num):
                sid=torch.LongTensor(range(self.dxnumAncestors)).to(self.device)
                self.gfeatures['extra_dx'].append(self.ICDancEmb_list[i](sid).to(self.device))

            self.gfeatures['rx']=self.EHRdrugEmb(torch.LongTensor(range(self.drugVocabSize)).to(self.device)).to(self.device)
            self.gfeatures['dx2']=self.EHRdxEmb2(torch.LongTensor(range(self.dxVocabSize)).to(self.device)).to(self.device)
            self.gfeatures['rx2']=self.EHRdrugEmb2(torch.LongTensor(range(self.drugVocabSize)).to(self.device)).to(self.device)
           
            if self.use_ctd==True:
                ctd_concat_embs=torch.cat((self.gfeatures['dx2'],self.gfeatures['rx2'])).to(self.device)
                ctd_embs=torch.squeeze(self.ctdontoEmb(self.G['ctd'],ctd_concat_embs))
                self.ctd_dxALLontoEmb = ctd_embs[:self.dxVocabSize]
                self.ctd_drugALLontoEmb = ctd_embs[self.dxVocabSize:]
                ctd_rxALLontoEmb = ctd_embs[self.dxVocabSize:]
                self.dxALLontoEmb = self.dxALLontoEmb+self.ctd_dxALLontoEmb
            
            if self.use_snomed==True:
               # new_row = torch.Tensor(np.zeros(shape=(1, self.EHREmbDim))).to(self.device)
               # self.dxALLontoEmb = torch.cat((self.dxALLontoEmb, new_row)).to(self.device)
#                self.classEmb=dxALLontoEmb
#                self.classEmb=F.normalize(torch.matmul(self.dx_to_ancestor_matrix,self.SingleclassEmb), p=2, dim=1)
                #dxrelationInputs = (self.dxALLontoEmb,self.classEmb,dxrelationLeavesList,dxrelationAncestorsList,dxrelationList,dxrelation_permuteindex)
                dxrelationInputs = (self.gfeatures['dx'],self.classEmb,dxrelationLeavesList,dxrelationAncestorsList,dxrelationList,dxrelation_permuteindex)

                dxrelationALLontoEmb = self.dxrelationEmb(dxrelationInputs)
               # self.dxALLontoEmb =self.dxALLontoEmb+ dxrelationALLontoEmb
                self.dxALLontoEmb =dxrelationALLontoEmb[:-1]
                self.snomed_extraEmb=self.dxrelationEmb.dxAnEmb
            # icd_emb_list=[self.gfeatures['dx']]+self.gfeatures['extra_dx']

            icd_emb_list=[self.dxALLontoEmb]+self.gfeatures['extra_dx']
            concat_embs=torch.cat(icd_emb_list).to(self.device)
            
            if self.use_icd==True:
                self.icd_dxAllontoEmb=torch.squeeze(self.icdontoEmb(self.G['icd'],concat_embs)[:self.dxVocabSize])
                self.icd_edges_attentions=self.G['icd'].edata['a'].cpu().detach().numpy()
                #dxALLontoEmb = f.normalize(self.icd_dxallontoemb, p=2, dim=1)
                self.dxALLontoEmb = self.icd_dxAllontoEmb
            else:
                dxALLontoEmb = 0

            new_row = torch.Tensor(np.zeros(shape=(1, self.EHREmbDim))).to(self.device)
            self.dxALLontoEmb = torch.cat((self.dxALLontoEmb, new_row)).to(self.device)
#            self.dxALLontoEmb=F.normalize(self.dxALLontoEmb, p=2, dim=1)
    def forward(self, inputs):
        (dxseqs, drugseqs, dx_onehot, drug_onehot, 
            dxLeavesList, dxAncestorsList, drugLeavesList, drugAncestorsList,dxrelationLeavesList,dxrelationAncestorsList,dxrelationList,dxrelation_permuteindex,drugrelationLeavesList,drugrelationAncestorsList,drugrelationList,drugrelation_permuteindex) = inputs

#        (dxseqs, drugseqs, dx_onehot, drug_onehot) = inputs
        dxseqs = dxseqs.to(self.device)
        drugseqs = drugseqs.to(self.device)
        dx_onehot = dx_onehot.to(self.device)
        drug_onehot = drug_onehot.to(self.device)
 #       print(dxseqs,dxseqs.size())
#        1/0
        dx_num = list(dxseqs.size())[2]
        drug_num = list(drugseqs.size())[2]
#        print(self.EHRdxEmb.cpu(),self.dxVocabSize)


#        print(dxseqs.cpu(),np.array(dxseqs.cpu()).shape)
 #       print(np.array(self.EHRdxEmb.cpu()).shape)
  #      1/0
        #print(dxseqs,np.array(dxseqs.cpu()).shape)
        dxEHREmb = self.EHRdxEmb(dxseqs)
#        print(dxEHREmb.cpu().detach().numpy().shape)
 #       1/0
        drugEHREmb = self.EHRdrugEmb(drugseqs)
        #EHREmb = torch.cat((dxEHREmb, drugEHREmb), dim=2)##change
#        EHREmb = dxEHREmb

        EHREmb = torch.cat((dxEHREmb, drugEHREmb), dim=2)##change
#        print('ddddddd',self.EHRdxEmb,self.gfeatures['dx'])
        if self.use_ontology==True:
            '''
#            aa=nn.Linear(400,400)

            self.gfeatures['dx']=self.EHRdxEmb(torch.LongTensor(range(self.dxVocabSize)).to(self.device)).to(self.device)
#            self.gfeatures['extra_dx']=self.ICDancEmb(torch.LongTensor(range(self.split_max_num*self.dxnumAncestors)).to(self.device)).to(self.device)
            self.gfeatures['extra_dx']=[]
            for i in range(self.split_max_num):
                sid=torch.LongTensor(range(self.dxnumAncestors)).to(self.device)
                self.gfeatures['extra_dx'].append(self.ICDancEmb_list[i](sid).to(self.device))

    #        print(self.gfeatures['extra_dx'].cpu().detach().numpy()[:5],'\n')
           # print(self.EHRdxEmb.weight,'\n')
           # print(self.gfeatures['dx'].cpu().detach().numpy()[:5])
            self.gfeatures['rx']=self.EHRdrugEmb(torch.LongTensor(range(self.drugVocabSize)).to(self.device)).to(self.device)
            self.gfeatures['dx2']=self.EHRdxEmb2(torch.LongTensor(range(self.dxVocabSize)).to(self.device)).to(self.device)
            self.gfeatures['rx2']=self.EHRdrugEmb2(torch.LongTensor(range(self.drugVocabSize)).to(self.device)).to(self.device)
        #    icdembds2=self.ICDancEmb2(torch.LongTensor(range(self.dxnumAncestors)).to(self.device)).to(self.device)
            #dxALLontoEmb = self.ontoEmb(self.G,self.gfeatures)['dx']
            #concat_embs=torch.cat((self.gfeatures['dx'],self.gfeatures['extra_dx'],icdembds2)).to(self.device)
            icd_emb_list=[self.gfeatures['dx']]+self.gfeatures['extra_dx']
            #concat_embs=torch.cat((self.gfeatures['dx'],self.gfeatures['extra_dx'])).to(self.device)
            concat_embs=torch.cat(icd_emb_list).to(self.device)
            
#            concat_embs=torch.cat((self.gfeatures['dx'],self.gfeatures['extra_dx'])).to(self.device)
            #concat_embs=torch.cat((self.gfeatures['dx'],self.gfeatures['extra_dx'],self.gfeatures['rx']))
 #           print('ddddddd',self.gfeatures['dx'].shape,concat_embs.shape)
#            1/0
            #dxALLontoEmb = self.icdontoEmb(concat_embs)[:self.dxVocabSize]
            
#            print(self.G[0].edges(),self.icd_edges_attentions.shape)
 #           1/0
            #print(self.G[2][1].size(),self.G[2][0].num_edges())
#            dxALLontoEmb = self.icd_dxAllontoEmb
#            1/0
            if self.use_icd==True:
                self.icd_dxAllontoEmb=torch.squeeze(self.icdontoEmb(self.G['icd'],concat_embs)[:self.dxVocabSize])
                self.icd_edges_attentions=self.G['icd'].edata['a'].cpu().detach().numpy()
                #dxALLontoEmb = f.normalize(self.icd_dxallontoemb, p=2, dim=1)
                self.dxALLontoEmb = self.icd_dxAllontoEmb
            else:
                self.dxALLontoEmb = 0
            if self.use_ctd==True:
                ctd_concat_embs=torch.cat((self.gfeatures['dx2'],self.gfeatures['rx2'])).to(self.device)
                ctd_embs=torch.squeeze(self.ctdontoEmb(self.G['ctd'],ctd_concat_embs))
                self.ctd_dxALLontoEmb = ctd_embs[:self.dxVocabSize]
                self.ctd_drugALLontoEmb = ctd_embs[self.dxVocabSize:]
                ctd_rxALLontoEmb = ctd_embs[self.dxVocabSize:]
            #dxALLontoEmb = (self.icd_dxAllontoEmb+self.ctd_dxALLontoEmb)/2
                self.dxALLontoEmb = self.dxALLontoEmb+self.ctd_dxALLontoEmb
#            dxALLontoEmb = (self.icd_dxAllontoEmb+self.snomed_dxAllontoEmb)/2
            
            if self.use_snomed==True:
#                new_row = torch.Tensor(np.zeros(shape=(1, self.EHREmbDim))).to(self.device)
 #               dxALLontoEmb = torch.cat((dxALLontoEmb, new_row)).to(self.device)
#                self.classEmb=dxALLontoEmb
#                self.classEmb=F.normalize(torch.matmul(self.dx_to_ancestor_matrix,self.SingleclassEmb), p=2, dim=1)
                dxrelationInputs = (self.dxALLontoEmb,self.classEmb,dxrelationLeavesList,dxrelationAncestorsList,dxrelationList,dxrelation_permuteindex)
                #dxrelationInputs = (self.gfeatures['dx'],classEmb,dxrelationLeavesList,dxrelationAncestorsList,dxrelationList,dxrelation_permuteindex)

                dxrelationALLontoEmb = self.dxrelationEmb(dxrelationInputs)
                self.dxALLontoEmb =dxrelationALLontoEmb
#                dxALLontoEmb =dxALLontoEmb+ dxrelationALLontoEmb
                #dxALLontoEmb =self.snomed_dxAllontoEmb
            #dxALLontoEmb = self.dxproLinear(torch.cat((self.icd_dxAllontoEmb,self.snomed_dxAllontoEmb),axis=1))
            #print('ddddddd',self.ontoEmb.rel_Ws,dxALLontoEmb,self.gfeatures['dx'])
#            print('ddddddd',self.ontoEmb.rel_Ws.weight)
#            print(aa.weight)
   #         print(dxALLontoEmb)
    #        print(dxALLontoEmb.size())
#            1/0
#            print(dxALLontoEmb.size())
 #           1/0
         #   drugALLontoEmb = torch.cat((ctd_rxALLontoEmb, new_row)).to(self.device)

        #    drugALLontoEmb = self.ontoEmb(self.G,self.gfeatures)['rx']
            '''
            drugALLontoEmb = self.EHRdrugEmb
#            new_row = torch.Tensor(np.zeros(shape=(1, self.EHREmbDim))).to(self.device)
#            self.dxALLontoEmb = torch.cat((self.dxALLontoEmb, new_row)).to(self.device)
            #drugALLontoEmb = self.ctd_drugALLontoEmb
            #drugALLontoEmb =self.EHRdrugEmb(drugseqs)

        '''
        dxrelationInputs = (dxALLontoEmb,dxrelationLeavesList,dxrelationAncestorsList,dxrelationList,dxrelation_permuteindex)
        drugrelationInputs = (drugALLontoEmb,drugrelationLeavesList,drugrelationAncestorsList,drugrelationList,drugrelation_permuteindex)
        dxrelationALLontoEmb = self.dxrelationEmb(dxrelationInputs)
        drugrelationALLontoEmb = self.drugrelationEmb(drugrelationInputs)
        '''
    #    dxALLontoEmb=0.5*dxALLontoEmb+0.5*dxrelationALLontoEmb
        #dxALLontoEmb=dxALLontoEmb+dxrelationALLontoEmb##chanfe

#        if use_ontology==True:
 #           dxALLontoEmb=dxALLontoEmb##chanfe
            #dxALLontoEmb=dxALLontoEmb
#            drugALLontoEmb=drugALLontoEmb
        #drugALLontoEmb=drugALLontoEmb+drugrelationALLontoEmb
        #dxALLontoEmb=dxrelationALLontoEmb
        #drugALLontoEmb=drugrelationALLontoEmb
        #print(np.array(dxALLontoEmb.cpu()).shape,np.array(dxEHRVEmb.cpu()).shape)
        '''
        print(dxseqs)
        print(dxALLontoEmb.size(),dxEHREmb.size(),dxseqs.size())
        1/0
        '''
        if self.use_ontology==True:
#            self.dxALLontoEmb=dxALLontoEmb
            dxOntoEmb = self.dxALLontoEmb[dxseqs]
            #drugOntoEmb = drugALLontoEmb[drugseqs]
            drugOntoEmb = drugALLontoEmb(drugseqs)
            ontoEmb = torch.cat((dxOntoEmb, drugOntoEmb), dim=2)
#            drugOntoEmb = drugALLontoEmb[drugseqs]
            #ontoEmb = torch.cat((dxOntoEmb, drugOntoEmb), dim=2)
         #   ontoEmb = dxOntoEmb
        '''
        dxrelationOntoEmb = dxrelationALLontoEmb[dxseqs]
        drugrelationOntoEmb = drugrelationALLontoEmb[drugseqs]
        '''



        #relontoEmb = torch.cat((dxrelationOntoEmb, drugrelationOntoEmb), dim=2)
#        print(ontoEmb.size(),relontoEmb.size())

        dxEHRVEmb = F.normalize(torch.sum(self.EHRdxEmb(dxseqs), dim=2), p=2, dim=2)
        drugEHRVEmb= F.normalize(torch.sum(self.EHRdrugEmb(drugseqs), dim=2), p=2, dim=2)

        EHRVEmb = dxEHRVEmb+drugEHRVEmb##chan
        #EHRVEmb = dxEHRVEmb##chan
        if self.use_ontology==True:
            dxontoVEmb = torch.matmul(dx_onehot.permute(1,0,2), self.dxALLontoEmb[:-1])
            #dxrelontoVEmb = torch.matmul(dx_onehot.permute(1,0,2), dxrelationALLontoEmb[:-1])
#            drugontoVEmb = torch.matmul(drug_onehot.permute(1,0,2), drugALLontoEmb[:-1])
        #drugrelontoVEmb = torch.matmul(drug_onehot.permute(1,0,2), drugrelationALLontoEmb[:-1])


        t=self.cooccurLinear(EHRVEmb)
        cooccurU = F.softmax(self.cooccurLinear(EHRVEmb), dim=2).contiguous()
        vonehot = torch.cat((dx_onehot.permute(1,0,2), drug_onehot.permute(1,0,2)), dim=2).contiguous()
        cooccur_loss = multi_class_cross_entropy_loss(cooccurU, vonehot)
        if self.use_ontology==True:
            #ontoVEmb = F.normalize(dxontoVEmb, p=2, dim=2)+F.normalize(drugontoVEmb, p=2, dim=2)
            ontoVEmb = F.normalize(dxontoVEmb, p=2, dim=2)

#        relontoVEmb = F.normalize(dxrelontoVEmb, p=2, dim=2)+F.normalize(drugrelontoVEmb, p=2, dim=2)

        #vs_emb = torch.cat((ontoVEmb, EHRVEmb), dim=2)
        #dxdrugEmb = torch.cat((EHREmb,ontoEmb), dim=3)
#        print(np.array(ontoVEmb.cpu()).shape,np.array(ontoEmb.cpu()).shape,np.array(EHRVEmb.cpu()).shape,np.array(EHREmb.cpu()).shape)
       # print(ontoVEmb.cpu().detach().numpy().shape,ontoEmb.cpu().detach().numpy().shape,EHRVEmb.cpu().detach().numpy().shape,EHREmb.cpu().detach().numpy().shape)
        if self.use_ontology==False:
            vs_emb =  EHRVEmb
            dxdrugEmb = EHREmb
        else:
            vs_emb = torch.cat((ontoVEmb, EHRVEmb), dim=2)
            dxdrugEmb = torch.cat((EHREmb,ontoEmb), dim=3)
#        print(vs_emb.size(),dxdrugEmb.size())
#        vs_emb = torch.unsqueeze(vs_emb, dim=2).repeat(1,1,dx_num+drug_num,1)#change
        vs_emb = torch.unsqueeze(vs_emb, dim=2).repeat(1,1,dx_num+drug_num,1)
 #       print(vs_emb.size(),dxdrugEmb.size())
       # print(vs_emb.cpu().detach().numpy().shape,dxdrugEmb.cpu().detach().numpy().shape,dx_num+drug_num)
        #1/0
        attnInput = torch.cat((vs_emb, dxdrugEmb), dim=3)
        mlpOutput = torch.tanh(self.attn(attnInput))
#        print('pp',self.attn(attnInput),'r',attnInput)
        preAttention = self.attnCombine(mlpOutput)
        attention = F.softmax(preAttention, dim=2)
        self.visit_attentions=attention
        if self.use_ontology==False:
            vs_emb = torch.sum(torch.mul(attention.repeat(1,1,1,self.EHREmbDim), dxdrugEmb), 2)
        else:
            vs_emb = torch.sum(torch.mul(attention.repeat(1,1,1,self.EHREmbDim+self.ontoEmbDim), dxdrugEmb), 2)
        vs_emb_dp = F.normalize(vs_emb, p=2, dim=2)
        DP_result = F.softmax(self.dpPredLinear(torch.tanh(vs_emb_dp)), dim=2)
        read_result = torch.sigmoid(self.readPredLinear(vs_emb_dp))
        mort_result = torch.sigmoid(self.mortPredLinear(vs_emb_dp))
        return mort_result,read_result,DP_result, cooccur_loss*10
class DGI_model(nn.Module):
    def __init__(self, args,G,rel_num,use_ontology=False,use_icd=False,use_snomed=False,use_ctd=False):
        super(DGI_model, self).__init__()
        self.use_ontology=use_ontology
        self.use_icd=use_icd
        self.use_snomed=use_snomed
        self.use_ctd=use_ctd
        for k, v in args.__dict__.items():
            self.__setattr__(k, v)
        self.G=G
        #self.ontoEmb = dgl_layer.GCNLayer(rel_num,400,400)        
        if self.use_ontology==True:
            self.icdontoEmb =GATConv(self.EHREmbDim, self.EHREmbDim,1 ).to(self.device)
    #       self.snomedontoEmb =GATConv(400, 400,1 ).to(self.device)
            if self.use_snomed==True:
                self.snomedontoEmb =RelGraphConv(self.EHREmbDim, self.EHREmbDim, self.snomed_rel_num, layer_norm=True)
                self.snomedAnsDxEmb = nn.Embedding(self.snomed_dx_ans_num, self.EHREmbDim)
            self.ICDancEmb_list=[]
            for i in range(self.split_max_num):self.ICDancEmb_list.append(nn.Embedding(self.dxnumAncestors, self.EHREmbDim).to(self.device))
            if self.use_ctd==True:
                #self.ctdontoEmb = dgl_layer.GATLayer(G['ctd'],50,50).to(self.device)
                self.ctdontoEmb =GATConv(self.EHREmbDim, self.EHREmbDim,1 ).to(self.device)
    #        self.ICDancEmb2 = nn.Embedding(self.dxnumAncestors, self.EHREmbDim)
        '''
        self.icdontoEmb = dgl_layer.GATLayer(G[0],50,50).to(self.device)
      #  self.icdontoEmb = dgl.nn.pytorch.GAT(G,in_dim=400,hidden_dim=100,out_dim=400,num_heads=1)
        self.ctdontoEmb = dgl_layer.GATLayer(G[1],50,50).to(self.device)
        '''

#        self.dxontoEmb = layers.OntoEmb(self.dxVocabSize, self.dxnumAncestors, self.ontoEmbDim, self.ontoattnDim)        
 #       self.drugontoEmb = layers.OntoEmb(self.drugVocabSize, self.drugnumAncestors, self.ontoEmbDim, self.ontoattnDim)        

        self.dxALLontoEmb=None
        self.cooccurLinear = nn.Linear(self.EHREmbDim, self.dxVocabSize+self.drugVocabSize)
        self.dxproLinear = nn.Linear(2*self.EHREmbDim, self.EHREmbDim)
        self.ontoLinear = nn.Linear(self.EHREmbDim,self.EHREmbDim,bias=False )
        self.ctdLinear = nn.Linear(self.EHREmbDim,self.EHREmbDim,bias=False )
        self.onto_b = Variable(torch.zeros(self.EHREmbDim).to(self.device), requires_grad=True)
        self.EHRdxEmb = nn.Embedding(self.dxVocabSize+1, self.EHREmbDim, padding_idx=self.dxVocabSize)
        #self.ICDancEmb = nn.Embedding(self.dxnumAncestors, self.EHREmbDim)
        
        self.EHRdrugEmb = nn.Embedding(self.drugVocabSize+1, self.EHREmbDim, padding_idx=self.drugVocabSize)
        self.EHRdxEmb2 = nn.Embedding(self.dxVocabSize+1, self.EHREmbDim, padding_idx=self.dxVocabSize)
        self.EHRdrugEmb2 = nn.Embedding(self.drugVocabSize+1, self.EHREmbDim, padding_idx=self.drugVocabSize)
        self.gfeatures={}

        #self.gfeatures['extra_rx']=self.ICDancEmb(torch.LongTensor(range(self.drugnumAncestors)))
#        self.gfeatures['extra_rx']=self.ICDancEmb(torch.LongTensor(range(self.drugnumAncestors)))

#        self.dxrelationEmb = layers.SnomedEmb(self.dxVocabSize, self.dxrelationnumAncestors,  self.dxrelationnum,self.ontoEmbDim, self.ontoattnDim)        
        #self.dxrelationEmb = layers.SnomedEmb(self.EHRdxEmb,self.dxVocabSize, self.dxrelationnumAncestors,  self.dxrelationnum,self.ontoEmbDim, self.ontoattnDim)        
 #       self.drugrelationEmb = layers.SnomedEmb(self.drugVocabSize, self.drugrelationnumAncestors,  self.drugrelationnum,self.ontoEmbDim, self.ontoattnDim)        
        if self.use_ontology==False:
            self.attn = nn.Linear(2*self.EHREmbDim, self.ptattnDim)
            self.dpPredLinear = nn.Linear(self.EHREmbDim, self.dpLabelSize)
            self.readPredLinear = nn.Linear(self.EHREmbDim, 1)
            self.mortPredLinear = nn.Linear(self.EHREmbDim, 1)
        else:
            self.attn = nn.Linear(2*self.ontoEmbDim+2*self.EHREmbDim, self.ptattnDim)
            self.dpPredLinear = nn.Linear(self.ontoEmbDim+self.EHREmbDim, self.dpLabelSize)
            self.readPredLinear = nn.Linear(self.ontoEmbDim+self.EHREmbDim, 1)
            self.mortPredLinear = nn.Linear(self.ontoEmbDim+self.EHREmbDim, 1)
        self.attnCombine = nn.Linear(self.ptattnDim, 1)

    def onto_loss(self,extra_ents,ents):

        #ctd_loss=1*torch.sum((self.ctd_W(self.EHRdxEmb(ctd_dx))+self.ctd_b-self.ctd_W2(self.EHRdrugEmb(ctd_rx)))**2)
        #ctd_loss=1*torch.sum((self.EHRdxEmb(ctd_dx)+self.ctd_b-self.EHRdrugEmb(ctd_rx))**2)
        self.gfeatures['dx']=self.EHRdxEmb(torch.LongTensor(range(self.dxVocabSize)).to(self.device)).to(self.device)
        self.gfeatures['extra_dx']=self.ICDancEmb(torch.LongTensor(range(self.split_max_num*self.dxnumAncestors)).to(self.device)).to(self.device)
        self.gfeatures['rx']=self.EHRdrugEmb(torch.LongTensor(range(self.drugVocabSize)).to(self.device)).to(self.device)
#        icdembds2=self.ICDancEmb2(torch.LongTensor(range(self.dxnumAncestors)).to(self.device)).to(self.device)
            #dxALLontoEmb = self.ontoEmb(self.G,self.gfeatures)['dx']
            #concat_embs=torch.cat((self.gfeatures['dx'],self.gfeatures['extra_dx'],icdembds2))
        concat_embs=torch.cat((self.gfeatures['dx'],self.gfeatures['extra_dx']))
        #concat_embs=torch.cat((self.gfeatures['dx'],self.gfeatures['extra_dx'],icdembds2))
        #ontoloss=1*torch.sum((self.ontoLinear(concat_embs[extra_ents])-concat_embs[ents])**2)/extra_ents.size(0)
        ontoloss=1*torch.sum((concat_embs[extra_ents]-concat_embs[ents])**2)/extra_ents.size(0)
        #ctd_loss=1*torch.sum((self.ctd_W(self.EHRdxEmb(ctd_dx))-self.EHRdxEmb(ctd_dx))**2)
        return ontoloss
    def align_loss(self):
        #l=torch.sum((self.icd_dxAllontoEmb-self.ctd_dxALLontoEmb)**2)
        #l=torch.sum((self.icd_dxAllontoEmb-self.snomed_dxAllontoEmb)**2)
        l=torch.mean((self.icd_dxAllontoEmb-self.gfeatures['dx'])**2)*1000
        return l
    def forward(self, inputs):
#        (dxseqs, drugseqs, dx_onehot, drug_onehot, 
 #           dxLeavesList, dxAncestorsList, drugLeavesList, drugAncestorsList,dxrelationLeavesList,dxrelationAncestorsList,dxrelationList,dxrelation_permuteindex,drugrelationLeavesList,drugrelationAncestorsList,drugrelationList,drugrelation_permuteindex) = inputs

        (dxseqs, drugseqs, dx_onehot, drug_onehot) = inputs
        dxseqs = dxseqs.to(self.device)
        drugseqs = drugseqs.to(self.device)
        dx_onehot = dx_onehot.to(self.device)
        drug_onehot = drug_onehot.to(self.device)
 #       print(dxseqs,dxseqs.size())
#        1/0
        dx_num = list(dxseqs.size())[2]
        drug_num = list(drugseqs.size())[2]
#        print(self.EHRdxEmb.cpu(),self.dxVocabSize)


#        print(dxseqs.cpu(),np.array(dxseqs.cpu()).shape)
 #       print(np.array(self.EHRdxEmb.cpu()).shape)
  #      1/0
        #print(dxseqs,np.array(dxseqs.cpu()).shape)
        dxEHREmb = self.EHRdxEmb(dxseqs)
#        print(dxEHREmb.cpu().detach().numpy().shape)
 #       1/0
        drugEHREmb = self.EHRdrugEmb(drugseqs)
        #EHREmb = torch.cat((dxEHREmb, drugEHREmb), dim=2)##change
#        EHREmb = dxEHREmb

        EHREmb = torch.cat((dxEHREmb, drugEHREmb), dim=2)##change
#        print('ddddddd',self.EHRdxEmb,self.gfeatures['dx'])
        if self.use_ontology==True:
#            aa=nn.Linear(400,400)

            self.gfeatures['dx']=self.EHRdxEmb(torch.LongTensor(range(self.dxVocabSize)).to(self.device)).to(self.device)
#            self.gfeatures['extra_dx']=self.ICDancEmb(torch.LongTensor(range(self.split_max_num*self.dxnumAncestors)).to(self.device)).to(self.device)
            self.gfeatures['extra_dx']=[]
            for i in range(self.split_max_num):
                sid=torch.LongTensor(range(self.dxnumAncestors)).to(self.device)
                self.gfeatures['extra_dx'].append(self.ICDancEmb_list[i](sid).to(self.device))

    #        print(self.gfeatures['extra_dx'].cpu().detach().numpy()[:5],'\n')
           # print(self.EHRdxEmb.weight,'\n')
           # print(self.gfeatures['dx'].cpu().detach().numpy()[:5])
            self.gfeatures['rx']=self.EHRdrugEmb(torch.LongTensor(range(self.drugVocabSize)).to(self.device)).to(self.device)
            self.gfeatures['dx2']=self.EHRdxEmb2(torch.LongTensor(range(self.dxVocabSize)).to(self.device)).to(self.device)
            self.gfeatures['rx2']=self.EHRdrugEmb2(torch.LongTensor(range(self.drugVocabSize)).to(self.device)).to(self.device)
        #    icdembds2=self.ICDancEmb2(torch.LongTensor(range(self.dxnumAncestors)).to(self.device)).to(self.device)
            #dxALLontoEmb = self.ontoEmb(self.G,self.gfeatures)['dx']
            #concat_embs=torch.cat((self.gfeatures['dx'],self.gfeatures['extra_dx'],icdembds2)).to(self.device)
            icd_emb_list=[self.gfeatures['dx']]+self.gfeatures['extra_dx']
            #concat_embs=torch.cat((self.gfeatures['dx'],self.gfeatures['extra_dx'])).to(self.device)
            concat_embs=torch.cat(icd_emb_list).to(self.device)
            
#            concat_embs=torch.cat((self.gfeatures['dx'],self.gfeatures['extra_dx'])).to(self.device)
            #concat_embs=torch.cat((self.gfeatures['dx'],self.gfeatures['extra_dx'],self.gfeatures['rx']))
 #           print('ddddddd',self.gfeatures['dx'].shape,concat_embs.shape)
#            1/0
            #dxALLontoEmb = self.icdontoEmb(concat_embs)[:self.dxVocabSize]
            
#            print(self.G[0].edges(),self.icd_edges_attentions.shape)
 #           1/0
            #print(self.G[2][1].size(),self.G[2][0].num_edges())
            ''' 
#            dxALLontoEmb = self.icd_dxAllontoEmb
#            1/0
            ''' 
            if self.use_icd==True:
                self.icd_dxAllontoEmb=torch.squeeze(self.icdontoEmb(self.G['icd'],concat_embs)[:self.dxVocabSize])
                self.icd_edges_attentions=self.G['icd'].edata['a'].cpu().detach().numpy()
                #dxALLontoEmb = F.normalize(self.icd_dxAllontoEmb, p=2, dim=1)
                dxALLontoEmb = self.icd_dxAllontoEmb
            else:
                dxALLontoEmb = 0
            if self.use_ctd==True:
                ctd_concat_embs=torch.cat((self.gfeatures['dx2'],self.gfeatures['rx2'])).to(self.device)
                ctd_embs=torch.squeeze(self.ctdontoEmb(self.G['ctd'],ctd_concat_embs))
                self.ctd_dxALLontoEmb = ctd_embs[:self.dxVocabSize]
                self.ctd_drugALLontoEmb = ctd_embs[self.dxVocabSize:]
                ctd_rxALLontoEmb = ctd_embs[self.dxVocabSize:]
            #dxALLontoEmb = (self.icd_dxAllontoEmb+self.ctd_dxALLontoEmb)/2
                dxALLontoEmb = dxALLontoEmb+self.ctd_dxALLontoEmb
#            dxALLontoEmb = (self.icd_dxAllontoEmb+self.snomed_dxAllontoEmb)/2
            
            if self.use_snomed==True:
                self.gfeatures['snomed_extra_dx']=self.snomedAnsDxEmb(torch.LongTensor(range(self.snomed_dx_ans_num)).to(self.device)).to(self.device)
                snomed_concat_embs=torch.cat((self.gfeatures['dx'],self.gfeatures['snomed_extra_dx'])).to(self.device)
                '''
                if self.use_icd==True:
                    snomed_concat_embs=torch.cat((dxALLontoEmb,self.gfeatures['snomed_extra_dx'])).to(self.device)#deep
                else:
                    snomed_concat_embs=torch.cat((self.gfeatures['dx'],self.gfeatures['snomed_extra_dx'])).to(self.device)
                '''
                self.snomed_dxAllontoEmb=torch.squeeze(self.snomedontoEmb(self.G['snomed'][0],snomed_concat_embs,self.G['snomed'][1])[:self.dxVocabSize])
                self.snomed_extraEmb=torch.squeeze(self.snomedontoEmb(self.G['snomed'][0],snomed_concat_embs,self.G['snomed'][1])[self.dxVocabSize:])
#                dxALLontoEmb =dxALLontoEmb+self.snomed_dxAllontoEmb

                dxALLontoEmb =dxALLontoEmb+ F.normalize(self.snomed_dxAllontoEmb, p=2, dim=1)
                #dxALLontoEmb =self.snomed_dxAllontoEmb
            #dxALLontoEmb = self.dxproLinear(torch.cat((self.icd_dxAllontoEmb,self.snomed_dxAllontoEmb),axis=1))
            #print('ddddddd',self.ontoEmb.rel_Ws,dxALLontoEmb,self.gfeatures['dx'])
#            print('ddddddd',self.ontoEmb.rel_Ws.weight)
#            print(aa.weight)
            new_row = torch.Tensor(np.zeros(shape=(1, self.EHREmbDim))).to(self.device)
#            print(dxALLontoEmb.size())
 #           1/0
            dxALLontoEmb = torch.cat((dxALLontoEmb, new_row)).to(self.device)
         #   drugALLontoEmb = torch.cat((ctd_rxALLontoEmb, new_row)).to(self.device)

        #    drugALLontoEmb = self.ontoEmb(self.G,self.gfeatures)['rx']
            drugALLontoEmb = self.EHRdrugEmb
            #drugALLontoEmb = self.ctd_drugALLontoEmb
            #drugALLontoEmb =self.EHRdrugEmb(drugseqs)

        '''
        dxrelationInputs = (dxALLontoEmb,dxrelationLeavesList,dxrelationAncestorsList,dxrelationList,dxrelation_permuteindex)
        drugrelationInputs = (drugALLontoEmb,drugrelationLeavesList,drugrelationAncestorsList,drugrelationList,drugrelation_permuteindex)
        dxrelationALLontoEmb = self.dxrelationEmb(dxrelationInputs)
        drugrelationALLontoEmb = self.drugrelationEmb(drugrelationInputs)
        '''
    #    dxALLontoEmb=0.5*dxALLontoEmb+0.5*dxrelationALLontoEmb
        #dxALLontoEmb=dxALLontoEmb+dxrelationALLontoEmb##chanfe

#        if use_ontology==True:
 #           dxALLontoEmb=dxALLontoEmb##chanfe
            #dxALLontoEmb=dxALLontoEmb
#            drugALLontoEmb=drugALLontoEmb
        #drugALLontoEmb=drugALLontoEmb+drugrelationALLontoEmb
        #dxALLontoEmb=dxrelationALLontoEmb
        #drugALLontoEmb=drugrelationALLontoEmb
        #print(np.array(dxALLontoEmb.cpu()).shape,np.array(dxEHRVEmb.cpu()).shape)
        '''
        print(dxseqs)
        print(dxALLontoEmb.size(),dxEHREmb.size(),dxseqs.size())
        1/0
        '''
        if self.use_ontology==True:
            self.dxALLontoEmb=dxALLontoEmb
            dxOntoEmb = dxALLontoEmb[dxseqs]
            #drugOntoEmb = drugALLontoEmb[drugseqs]
            drugOntoEmb = drugALLontoEmb(drugseqs)
            ontoEmb = torch.cat((dxOntoEmb, drugOntoEmb), dim=2)
#            drugOntoEmb = drugALLontoEmb[drugseqs]
            #ontoEmb = torch.cat((dxOntoEmb, drugOntoEmb), dim=2)
         #   ontoEmb = dxOntoEmb
        '''
        dxrelationOntoEmb = dxrelationALLontoEmb[dxseqs]
        drugrelationOntoEmb = drugrelationALLontoEmb[drugseqs]
        '''



        #relontoEmb = torch.cat((dxrelationOntoEmb, drugrelationOntoEmb), dim=2)
#        print(ontoEmb.size(),relontoEmb.size())

        dxEHRVEmb = F.normalize(torch.sum(self.EHRdxEmb(dxseqs), dim=2), p=2, dim=2)
        drugEHRVEmb= F.normalize(torch.sum(self.EHRdrugEmb(drugseqs), dim=2), p=2, dim=2)

        EHRVEmb = dxEHRVEmb+drugEHRVEmb##chan
        #EHRVEmb = dxEHRVEmb##chan
        if self.use_ontology==True:
            dxontoVEmb = torch.matmul(dx_onehot.permute(1,0,2), dxALLontoEmb[:-1])
            #dxrelontoVEmb = torch.matmul(dx_onehot.permute(1,0,2), dxrelationALLontoEmb[:-1])
#            drugontoVEmb = torch.matmul(drug_onehot.permute(1,0,2), drugALLontoEmb[:-1])
        #drugrelontoVEmb = torch.matmul(drug_onehot.permute(1,0,2), drugrelationALLontoEmb[:-1])


        t=self.cooccurLinear(EHRVEmb)
        cooccurU = F.softmax(self.cooccurLinear(EHRVEmb), dim=2).contiguous()
        vonehot = torch.cat((dx_onehot.permute(1,0,2), drug_onehot.permute(1,0,2)), dim=2).contiguous()
        cooccur_loss = multi_class_cross_entropy_loss(cooccurU, vonehot)
        if self.use_ontology==True:
            #ontoVEmb = F.normalize(dxontoVEmb, p=2, dim=2)+F.normalize(drugontoVEmb, p=2, dim=2)
            ontoVEmb = F.normalize(dxontoVEmb, p=2, dim=2)

#        relontoVEmb = F.normalize(dxrelontoVEmb, p=2, dim=2)+F.normalize(drugrelontoVEmb, p=2, dim=2)

        #vs_emb = torch.cat((ontoVEmb, EHRVEmb), dim=2)
        #dxdrugEmb = torch.cat((EHREmb,ontoEmb), dim=3)
#        print(np.array(ontoVEmb.cpu()).shape,np.array(ontoEmb.cpu()).shape,np.array(EHRVEmb.cpu()).shape,np.array(EHREmb.cpu()).shape)
       # print(ontoVEmb.cpu().detach().numpy().shape,ontoEmb.cpu().detach().numpy().shape,EHRVEmb.cpu().detach().numpy().shape,EHREmb.cpu().detach().numpy().shape)
        if self.use_ontology==False:
            vs_emb =  EHRVEmb
            dxdrugEmb = EHREmb
        else:
            vs_emb = torch.cat((ontoVEmb, EHRVEmb), dim=2)
            dxdrugEmb = torch.cat((EHREmb,ontoEmb), dim=3)
#        print(vs_emb.size(),dxdrugEmb.size())
#        vs_emb = torch.unsqueeze(vs_emb, dim=2).repeat(1,1,dx_num+drug_num,1)#change
        vs_emb = torch.unsqueeze(vs_emb, dim=2).repeat(1,1,dx_num+drug_num,1)
 #       print(vs_emb.size(),dxdrugEmb.size())
       # print(vs_emb.cpu().detach().numpy().shape,dxdrugEmb.cpu().detach().numpy().shape,dx_num+drug_num)
        #1/0
        attnInput = torch.cat((vs_emb, dxdrugEmb), dim=3)
        mlpOutput = torch.tanh(self.attn(attnInput))
#        print('pp',self.attn(attnInput),'r',attnInput)
        preAttention = self.attnCombine(mlpOutput)
        attention = F.softmax(preAttention, dim=2)
        self.visit_attentions=attention
        if self.use_ontology==False:
            vs_emb = torch.sum(torch.mul(attention.repeat(1,1,1,self.EHREmbDim), dxdrugEmb), 2)
        else:
            vs_emb = torch.sum(torch.mul(attention.repeat(1,1,1,self.EHREmbDim+self.ontoEmbDim), dxdrugEmb), 2)
        vs_emb_dp = F.normalize(vs_emb, p=2, dim=2)
        DP_result = F.softmax(self.dpPredLinear(torch.tanh(vs_emb_dp)), dim=2)
        read_result = torch.sigmoid(self.readPredLinear(vs_emb_dp))
        mort_result = torch.sigmoid(self.mortPredLinear(vs_emb_dp))
        return mort_result,read_result,DP_result, cooccur_loss*10

class MMORE_DXRX(nn.Module):
    def __init__(self, args):
        super(MMORE_DXRX, self).__init__()
        for k, v in args.__dict__.items():
            self.__setattr__(k, v)
        self.dxontoEmb = layers.OntoEmb(self.dxVocabSize, self.dxnumAncestors, self.ontoEmbDim, self.ontoattnDim)        
        self.drugontoEmb = layers.OntoEmb(self.drugVocabSize, self.drugnumAncestors, self.ontoEmbDim, self.ontoattnDim)        
        self.relationEmb = layers.SnomedEmb(0, 0, 0 ,0, 0)        

        self.cooccurLinear = nn.Linear(self.EHREmbDim, self.dxVocabSize+self.drugVocabSize)
        self.EHRdxEmb = nn.Embedding(self.dxVocabSize+1, self.EHREmbDim, padding_idx=self.dxVocabSize)
        self.EHRdrugEmb = nn.Embedding(self.drugVocabSize+1, self.EHREmbDim, padding_idx=self.drugVocabSize)

#        self.dxrelationEmb = layers.SnomedEmb(self.dxVocabSize, self.dxrelationnumAncestors,  self.dxrelationnum,self.ontoEmbDim, self.ontoattnDim)        
        #self.dxrelationEmb = layers.SnomedEmb(self.EHRdxEmb,self.dxVocabSize, self.dxrelationnumAncestors,  self.dxrelationnum,self.ontoEmbDim, self.ontoattnDim)        
 #       self.drugrelationEmb = layers.SnomedEmb(self.drugVocabSize, self.drugrelationnumAncestors,  self.drugrelationnum,self.ontoEmbDim, self.ontoattnDim)        
        if use_ontology==False:
            self.attn = nn.Linear(2*self.EHREmbDim, self.ptattnDim)
            self.dpPredLinear = nn.Linear(self.EHREmbDim, self.dpLabelSize)
        else:
            self.attn = nn.Linear(2*self.ontoEmbDim+2*self.EHREmbDim, self.ptattnDim)
            self.dpPredLinear = nn.Linear(self.ontoEmbDim+self.EHREmbDim, self.dpLabelSize)
        self.attnCombine = nn.Linear(self.ptattnDim, 1)

    def forward(self, inputs):
#        (dxseqs, drugseqs, dx_onehot, drug_onehot, 
 #           dxLeavesList, dxAncestorsList, drugLeavesList, drugAncestorsList,dxrelationLeavesList,dxrelationAncestorsList,dxrelationList,dxrelation_permuteindex,drugrelationLeavesList,drugrelationAncestorsList,drugrelationList,drugrelation_permuteindex) = inputs

        (dxseqs, drugseqs, dx_onehot, drug_onehot,dxLeavesList, dxAncestorsList, drugLeavesList, drugAncestorsList) = inputs
        dxontoInputs = (dxLeavesList, dxAncestorsList)
        drugontoInputs = (drugLeavesList, drugAncestorsList)
        dxseqs = dxseqs.to(self.device)
        drugseqs = drugseqs.to(self.device)
        dx_onehot = dx_onehot.to(self.device)
        drug_onehot = drug_onehot.to(self.device)
 #       print(dxseqs,dxseqs.size())
#        1/0
        dx_num = list(dxseqs.size())[2]
        drug_num = list(drugseqs.size())[2]
#        print(self.EHRdxEmb.cpu(),self.dxVocabSize)


#        print(dxseqs.cpu(),np.array(dxseqs.cpu()).shape)
 #       print(np.array(self.EHRdxEmb.cpu()).shape)
  #      1/0
        #print(dxseqs,np.array(dxseqs.cpu()).shape)
        dxEHREmb = self.EHRdxEmb(dxseqs)
#        print(dxEHREmb.cpu().detach().numpy().shape)
 #       1/0
        drugEHREmb = self.EHRdrugEmb(drugseqs)
        #EHREmb = torch.cat((dxEHREmb, drugEHREmb), dim=2)##change
        EHREmb = dxEHREmb
        if use_ontology==True:
            dxALLontoEmb = self.dxontoEmb(dxontoInputs)
            drugALLontoEmb = self.drugontoEmb(drugontoInputs)


        '''
        dxrelationInputs = (dxALLontoEmb,dxrelationLeavesList,dxrelationAncestorsList,dxrelationList,dxrelation_permuteindex)
        drugrelationInputs = (drugALLontoEmb,drugrelationLeavesList,drugrelationAncestorsList,drugrelationList,drugrelation_permuteindex)
        dxrelationALLontoEmb = self.dxrelationEmb(dxrelationInputs)
        drugrelationALLontoEmb = self.drugrelationEmb(drugrelationInputs)
        '''
    #    dxALLontoEmb=0.5*dxALLontoEmb+0.5*dxrelationALLontoEmb
        #dxALLontoEmb=dxALLontoEmb+dxrelationALLontoEmb##chanfe

        if use_ontology==True:
            dxALLontoEmb=dxALLontoEmb##chanfe
            #dxALLontoEmb=dxALLontoEmb
            drugALLontoEmb=drugALLontoEmb
        #drugALLontoEmb=drugALLontoEmb+drugrelationALLontoEmb
        #dxALLontoEmb=dxrelationALLontoEmb
        #drugALLontoEmb=drugrelationALLontoEmb
        #print(np.array(dxALLontoEmb.cpu()).shape,np.array(dxEHRVEmb.cpu()).shape)
        '''
        print(dxseqs)
        print(dxALLontoEmb.size(),dxEHREmb.size(),dxseqs.size())
        1/0
        '''
        if use_ontology==True:
            dxOntoEmb = dxALLontoEmb[dxseqs]
            drugOntoEmb = drugALLontoEmb[drugseqs]
            ontoEmb = torch.cat((dxOntoEmb, drugOntoEmb), dim=2)
        '''
        dxrelationOntoEmb = dxrelationALLontoEmb[dxseqs]
        drugrelationOntoEmb = drugrelationALLontoEmb[drugseqs]
        '''



        #relontoEmb = torch.cat((dxrelationOntoEmb, drugrelationOntoEmb), dim=2)
#        print(ontoEmb.size(),relontoEmb.size())

        dxEHRVEmb = F.normalize(torch.sum(self.EHRdxEmb(dxseqs), dim=2), p=2, dim=2)
        drugEHRVEmb= F.normalize(torch.sum(self.EHRdrugEmb(drugseqs), dim=2), p=2, dim=2)

        #EHRVEmb = dxEHRVEmb+drugEHRVEmb##chan
        EHRVEmb = dxEHRVEmb##chan
        if use_ontology==True:
            dxontoVEmb = torch.matmul(dx_onehot.permute(1,0,2), dxALLontoEmb[:-1])
            #dxrelontoVEmb = torch.matmul(dx_onehot.permute(1,0,2), dxrelationALLontoEmb[:-1])
            drugontoVEmb = torch.matmul(drug_onehot.permute(1,0,2), drugALLontoEmb[:-1])
        #drugrelontoVEmb = torch.matmul(drug_onehot.permute(1,0,2), drugrelationALLontoEmb[:-1])


        t=self.cooccurLinear(EHRVEmb)
        cooccurU = F.softmax(self.cooccurLinear(EHRVEmb), dim=2).contiguous()
        vonehot = torch.cat((dx_onehot.permute(1,0,2), drug_onehot.permute(1,0,2)), dim=2).contiguous()
        cooccur_loss = multi_class_cross_entropy_loss(cooccurU, vonehot)
        if use_ontology==True:
            ontoVEmb = F.normalize(dxontoVEmb, p=2, dim=2)+F.normalize(drugontoVEmb, p=2, dim=2)
#        relontoVEmb = F.normalize(dxrelontoVEmb, p=2, dim=2)+F.normalize(drugrelontoVEmb, p=2, dim=2)

        #vs_emb = torch.cat((ontoVEmb, EHRVEmb), dim=2)
        #dxdrugEmb = torch.cat((EHREmb,ontoEmb), dim=3)
#        print(np.array(ontoVEmb.cpu()).shape,np.array(ontoEmb.cpu()).shape,np.array(EHRVEmb.cpu()).shape,np.array(EHREmb.cpu()).shape)
       # print(ontoVEmb.cpu().detach().numpy().shape,ontoEmb.cpu().detach().numpy().shape,EHRVEmb.cpu().detach().numpy().shape,EHREmb.cpu().detach().numpy().shape)
        if use_ontology==False:
            vs_emb =  EHRVEmb
            dxdrugEmb = EHREmb
        else:
            vs_emb = torch.cat((ontoVEmb, EHRVEmb), dim=2)
            dxdrugEmb = torch.cat((EHREmb,ontoEmb), dim=3)
#        print(vs_emb.size(),dxdrugEmb.size())
#        vs_emb = torch.unsqueeze(vs_emb, dim=2).repeat(1,1,dx_num+drug_num,1)#change
        vs_emb = torch.unsqueeze(vs_emb, dim=2).repeat(1,1,dx_num,1)
 #       print(vs_emb.size(),dxdrugEmb.size())
       # print(vs_emb.cpu().detach().numpy().shape,dxdrugEmb.cpu().detach().numpy().shape,dx_num+drug_num)
        #1/0
        attnInput = torch.cat((vs_emb, dxdrugEmb), dim=3)
        mlpOutput = torch.tanh(self.attn(attnInput))
        preAttention = self.attnCombine(mlpOutput)
        attention = F.softmax(preAttention, dim=2)
        if use_ontology==False:
            vs_emb = torch.sum(torch.mul(attention.repeat(1,1,1,self.EHREmbDim), dxdrugEmb), 2)
        else:
            vs_emb = torch.sum(torch.mul(attention.repeat(1,1,1,self.EHREmbDim+self.ontoEmbDim), dxdrugEmb), 2)
        vs_emb_dp = F.normalize(vs_emb, p=2, dim=2)
        DP_result = F.softmax(self.dpPredLinear(torch.tanh(vs_emb_dp)), dim=2)
        return DP_result, cooccur_loss*10
    
class MMORE_DX(nn.Module):
    def __init__(self, args):
        super(MMORE_DX, self).__init__()
        for k, v in args.__dict__.items():
            self.__setattr__(k, v)
        self.dxontoEmb = layers.OntoEmb(self.dxVocabSize, self.dxnumAncestors, self.ontoEmbDim, self.ontoattnDim)        
        self.EHRdxEmb = nn.Embedding(self.dxVocabSize+1, self.EHREmbDim, padding_idx=self.dxVocabSize)
        self.attn = nn.Linear(2*self.EHREmbDim+2*self.ontoEmbDim, self.ptattnDim)
        self.attnCombine = nn.Linear(self.ptattnDim, 1)
        self.cooccurLinear = nn.Linear(self.EHREmbDim, self.dxVocabSize)
        self.dpPredLinear = nn.Linear(self.ontoEmbDim+self.EHREmbDim, self.dpLabelSize)

    def forward(self, inputs):
        (dxseqs, dx_onehot, dxLeavesList, dxAncestorsList,) = inputs
        dxontoInputs = (dxLeavesList, dxAncestorsList)
        dxseqs = dxseqs.to(self.device)
        dx_onehot = dx_onehot.to(self.device)
        dx_num = list(dxseqs.size())[2]
        dxALLontoEmb = self.dxontoEmb(dxontoInputs)
        dxOntoEmb = dxALLontoEmb[dxseqs]
        dxontoVEmb = torch.matmul(dx_onehot.permute(1,0,2), dxALLontoEmb[:-1])
        ontoVEmb = dxontoVEmb
        dxEHREmb = self.EHRdxEmb(dxseqs)
        dxEHRVEmb = F.normalize(torch.sum(self.EHRdxEmb(dxseqs), dim=2), p=2, dim=2)
        EHRVEmb = dxEHRVEmb
        cooccurU = F.softmax(self.cooccurLinear(EHRVEmb), dim=2).contiguous()
        vonehot = dx_onehot.permute(1,0,2).contiguous()
        cooccur_loss = multi_class_cross_entropy_loss(cooccurU, vonehot)
        vs_emb = torch.cat((ontoVEmb, EHRVEmb), dim=2)
        dxEmb = torch.cat((dxEHREmb, dxOntoEmb), dim=3)
        vs_emb = torch.unsqueeze(vs_emb, dim=2).repeat(1,1,dx_num,1)
        attnInput = torch.cat((vs_emb, dxEmb), dim=3)
        mlpOutput = torch.tanh(self.attn(attnInput))
        preAttention = self.attnCombine(mlpOutput)
        attention = F.softmax(preAttention, dim=2)
        vs_emb = torch.sum(torch.mul(attention.repeat(1,1,1,self.EHREmbDim+self.ontoEmbDim), dxEmb), dim=2)
        vs_emb_dp = F.normalize(vs_emb, p=2, dim=2)
        DP_result = F.softmax(self.dpPredLinear(torch.tanh(vs_emb_dp)), dim=2) 
        return DP_result, cooccur_loss*10
class MMORE_GAT(nn.Module):
    def __init__(self, args):
        super(MMORE_GAT, self).__init__()
        for k, v in args.__dict__.items():
            self.__setattr__(k, v)

        self.dxontoEmb = layers.OntoEmb(self.dxVocabSize, self.dxnumAncestors, self.ontoEmbDim, self.ontoattnDim)        
        self.drugontoEmb = layers.OntoEmb(self.drugVocabSize, self.drugnumAncestors, self.ontoEmbDim, self.ontoattnDim)        
       # self.relationEmb = layers.SnomedEmb(0, 0, 0 ,0, 0)        

        self.ctd_W = nn.Linear(self.EHREmbDim, self.EHREmbDim)
        self.ctd_W2 = nn.Linear(self.EHREmbDim, self.EHREmbDim)
        self.ctd_b = Variable(torch.zeros(self.EHREmbDim).to(self.device), requires_grad=True)
        self.cooccurLinear = nn.Linear(self.EHREmbDim, self.dxVocabSize+self.drugVocabSize)
        self.EHRdxEmb = nn.Embedding(self.dxVocabSize+1, self.EHREmbDim, padding_idx=self.dxVocabSize)
        self.EHRdrugEmb = nn.Embedding(self.drugVocabSize+1, self.EHREmbDim, padding_idx=self.drugVocabSize)
        self.medattn = nn.Linear(2*self.EHREmbDim, self.ptattnDim)
        if use_ontology==False:
            self.attn = nn.Linear(2*self.EHREmbDim, self.ptattnDim)
            self.dpPredLinear = nn.Linear(self.EHREmbDim, self.dpLabelSize)
            self.readPredLinear = nn.Linear(self.EHREmbDim, 1)
        else:
            self.attn = nn.Linear(2*self.ontoEmbDim+2*self.EHREmbDim, self.ptattnDim)
            self.dpPredLinear = nn.Linear(self.ontoEmbDim+self.EHREmbDim, self.dpLabelSize)
            self.readPredLinear = nn.Linear(self.ontoEmbDim+self.EHREmbDim, 1)
        self.attnCombine = nn.Linear(self.ptattnDim, 1)
        self.medattnCombine = nn.Linear(self.ptattnDim, 1)
        self.medQ = nn.Linear(self.EHREmbDim, self.EHREmbDim)
        self.medV = nn.Linear(self.EHREmbDim, self.EHREmbDim)
        self.medK = nn.Linear(self.EHREmbDim, self.EHREmbDim)

        self.ctdDxEmb = layers.SnomedEmb(self.dxVocabSize, self.drugVocabSize,  1,self.ontoEmbDim, self.ontoattnDim,dxEmb=self.EHRdxEmb,dxAnEmb=self.EHRdrugEmb)        
        self.ctdRxEmb = layers.SnomedEmb(self.drugVocabSize, self.dxVocabSize,  1,self.ontoEmbDim, self.ontoattnDim,dxEmb=self.EHRdrugEmb,dxAnEmb=self.EHRdxEmb)        
        self.ctd_done=False

        '''
        iin=torch.LongTensor(range(self.dxVocabSize+1))
        iin = iin.to('cuda')
        self.EHRdxEmb=self.EHRdxEmb.to('cuda')
        self.medattn=self.medattn.to('cuda')
        self.medattnCombine=self.medattnCombine.to('cuda')
        EHRdxEmb=self.EHRdxEmb(iin)
#        iin=iin.to(self.device)
#        print(EHRdxEmb.size())
        queryEHRdxEmb=EHRdxEmb[:,:,None]
        keyEHRdxEmb=EHRdxEmb[:,None,:]
        query=queryEHRdxEmb.repeat(1,1,self.dxVocabSize+1).permute(0,2,1)
        key=keyEHRdxEmb.repeat(1,self.dxVocabSize+1,1)
        attnInput = torch.cat((key, query), dim=2)
        mlpOutput = torch.tanh(self.medattn(attnInput))
        preAttention = self.medattnCombine(mlpOutput)
        attention = F.softmax(preAttention, dim=2)
        self.dxEHREmb = torch.sum(torch.mul(attention.repeat(1,1,self.EHREmbDim), key), 1)
        '''
    def ctd_loss(self,ctd_dx,ctd_rx):

        #ctd_loss=1*torch.sum((self.ctd_W(self.EHRdxEmb(ctd_dx))+self.ctd_b-self.ctd_W2(self.EHRdrugEmb(ctd_rx)))**2)
        #ctd_loss=1*torch.sum((self.EHRdxEmb(ctd_dx)+self.ctd_b-self.EHRdrugEmb(ctd_rx))**2)
        ctd_loss=1*torch.sum((self.EHRdxEmb(ctd_dx)+self.ctd_b-self.EHRdrugEmb(ctd_rx))**2)
        #ctd_loss=1*torch.sum((self.ctd_W(self.EHRdxEmb(ctd_dx))-self.EHRdxEmb(ctd_dx))**2)
        return ctd_loss


    def forward(self, inputs):

        #(dxseqs, drugseqs, dx_onehot, drug_onehot,dxLeavesList, dxAncestorsList, drugLeavesList, drugAncestorsList,ctd_dx,ctd_rx) = inputs
        (dxseqs, drugseqs, dx_onehot, drug_onehot,dxLeavesList, dxAncestorsList, drugLeavesList, drugAncestorsList,ctd_dx_leaves_list,ctd_dx_ancesster_list,ctd_dx_rel_list,ctd_dx_permute_list,ctd_rx_leaves_list,ctd_rx_ancesster_list,ctd_rx_rel_list,ctd_rx_permute_list)=inputs

        
        dxontoInputs = (dxLeavesList, dxAncestorsList)


        ctdDxInputs = (self.EHRdxEmb,ctd_dx_leaves_list,ctd_dx_ancesster_list,ctd_dx_rel_list,ctd_dx_permute_list)
        ctdRxInputs = (self.EHRdrugEmb,ctd_rx_leaves_list,ctd_rx_ancesster_list,ctd_rx_rel_list,ctd_rx_permute_list)
        #print(np.array(ctd_dx_leaves_list[1].shape),np.array(ctd_dx_ancesster_list[1].shape),np.array(ctd_dx_rel_list[1].shape))
#        1/0
        dxseqs = dxseqs.to(self.device)

        drugontoInputs = (drugLeavesList, drugAncestorsList)
        dxseqs = dxseqs.to(self.device)
        drugseqs = drugseqs.to(self.device)
        dx_onehot = dx_onehot.to(self.device)
        drug_onehot = drug_onehot.to(self.device)
 #       print(dxseqs,dxseqs.size())
#        1/0

        dx_num = list(dxseqs.size())[2]
        drug_num = list(drugseqs.size())[2]
        if use_ctd==True:
            if self.ctd_done==False:
                print(self.ctd_done)
                self.ctdDxALLontoEmb = self.ctdDxEmb(ctdDxInputs)
                self.ctdRxALLontoEmb = self.ctdRxEmb(ctdRxInputs)
                self.ctd_done=True
    #        print(self.ctd_done)
            ctdDxOntoVEmb = F.normalize(torch.sum(self.ctdDxALLontoEmb[dxseqs], dim=2), p=2, dim=2)
            ctdRxOntoVEmb = F.normalize(torch.sum(self.ctdRxALLontoEmb[drugseqs], dim=2), p=2, dim=2)

#        ctdDxOntoVEmb = torch.matmul(dx_onehot.permute(1,0,2), ctdDxALLontoEmb[:-1])
        #print(dxseqs.size())
        #1/0

#        print(self.EHRdxEmb.cpu(),self.dxVocabSize)


#        print(dxseqs.cpu(),np.array(dxseqs.cpu()).shape)
 #       print(np.array(self.EHRdxEmb.cpu()).shape)
  #      1/0
        #print(dxseqs,np.array(dxseqs.cpu()).shape)
#        print(self.EHRdxEmb.size())
        ##med graph
        '''
        iin=torch.LongTensor(range(self.dxVocabSize+1))
        iin = iin.to('cuda')
        EHRdxEmb=self.EHRdxEmb(iin)
#        iin=iin.to(self.device)
        print(EHRdxEmb.size())
        EHRdxEmb=EHRdxEmb[:,:,None]
        query=EHRdxEmb.repeat(1,1,self.dxVocabSize+1)
        print(query.size())
        1/0
        dxEHREmb = self.EHRdxEmb(dxseqs)
        ##med graph attention!!
        td = dxEHREmb[:,:,:,:,None].permute(0,1,4,2,3)
#        print(td.size())
        query=td.repeat(1,1,dx_num,1,1)
        key=td.permute(0,1,3,2,4).repeat(1,1,1,dx_num,1)
#        print(key.cpu().detach().numpy()[0,0,0,:3])
        #print(query.cpu().detach().numpy()[0,0,0,:3])
 #       1/0
        dxmask=dxseqs.clone()
        dxmask[dxseqs!=self.dxVocabSize]=1
        dxmask[dxseqs==self.dxVocabSize]=0
#        print(dxmask.cpu().detach().numpy())
        dxmask=dxmask[:,:,:,None,None].permute(0,1,3,2,4).repeat(1,1,dx_num,1,1)
        attnInput = torch.cat((self.medK(key), self.medQ(query)), dim=4)
        mlpOutput = torch.tanh(self.medattn(attnInput))
        preAttention = self.medattnCombine(mlpOutput)
        preAttention[dxmask==0] = -999999999
#        print(preAttention.cpu().detach().numpy()[0,0,0])
        attention = F.softmax(preAttention, dim=3)
 #       print(attention.cpu().detach().numpy()[0,0,0])
 #       print(preAttention.size(),attention.size())
        dxEHREmb = torch.sum(torch.mul(attention.repeat(1,1,1,1,self.EHREmbDim), self.medV(query)), 3)
        '''
        ##global graph
        

        #dxEHREmb = self.dxEHREmb[dxseqs]
        dxEHREmb = self.EHRdxEmb(dxseqs)
  #      print(dxEHREmb.size())
   #     1/0
#        print(dxEHREmb.cpu().detach().numpy().shape)
 #       1/0
        drugEHREmb = self.EHRdrugEmb(drugseqs)
        EHREmb = torch.cat((dxEHREmb, drugEHREmb), dim=2)##change
        #EHREmb = dxEHREmb
        if use_ontology==True:
            dxALLontoEmb = self.dxontoEmb(dxontoInputs)
            drugALLontoEmb = self.drugontoEmb(drugontoInputs)




        if use_ontology==True:
            dxALLontoEmb=dxALLontoEmb##chanfe
            #dxALLontoEmb=dxALLontoEmb
            drugALLontoEmb=drugALLontoEmb

        if use_ontology==True:
            dxOntoEmb = dxALLontoEmb[dxseqs]
            drugOntoEmb = drugALLontoEmb[drugseqs]
            ontoEmb = torch.cat((dxOntoEmb, drugOntoEmb), dim=2)
        '''
        dxrelationOntoEmb = dxrelationALLontoEmb[dxseqs]
        drugrelationOntoEmb = drugrelationALLontoEmb[drugseqs]
        '''



        #relontoEmb = torch.cat((dxrelationOntoEmb, drugrelationOntoEmb), dim=2)
#        print(ontoEmb.size(),relontoEmb.size())

        dxEHRVEmb = F.normalize(torch.sum(self.EHRdxEmb(dxseqs), dim=2), p=2, dim=2)
        drugEHRVEmb= F.normalize(torch.sum(self.EHRdrugEmb(drugseqs), dim=2), p=2, dim=2)

        if use_ctd==True:
            EHRVEmb = ctdDxOntoVEmb+ctdRxOntoVEmb##chan
        else:
            EHRVEmb = dxEHRVEmb+drugEHRVEmb##chan
        #EHRVEmb = dxEHRVEmb##chan
        if use_ontology==True:
            dxontoVEmb = torch.matmul(dx_onehot.permute(1,0,2), dxALLontoEmb[:-1])
            drugontoVEmb = torch.matmul(drug_onehot.permute(1,0,2), drugALLontoEmb[:-1])


        cooccurU = F.softmax(self.cooccurLinear(EHRVEmb), dim=2).contiguous()
        vonehot = torch.cat((dx_onehot.permute(1,0,2), drug_onehot.permute(1,0,2)), dim=2).contiguous()
        #print(dx_onehot.permute(1,0,2).shape)
        cooccur_loss = multi_class_cross_entropy_loss(cooccurU, vonehot)
        if use_ontology==True:
            ontoVEmb = F.normalize(dxontoVEmb, p=2, dim=2)+F.normalize(drugontoVEmb, p=2, dim=2)
#       
        if use_ontology==False:
            vs_emb =  EHRVEmb
            dxdrugEmb = EHREmb
        else:
            vs_emb = torch.cat((ontoVEmb, EHRVEmb), dim=2)
            dxdrugEmb = torch.cat((EHREmb,ontoEmb), dim=3)
#        print(vs_emb.size(),dxdrugEmb.size())
        vs_emb = torch.unsqueeze(vs_emb, dim=2).repeat(1,1,dx_num+drug_num,1)#change
#        vs_emb = torch.unsqueeze(vs_emb, dim=2).repeat(1,1,dx_num,1)
        #1/0
        attnInput = torch.cat((vs_emb, dxdrugEmb), dim=3)
        mlpOutput = torch.tanh(self.attn(attnInput))
        preAttention = self.attnCombine(mlpOutput)
        attention = F.softmax(preAttention, dim=2)
        if use_ontology==False:
        #    print(attention.size(),dxdrugEmb.size(),torch.sum(torch.mul(attention.repeat(1,1,1,self.EHREmbDim), dxdrugEmb), 2).size())
         #   1/0
            vs_emb = torch.sum(torch.mul(attention.repeat(1,1,1,self.EHREmbDim), dxdrugEmb), 2)
        else:
            vs_emb = torch.sum(torch.mul(attention.repeat(1,1,1,self.EHREmbDim+self.ontoEmbDim), dxdrugEmb), 2)
        vs_emb_dp = F.normalize(vs_emb, p=2, dim=2)
        DP_result = F.softmax(self.dpPredLinear(torch.tanh(vs_emb_dp)), dim=2)
        read_result = torch.sigmoid(self.readPredLinear(vs_emb_dp))
        return read_result,DP_result, cooccur_loss*10
class Naive_DX(nn.Module):#embes are loades
    def __init__(self, args,dxembs):
        super(Naive_DX, self).__init__()
        for k, v in args.__dict__.items():
            self.__setattr__(k, v)
        self.EHRdxEmb = torch.Tensor(dxembs).to(self.device)
        self.attn = nn.Linear(2*self.EHREmbDim, self.ptattnDim)
        self.attnCombine = nn.Linear(self.ptattnDim, 1)
        self.cooccurLinear = nn.Linear(self.EHREmbDim, self.dxVocabSize)
        self.dpPredLinear = nn.Linear(self.EHREmbDim, self.dpLabelSize)
        self.readPredLinear = nn.Linear(self.EHREmbDim, 1)

    def forward(self, inputs):
        (dxseqs, dx_onehot, dxLeavesList, dxAncestorsList,) = inputs
        dxseqs = dxseqs.to(self.device)
        dx_onehot = dx_onehot.to(self.device)
        dx_num = list(dxseqs.size())[2]
        dxEHREmb = self.EHRdxEmb[dxseqs]
        dxEHRVEmb = F.normalize(torch.sum(self.EHRdxEmb[dxseqs], dim=2), p=2, dim=2)
        EHRVEmb = dxEHRVEmb
        cooccurU = F.softmax(self.cooccurLinear(EHRVEmb), dim=2).contiguous()
        vonehot = dx_onehot.permute(1,0,2).contiguous()
#        cooccur_loss = multi_class_cross_entropy_loss(cooccurU, vonehot)
        vs_emb =  EHRVEmb
        dxEmb = dxEHREmb
        vs_emb = torch.unsqueeze(vs_emb, dim=2).repeat(1,1,dx_num,1)
        attnInput = torch.cat((vs_emb, dxEmb), dim=3)
        mlpOutput = torch.tanh(self.attn(attnInput))
        preAttention = self.attnCombine(mlpOutput)
        attention = F.softmax(preAttention, dim=2)
        vs_emb = torch.sum(torch.mul(attention.repeat(1,1,1,self.EHREmbDim), dxEmb), dim=2)
        vs_emb_dp = F.normalize(vs_emb, p=2, dim=2)
        DP_result = F.softmax(self.dpPredLinear(torch.tanh(vs_emb_dp)), dim=2) 
        read_result = torch.sigmoid(self.readPredLinear(vs_emb_dp))
        return read_result,DP_result, 0
