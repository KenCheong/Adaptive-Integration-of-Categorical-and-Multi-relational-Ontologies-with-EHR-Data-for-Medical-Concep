"""
Additional layers.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl import DGLGraph

gcn_msg = fn.copy_u(u='h', out='m')
gcn_reduce = fn.sum(msg='m', out='h')

device = torch.device('cuda')
#device = torch.device('cpu')
#device = torch.device('cuda')
#ontology_data={(('dx','ctd','rx'):None),(('dx','icd','extra_dx'):None),(('rx','atc','extra_rx'):None),(('dx','snomed','extra'):None)}
#def load_dgi_icd(ontology_data):
 #   return ontology_data
class GATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim):
        super(GATLayer, self).__init__()
        self.g = g
        # equation (1)
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        # equation (2)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        # edge UDF for equation (2)
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        # reduce UDF for equation (3) & (4)
        # equation (3)
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        # equation (4)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, h):
        # equation (1)
        z = self.fc(h)
        self.g.ndata['z'] = z
        # equation (2)
        self.g.apply_edges(self.edge_attention)
        # equation (3) & (4)
        self.g.update_all(self.message_func, self.reduce_func)
        1/0
        return self.g.ndata.pop('h')
class GCNLayer(nn.Module):

    def __init__(self,rels, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
#        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        # equation (2)
       # self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
       # self.reset_parameters()
        self.rel_Ws={}
        for i in rels:
            #self.rel_Ws[i[1]]=  nn.Linear(in_feats, out_feats,bias=False)
            self.rel_Ws=  nn.Linear(in_feats, out_feats,bias=False)
#            self.rel_Ws[i[1]].to(device)
        

    def conv_mess(self,edges):
        return {'m': self.rel_Ws[edges.etype](edges.src['h'])}
    def reduce_f(self,nodes):
#        print((nodes.mailbox['m']).detach().numpy().shape,nodes.data['h'].detach().numpy().shape)
        return {'h': torch.mean(nodes.mailbox['m'], dim=1)+nodes.data['h']}
#        return {'h': nodes.data['h']}
    def forward(self, g, feature):
        # Creating a local scope so that all the stored ndata and edata
        # (such as the `'h'` ndata below) are automatically popped out
        # when the scope exits.
        for srctype in g.ntypes:
#            if srctype=='extra_rx' or :continue
            g.nodes[srctype].data['h'] = feature[srctype]
        funcs={}
        for c_etype in g.canonical_etypes:
 #           if etype!='icd':continue
            srctype, etype, dsttype = c_etype
#            Wh = self.weight[etype]
            # Save it in graph for message passing
            #G.nodes[srctype].data['Wh_%s' % etype] = Wh
            # Specify per-relation message passing functions: (message_func, reduce_func).
            # Note that the results are saved to the same destination feature 'h', which
            # hints the type wise reducer for aggregation.
            #funcs[etype] = (fn.copy_u('Wh_%s' % etype, 'm'), fn.mean('m', 'h'))
            #funcs[etype] = (lambda edges: {'m': self.rel_Ws[etype](edges.src['h'])},self.reduce_f)
            #funcs[etype] = (lambda edges: {'m': self.rel_Ws[etype](edges.src['h'])},self.reduce_f)
            funcs[etype] = (lambda edges: {'m': self.rel_Ws(edges.src['h'])},self.reduce_f)
#        print(self.rel_Ws['icd'].weight)

        # Trigger message passing of multiple types.
        g.multi_update_all(funcs, lambda nodes:{'h':torch.mean(nodes.mailbox['h'],dim=1)})
        # return the updated node feature dictionary
        #return {ntype : self.linear(g.nodes[ntype].data['h']) for ntype in g.ntypes}
        return {ntype : g.nodes[ntype].data['h'] for ntype in g.ntypes}
        '''
        with g.local_scope():
            g.ndata['h'] = feature
            g.update_all(gcn_msg, gcn_reduce)
            h = g.ndata['h']
            return self.linear(h)
        '''
class GCNNet(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = GCNLayer(400, 400)
#        self.layer2 = GCNLayer(16, 7)

    def forward(self, g, features):
        x = F.relu(self.layer1(g, features))
        x = self.layer1(g, x)
        return x
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
        dxEmb,leavesList, ancestorsList,relationList,permute_index = ontoInput
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
            #tempEmb = torch.sum((ancestorsEmb)*attn, dim=1)
            tempEmb = torch.sum((ancestorsEmb), dim=1)
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
