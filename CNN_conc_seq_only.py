import torch
import torch.nn as nn

from utils import *
import pdb
import argparse
import random
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--l0', type=float, default=0.01)
parser.add_argument('--l1', type=float, default=0.01)
parser.add_argument('--l2', type=float, default=0.0001)
parser.add_argument('--l3', type=float, default=1000.0)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--epoch', type=int, default=200)
parser.add_argument('--train', type=int, default=1)
parser.add_argument('--data_processed_dir', type=str, default='/home/at/work/dataset/ECEN_404_dataset/vector_machine_data/')
args = parser.parse_args()
print(args)


torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
np.random.seed(0)
random.seed(0)


vocab_size_protein = 29
GRU_size_prot = 256


###### define network ######
class net_crossInteraction(nn.Module):
    def __init__(self, lambda_l1, lambda_fused, lambda_group, lambda_bind):
        super().__init__()
        self.mod_aminoAcid_embedding = nn.Embedding(vocab_size_protein, GRU_size_prot)
    
        self.conv0 = nn.Sequential(nn.ConstantPad1d((1,2), 0), 
                                   nn.Conv1d(GRU_size_prot, 32*8, 4, stride = 1),
                                   nn.ReLU())
        self.conv1 = nn.Sequential(nn.ConstantPad1d((3,4), 0), 
                                   nn.Conv1d(GRU_size_prot, 32*8, 8, stride = 1),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.ConstantPad1d((5,6), 0), 
                                   nn.Conv1d(GRU_size_prot, 32*8, 12, stride = 1),
                                   nn.ReLU())
        


        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.joint_attn_prot1, self.joint_attn_prot2 = nn.Linear(256, 256), nn.Linear(256, 256)
        self.tanh = nn.Tanh()

        self.regressor0 = nn.Sequential(nn.Conv1d(1, 64, kernel_size=4, stride=2, padding=1),
                                       nn.LeakyReLU(0.1),
                                       nn.MaxPool1d(kernel_size=4, stride=4))
        self.regressor1 = nn.Sequential(nn.Linear(64*32, 600),
                                        nn.LeakyReLU(0.1),
                                        nn.Dropout(0.5),
                                        nn.Linear(600, 300),
                                        nn.LeakyReLU(0.1),
                                        nn.Dropout(0.5),
                                        nn.Linear(300, 1))

        self.lambda_l1, self.lambda_fused, self.lambda_group = lambda_l1, lambda_fused, lambda_group
        self.lambda_bind = lambda_bind

    def forward(self, prot_data1, prot_data2, label):   
        # protein embedding 1
        aminoAcid_embedding1 = self.mod_aminoAcid_embedding(prot_data1)

        prot_seq_embedding1 = self.conv2(self.conv1(self.conv0(aminoAcid_embedding1)))
        
        
        # protein embedding 2
        aminoAcid_embedding2 = self.mod_aminoAcid_embedding(prot_data2)

        prot_seq_embedding2 = self.conv2(self.conv1(self.conv0(aminoAcid_embedding2)))


        prot_embedding1 = prot_seq_embedding1
        prot_embedding2 = prot_seq_embedding2

        # protein-protein interaction
        inter_prot_prot = self.sigmoid(torch.einsum('bij,bkj->bik', self.joint_attn_prot1(self.relu(prot_embedding1)), self.joint_attn_prot2(self.relu(prot_embedding2))))
        inter_prot_prot_sum = torch.einsum('bij->b', inter_prot_prot)
        inter_prot_prot = torch.einsum('bij,b->bij', inter_prot_prot, 1/inter_prot_prot_sum)

        # protein-protein joint embedding
        pp_embedding = self.tanh(torch.einsum('bij,bkj->bikj', prot_embedding1, prot_embedding2))
        pp_embedding = torch.einsum('bijk,bij->bk', pp_embedding, inter_prot_prot)

        # protein-protein affinity
        affn_prot_prot = pp_embedding[:, None, :]
        affn_prot_prot = self.regressor0(affn_prot_prot)
        affn_prot_prot = affn_prot_prot.view(b, 64*32)
        affn_prot_prot = self.regressor1(affn_prot_prot)

        loss0, loss1 = self.loss_reg(inter_prot_prot, fused_matrix.to(inter_prot_prot.device)), self.loss_affn(affn_prot_prot, label)
        loss = loss0 + loss1

        return loss

    def forward_inter_affn(self, prot_data1, prot_data2):
        # protein embedding 1
        aminoAcid_embedding1 = self.mod_aminoAcid_embedding(prot_data1)

        prot_seq_embedding1 = self.conv2(self.conv1(self.conv0(aminoAcid_embedding1)))
        
        
        # protein embedding 2
        aminoAcid_embedding2 = self.mod_aminoAcid_embedding(prot_data2)

        prot_seq_embedding2 = self.conv2(self.conv1(self.conv0(aminoAcid_embedding2)))

        prot_embedding1 = prot_seq_embedding1
        prot_embedding2 = prot_seq_embedding2


        # protein-protein interaction
        inter_prot_prot = self.sigmoid(torch.einsum('bij,bkj->bik', self.joint_attn_prot1(self.relu(prot_embedding1)), self.joint_attn_prot2(self.relu(prot_embedding2))))
        inter_prot_prot_sum = torch.einsum('bij->b', inter_prot_prot)
        inter_prot_prot = torch.einsum('bij,b->bij', inter_prot_prot, 1/inter_prot_prot_sum)

        # protein-protein joint embedding
        pp_embedding = self.tanh(torch.einsum('bij,bkj->bikj', prot_embedding1, prot_embedding2))
        pp_embedding = torch.einsum('bijk,bij->bk', pp_embedding, inter_prot_prot)

        # protein-protein affinity
        affn_prot_prot = pp_embedding[:, None, :]
        affn_prot_prot = self.regressor0(affn_prot_prot)
        affn_prot_prot = affn_prot_prot.view(b, 64*32)
        affn_prot_prot = self.regressor1(affn_prot_prot)

        return inter_prot_prot, affn_prot_prot

    def loss_reg(self, inter, fused_matrix):     
        reg_l1 = torch.abs(inter).sum(dim=(1,2)).mean()
        reg_fused = torch.abs(torch.einsum('bij,ti->bjt', inter, fused_matrix)).sum(dim=(1,2)).mean()
        
        # group = torch.einsum('bij,bki->bjk', inter**2, prot_contacts2).sum(dim=1)
        # group[group==0] = group[group==0] + 1e10
        # reg_group = ( torch.sqrt(group) * torch.sqrt(prot_contacts2.sum(dim=2)) ).sum(dim=1).mean()
        
        reg_loss = self.lambda_l1 * reg_l1 + self.lambda_fused * reg_fused
        return reg_loss

    def loss_inter(self, inter, prot_inter, prot_inter_exist):
        label = torch.einsum('b,bij->bij', prot_inter_exist, prot_inter)
        loss = torch.sqrt(((inter - label) ** 2).sum(dim=(1,2))).mean() * self.lambda_bind
        return loss

    def loss_affn(self, affn, label):
        loss = ((affn - label) ** 2).mean()
        return loss


import scipy.sparse
class dataset(torch.utils.data.Dataset):
    def __init__(self, name_split='train'):
        if name_split == 'train':
            #self.prot_data1, self.prot_data2, self.prot_contacts1, self.prot_contacts2, _, self.prot_inter, self.prot_inter_exist, self.label = load_train_data(args.data_processed_dir)
            self.prot_data1, self.prot_data2, self.label = load_train_data(args.data_processed_dir)
        elif name_split == 'val':
            #self.prot_data1, self.prot_data2, self.prot_contacts1, self.prot_contacts2, _, self.prot_inter, self.prot_inter_exist, self.label = load_val_data(args.data_processed_dir)
            self.prot_data1, self.prot_data2, self.label = load_val_data(
                args.data_processed_dir)
        elif name_split == 'test':
            #self.prot_data1, self.prot_data2, self.prot_contacts1, self.prot_contacts2, _, self.prot_inter, self.prot_inter_exist, self.label = load_test_data(args.data_processed_dir)
            self.prot_data1, self.prot_data2, self.label = load_test_data(
                args.data_processed_dir)

        elif name_split == 'one_unseen_prot':
            self.prot_data1, self.prot_data2, self.prot_contacts1, self.prot_contacts2, _, self.prot_inter, self.prot_inter_exist, self.label = load_uniqOne_data(args.data_processed_dir)
        elif name_split == 'unseen_both':
            self.prot_data1, self.prot_data2, self.prot_contacts1, self.prot_contacts2, _, self.prot_inter, self.prot_inter_exist, self.label = load_uniqTwo_data(args.data_processed_dir)

        #self.prot_data1, self.prot_data2, self.prot_contacts1, self.prot_contacts2, self.prot_inter, self.prot_inter_exist, self.label = torch.tensor(self.prot_data1), torch.tensor(self.prot_data2), torch.tensor(self.prot_contacts1).float(), torch.tensor(self.prot_contacts2).float(), torch.tensor(self.prot_inter).float(), torch.tensor(self.prot_inter_exist).float().squeeze().float(), torch.tensor(self.label).float()
        self.prot_data1, self.prot_data2, self.label = torch.tensor(
            self.prot_data1), torch.tensor(self.prot_data2), torch.tensor(self.label).float()


    def __len__(self):
       # return [self.prot_data1.size()[0], self.prot_data2.size()[0]]
       return self.prot_data1.size()[0]
    def __getitem__(self, index):
        #return self.prot_data1[index], self.prot_data2[index], self.prot_contacts1[index], self.prot_contacts2[index], self.prot_inter[index], self.prot_inter_exist[index], self.label[index]
        return self.prot_data1[index], self.prot_data2[index], self.label[index]

"""
class dataset(torch.utils.data.Dataset):
    def __init__(self, name_split='train'):
        if name_split == 'train':
            self.prot_data1, self.prot_data2, _, self.label = load_train_data(args.data_processed_dir)
        elif name_split == 'val':
            self.prot_data1, self.prot_data2, _, self.label = load_val_data(args.data_processed_dir)
        elif name_split == 'test':
            self.prot_data1, self.prot_data2, _, self.label = load_test_data(args.data_processed_dir)
        elif name_split == 'one_unseen_prot':
            self.prot_data1, self.prot_data2, _, self.label = load_uniqOne_data(args.data_processed_dir)
        elif name_split == 'unseen_both':
            self.prot_data1, self.prot_data2, _, self.label = load_uniqTwo_data(args.data_processed_dir)
        self.prot_data1, self.prot_data2, self.label = torch.tensor(self.prot_data1), torch.tensor(self.prot_data2), torch.tensor(self.label).float()
    def __len__(self):
        return [self.prot_data1.size()[0], self.prot_data2.size()[0]]
    def __getitem__(self, index):
        return self.prot_data1[index], self.prot_data2[index], self.label[index]    
"""





###### train ######
train_set = dataset('train')
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True)
val_set = dataset('val')
val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=args.batch_size, shuffle=False)
model = net_crossInteraction(args.l0, args.l1, args.l2, args.l3)
model = nn.DataParallel(model)
model = model.cuda()
optimizer = torch.optim.Adam(model.parameters(), 1e-4)

#fused_matrix = torch.tensor(np.load(args.data_processed_dir+'fused_matrix.npy')).cuda()
loss_val_best = 1e10

if args.train == 1:
    # train
    torch.cuda.empty_cache()
    for epoch in range(args.epoch):
        model.train()
        loss_epoch, batch = 0, 0
        for prot_data1, prot_data2, label in train_loader:
            prot_data1, prot_data2, label = prot_data1.cuda(), prot_data2.cuda(), label.cuda()

            optimizer.zero_grad()
            loss = model(prot_data1, prot_data2, label).mean()
            # print('epoch', epoch, 'batch', batch, loss.detach().cpu().numpy())

            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 5)
            optimizer.step()
            loss_epoch += loss.detach().cpu().numpy()
            batch += 1

        model.eval()
        loss_epoch_val, batch_val = 0, 0
        for prot_data1, prot_data2, label in val_loader:
            prot_data1, prot_data2, label = prot_data1.cuda(), prot_data2.cuda(), label.cuda()
            with torch.no_grad():
                loss = model(prot_data1, prot_data2, label).mean()
            loss_epoch_val += loss.detach().cpu().numpy()
            batch_val += 1

        print('epoch', epoch, 'train loss', loss_epoch/batch, 'val loss', loss_epoch_val/batch_val)
        if loss_epoch_val/batch_val < loss_val_best:
            loss_val_best = loss_epoch_val/batch_val
            torch.save(model.module.state_dict(), './weights/concatenation_' + str(args.l0) + '_' + str(args.l1) + '_' + str(args.l2) + '_' + str(args.l3) + '.pth')

del train_loader
del val_loader


###### evaluation ######
# evaluation
model = net_crossInteraction(args.l0, args.l1, args.l2, args.l3).cuda()
#model.load_state_dict(torch.load('./weights/concatenation_' + str(args.l0) + '_' + str(args.l1) + '_' + str(args.l2) + '_' + str(args.l3) + '.pth'))
model.eval()

data_processed_dir = args.data_processed_dir

print('train')
eval_set = dataset('train')
eval_loader = torch.utils.data.DataLoader(dataset=eval_set, batch_size=args.batch_size, shuffle=False)
#cal_affinity_torch(model, eval_loader)
#prot_length1 = np.load(data_processed_dir+'prot_train_length1.npy')
#prot_length2 = np.load(data_processed_dir+'prot_train_length2.npy')
#cal_interaction_torch(model, eval_loader, prot_length1, prot_length2)

print('val')
eval_set = dataset('val')
eval_loader = torch.utils.data.DataLoader(dataset=eval_set, batch_size=args.batch_size, shuffle=False)
cal_affinity_torch(model, eval_loader)
prot_length1 = np.load(data_processed_dir+'prot_dev_length1.npy')
prot_length2 = np.load(data_processed_dir+'prot_dev_length2.npy')
cal_interaction_torch(model, eval_loader, prot_length1, prot_length2)

print('test')
eval_set = dataset('test')
eval_loader = torch.utils.data.DataLoader(dataset=eval_set, batch_size=args.batch_size, shuffle=False)
cal_affinity_torch(model, eval_loader)
prot_length1 = np.load(data_processed_dir+'prot_test_length1.npy')
prot_length2 = np.load(data_processed_dir+'prot_test_length2.npy')
cal_interaction_torch(model, eval_loader, prot_length1, prot_length2)

print('exactly one unseen protein')
eval_set = dataset('unseen_prot')
eval_loader = torch.utils.data.DataLoader(dataset=eval_set, batch_size=args.batch_size, shuffle=False)
cal_affinity_torch(model, eval_loader)
prot_length1 = np.load(data_processed_dir+'uniq_one_length1.npy')
prot_length2 = np.load(data_processed_dir+'uniq_one_length2.npy')
cal_interaction_torch(model, eval_loader, prot_length1, prot_length2)


print('unseen both')
eval_set = dataset('unseen_both')
eval_loader = torch.utils.data.DataLoader(dataset=eval_set, batch_size=args.batch_size, shuffle=False)
cal_affinity_torch(model, eval_loader)
rot_length1 = np.load(data_processed_dir+'uniq_both_length1.npy')
prot_length2 = np.load(data_processed_dir+'uniq_both_length2.npy')
cal_interaction_torch(model, eval_loader, prot_length1, prot_length2)