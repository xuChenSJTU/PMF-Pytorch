from __future__ import print_function
import torch.utils.data
import torch.nn.init as init
import matplotlib.pyplot as plt
from PMF_model import *
from evaluations import *
import pickle
import os

print('PMF Recommendation Model Example')
# input batch size for training
batch_size = 1024
# number of epoches to train
epoches = 500
# enables CUDA training
no_cuda = False
# random seed
seed = 1
# weight decay
weight_decay = 0.1

# choose dataset name and load dataset
dataset = 'ml-100k'
processed_data_path = os.path.join(os.getcwd(), 'processed_data', dataset)
user_id_index = pickle.load(open(os.path.join(processed_data_path, 'user_id_index.pkl'), 'rb'))
item_id_index = pickle.load(open(os.path.join(processed_data_path, 'item_id_index.pkl'), 'rb'))
data = np.loadtxt(os.path.join(processed_data_path, 'data.txt'), dtype=float)

# set split ratio
ratio = 0.6
train_data = data[:int(ratio*data.shape[0])]
vali_data = data[int(ratio*data.shape[0]):int((ratio+(1-ratio)/2)*data.shape[0])]
test_data = data[int((ratio+(1-ratio)/2)*data.shape[0]):]

NUM_USERS = max(user_id_index.values()) + 1
NUM_ITEMS = max(item_id_index.values()) + 1


print('dataset density:{:f}'.format(len(data)*1.0/(NUM_USERS*NUM_ITEMS)))
#
cuda = not no_cuda and torch.cuda.is_available()
torch.manual_seed(seed=seed)
if cuda:
    torch.cuda.manual_seed(seed=seed)

kwargs = {'num_workers':1, 'pin_memory':True} if cuda else {}

# construct data_loader
train_data_loader = torch.utils.data.DataLoader(torch.from_numpy(train_data), batch_size=batch_size, shuffle=False, **kwargs)
test_data_loader = torch.utils.data.DataLoader(torch.from_numpy(test_data), batch_size=batch_size, shuffle=False, **kwargs)

model = PMF(n_users=NUM_USERS, n_items=NUM_ITEMS, n_factors=20, no_cuda=no_cuda)

if cuda:
    model.cuda()

# loss function and optimizer
loss_function = nn.MSELoss(size_average=False)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=weight_decay)

# train function
def train(epoch, train_data_loader):
    model.train()
    epoch_loss = 0.0

    optimizer.zero_grad()

    for batch_idx, ele in enumerate(train_data_loader):
        optimizer.zero_grad()

        row = ele[:, 0]
        col = ele[:, 1]
        val = ele[:, 2]

        row = Variable(row.long())
        # TODO: turn this into a collate_fn like the data_loader
        if isinstance(col, list):
            col = tuple(Variable(c.long()) for c in col)
        else:
            col = Variable(col.long())
        val = Variable(val.float())

        if cuda:
            row = row.cuda()
            col = col.cuda()
            val = val.cuda()

        preds = model.forward(row, col)
        loss = loss_function(preds, val)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.data[0]

    epoch_loss /= train_data_loader.dataset.shape[0]

    return epoch_loss

def validate(epoch, vali_data_loader):
    model.eval()
    epoch_loss = 0.0
    for batch_idx, ele in enumerate(vali_data_loader):
        optimizer.zero_grad()

        row = ele[:, 0]
        col = ele[:, 1]
        val = ele[:, 2]

        row = Variable(row.long())
        # TODO: turn this into a collate_fn like the data_loader
        if isinstance(col, list):
            col = tuple(Variable(c.long()) for c in col)
        else:
            col = Variable(col.long())
        val = Variable(val.float())

        if cuda:
            row = row.cuda()
            col = col.cuda()
            val = val.cuda()

        preds = model.forward(row, col)
        loss = loss_function(preds, val)

        epoch_loss += loss.data[0]

    epoch_loss /= vali_data_loader.dataset.shape[0]

    return epoch_loss

# training model part
print('################Training model###################')
train_loss_list = []
last_vali_rmse = None
train_rmse_list = []
vali_rmse_list = []
print('parameters are: train ratio:{:f},batch_size:{:d}, epoches:{:d}, weight_decay:{:f}'.format(ratio, batch_size, epoches, weight_decay))
print(model)
for epoch in range(1, epoches+1):

    # construct train and vali loss list
    train_epoch_loss = train(epoch, train_data_loader)
    train_loss_list.append(train_epoch_loss)

    # construct early stop condition
    if cuda:
        vali_row = Variable(torch.from_numpy(vali_data[:, 0]).long()).cuda()
        vali_col = Variable(torch.from_numpy(vali_data[:, 1]).long()).cuda()
    else:
        vali_row = Variable(torch.from_numpy(vali_data[:, 0]).long())
        vali_col = Variable(torch.from_numpy(vali_data[:, 1]).long())

    vali_preds = model.predict(vali_row, vali_col)

    train_rmse = np.sqrt(train_epoch_loss)
    if cuda:
        vali_rmse = RMSE(vali_preds.cpu().data.numpy(), vali_data[:, 2])
    else:
        vali_rmse = RMSE(vali_preds.data.numpy(), vali_data[:, 2])

    train_rmse_list.append(train_rmse)
    vali_rmse_list.append(vali_rmse)

    print('training epoch:{: d}, training rmse:{: .6f}, vali rmse:{:.6f}'. \
              format(epoch, train_rmse, vali_rmse))

    if last_vali_rmse and last_vali_rmse < vali_rmse:
        break
    else:
        last_vali_rmse = vali_rmse

# test part
print('################Testing trained model###################')

if cuda:
    test_row = Variable(torch.from_numpy(test_data[:, 0]).long()).cuda()
    test_col = Variable(torch.from_numpy(test_data[:, 1]).long()).cuda()
else:
    test_row = Variable(torch.from_numpy(test_data[:, 0]).long())
    test_col = Variable(torch.from_numpy(test_data[:, 1]).long())

preds = model.predict(test_row, test_col)

if cuda:
    test_rmse = RMSE(preds.cpu().data.numpy(), test_data[:, 2])
else:
    test_rmse = RMSE(preds.data.numpy(), test_data[:, 2])
print('test rmse: {:f}'.format(test_rmse))

plt.figure(1)
plt.plot(range(1, len(train_rmse_list)+1), train_rmse_list, color='r', label='train rmse')
plt.plot(range(1, len(vali_rmse_list)+1), vali_rmse_list, color='b', label='test rmse')
plt.legend()
plt.annotate(r'train=%f' % (train_rmse_list[-1]), xy=(len(train_rmse_list), train_rmse_list[-1]),
             xycoords='data', xytext=(-30, 30), textcoords='offset points', fontsize=10,
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3, rad=.2'))
plt.annotate(r'vali=%f' % (vali_rmse_list[-1]), xy=(len(vali_rmse_list), vali_rmse_list[-1]),
             xycoords='data', xytext=(-30, 30), textcoords='offset points', fontsize=10,
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3, rad=.2'))
plt.xlim([1, len(train_rmse_list)+10])
plt.xlabel('iterations')
plt.ylabel('RMSE')
plt.title('RMSE Curve in Training Process')
plt.show()




