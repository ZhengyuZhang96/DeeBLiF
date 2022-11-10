import sys
import time
import torch
import argparse
from Val import val
from Utils import *
from Model import Network
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--n_steps', type=int, default=30, help='number of epochs to update learning rate')
    parser.add_argument('--n_epochs', type=int, default=70, help='number of epochs to train')
    parser.add_argument('--gamma', type=float, default=0.1, help='learning rate decaying factor')
    parser.add_argument('--trainset_dir', type=str, default='./Datasets/Win5_160x160/')

    return parser.parse_args()

def train(train_loader, cfg, test_scene_id):

    os.makedirs('./log/' + str(test_scene_id[0]) + '_' + str(test_scene_id[1]))
    sys.stdout = open('./log/' + str(test_scene_id[0]) + '_' + str(test_scene_id[1])
                      + '/' + str(test_scene_id[0]) + '_' + str(test_scene_id[1]) + '.txt', 'a')
    print(cfg)
    print(test_scene_id)

    net = Network().to(cfg.device).apply(weights_init_xavier)
    cudnn.benchmark = True
    optimizer = torch.optim.SGD([paras for paras in net.parameters() if paras.requires_grad == True],
                               lr=cfg.lr, momentum=0.9, weight_decay= 1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.n_steps, gamma=cfg.gamma)

    for idx_epoch in range(0, cfg.n_epochs):

        loss_epoch = []
        loss_list = []
        start_time = time.time()

        for idx_iter, (data, score_label) in enumerate(train_loader):

            data, score_label= Variable(data).to(cfg.device), Variable(score_label).to(cfg.device)
            score_label = score_label.view(score_label.size()[0], -1)
            score_out = net(data)
            loss = torch.nn.MSELoss().to(cfg.device)(score_out, score_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_epoch.append(loss.data.cpu())

        loss_list.append(float(np.array(loss_epoch).mean()))
        end_time = time.time()
        print('Test Epoch----%5d,'
              ' loss---%f,'
              ' Time---%f s'
              ' lr---%7f s'
              % (idx_epoch + 1,
                 float(np.array(loss_epoch).mean()),
                 end_time-start_time,
                 scheduler.get_lr()[0]))

        save_ckpt({'epoch': idx_epoch + 1,'state_dict': net.state_dict(),'loss': loss_list,},
                  save_path = './log/' + str(test_scene_id[0]) + '_' + str(test_scene_id[1]) + '/',
                  filename='DeeBLiF_epoch' + str(idx_epoch + 1) + '.pth.tar')
        load_model_path = './log/'+ str(test_scene_id[0]) + '_' + str(test_scene_id[1]) + \
                          '/DeeBLiF_epoch' + str(idx_epoch + 1) + '.pth.tar'
        val(valset_dir=cfg.trainset_dir, test_scene_id=test_scene_id, load_model_path=load_model_path)

        scheduler.step()

def save_ckpt(state, save_path='./log', filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(save_path,filename))

def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)

def main(cfg):

    scene_num = 10 # for Win5-LID dataset
    full_dataset_dir = cfg.trainset_dir
    for i in range(scene_num):
        for j in range(i+1,scene_num):
            test_scene_id = [i,j]
            train_set = MyTrainSetLoader_Kfold(dataset_dir=full_dataset_dir, test_scene_id = test_scene_id)
            train_loader = DataLoader(dataset=train_set, num_workers=6, batch_size=cfg.batch_size, shuffle=True)
            train(train_loader, cfg, test_scene_id)

if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)
