from models import *
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import logging
from utils import *
import argparse
import numpy as np
import random
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale_factor", type=int, default=2)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate')
    parser.add_argument('--gamma', type=float, default=0.5, help='')
    parser.add_argument('--n_epochs', type=int, default=40, help='number of epochs to train')
    parser.add_argument('--n_steps', type=int, default=30, help='number of epochs to update learning rate')
    parser.add_argument('--trainset_name', type=str, default='Davinci')
    parser.add_argument('--fold', type=str, default='0000')
    return parser.parse_args()

    
def train(train_loader, cfg):

    net = DCSSR(cfg.scale_factor).to(cfg.device)
    net.apply(weights_init_xavier)
    cudnn.benchmark = True

    criterion_mse = torch.nn.MSELoss().to(cfg.device)
    criterion_L1 = L1Loss()
    optimizer = torch.optim.Adam([paras for paras in net.parameters() if paras.requires_grad == True], lr=cfg.lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.n_steps, gamma=cfg.gamma)

    psnr_epoch = []
    loss_epoch = []
    loss_list = []
    psnr_list = []

    for idx_epoch in range(cfg.n_epochs):
        scheduler.step()
        for idx_iter, (HR_left, HR_right, LR_left, LR_right) in enumerate(train_loader):
            b, c, h, w = LR_left.shape
            HR_left, HR_right, LR_left, LR_right  = Variable(HR_left).to(cfg.device), Variable(HR_right).to(cfg.device), Variable(LR_left).to(cfg.device), Variable(LR_right).to(cfg.device)

            SR_left, SR_right, (M_2_to_1, M_1_to_2), (M_1_2_1, M_2_1_2), \
            (V_1_to_2, V_2_to_1), (M_2_to_1_sr, M_1_to_2_sr), \
            (M_2_to_1_interp, M_1_to_2_interp), (_M_2_to_1, _M_1_to_2), (_M_1_2_1, _M_2_1_2), \
            (_V_1_to_2, _V_2_to_1), (_M_2_to_1_sr, _M_1_to_2_sr), \
            (_M_2_to_1_interp, _M_1_to_2_interp) = net(LR_left, LR_right, is_training=1)

            ### loss_SR
            loss_SR = criterion_mse(SR_left, HR_left) + criterion_mse(SR_right, HR_right)

            ### loss_smoothness
            loss_h_left = criterion_L1(M_2_to_1[:, :-1, :, :], M_2_to_1[:, 1:, :, :]) + \
                     criterion_L1(M_1_to_2[:, :-1, :, :], M_1_to_2[:, 1:, :, :])
            loss_h_right = criterion_L1(_M_2_to_1[:, :-1, :, :], _M_2_to_1[:, 1:, :, :]) + \
                     criterion_L1(_M_1_to_2[:, :-1, :, :], _M_1_to_2[:, 1:, :, :])
            loss_w_left = criterion_L1(M_2_to_1[:, :, :-1, :-1], M_2_to_1[:, :, 1:, 1:]) + \
                     criterion_L1(M_1_to_2[:, :, :-1, :-1], M_1_to_2[:, :, 1:, 1:])
            loss_w_right = criterion_L1(_M_2_to_1[:, :, :-1, :-1], _M_2_to_1[:, :, 1:, 1:]) + \
                     criterion_L1(_M_1_to_2[:, :, :-1, :-1], _M_1_to_2[:, :, 1:, 1:])
            loss_h = loss_h_left + loss_h_right
            loss_w = loss_w_left + loss_w_right
            loss_smooth = loss_w + loss_h

            ### loss_cycle
            Identity = Variable(torch.eye(w, w).repeat(b, h, 1, 1), requires_grad=False).to(cfg.device)
                         
            loss_cycle_left = criterion_L1(M_1_2_1, Identity) + \
                         criterion_L1(M_2_1_2, Identity)
            loss_cycle_right = criterion_L1(_M_1_2_1, Identity) + \
                         criterion_L1(M_2_1_2, Identity )
            
            loss_cycle = loss_cycle_left + loss_cycle_right


            ### loss_photometric
            LR_right_warped = torch.bmm(M_2_to_1.contiguous().view(b*h,w,w), LR_right.permute(0, 2, 3, 1).contiguous().view(b*h, w, c))
            LR_right_warped = LR_right_warped.view(b, h, w, c).contiguous().permute(0, 3, 1, 2)
            LR_left_warped = torch.bmm(M_1_to_2.contiguous().view(b * h, w, w), LR_left.permute(0, 2, 3, 1).contiguous().view(b * h, w, c))
            LR_left_warped = LR_left_warped.view(b, h, w, c).contiguous().permute(0, 3, 1, 2)

            loss_photo = criterion_L1(LR_left, LR_right_warped) + criterion_L1(LR_right, LR_left_warped)
            ### Loss_consistence
            loss_consistence_left = (criterion_mse(M_2_to_1_sr, M_2_to_1_interp) + criterion_mse(M_1_to_2_sr, M_1_to_2_interp)) / 2
            loss_consistence_right = (criterion_mse(_M_2_to_1_sr, _M_2_to_1_interp) + criterion_mse(_M_1_to_2_sr, _M_1_to_2_interp)) / 2
            loss_consistence = loss_consistence_left + loss_consistence_right
            
            ### losses
            loss = loss_SR + 0.01 * (loss_photo + loss_smooth + loss_cycle) + 0.005 * loss_consistence

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            psnr_epoch.append(cal_psnr(HR_left[:,:,:,64:].data.cpu(), SR_left[:,:,:,64:].data.cpu()))
            loss_epoch.append(loss.data.cpu())
            
            if idx_iter % 100 == 0:
                 print('Iter----%04d, Epoch----%02d, loss----%f' % (idx_iter, idx_epoch + 1, float(loss)))

        if idx_epoch % 1 == 0:
            loss_list.append(float(np.array(loss_epoch).mean()))
            psnr_list.append(float(np.array(psnr_epoch).mean()))
            print('Epoch----%5d, loss----%f, PSNR----%f' % (idx_epoch + 1, float(np.array(loss_epoch).mean()), float(np.array(psnr_epoch).mean())))

            save_ckpt({
                'epoch': idx_epoch + 1,
                'state_dict': net.state_dict(),
                'loss': loss_list,
                'psnr': psnr_list,
            }, save_path = 'model/x' + str(cfg.scale_factor) + '/', filename=net.model_name + '_x' + str(cfg.scale_factor) + '_epoch' + str(idx_epoch + 1) + 'fold' + cfg.fold + '.pth.tar')
            psnr_epoch = []
            loss_epoch = []
    save_ckpt(net.state_dict(), save_path = 'model/x' + str(cfg.scale_factor) + '/', filename=net.model_name + '_x' + str(cfg.scale_factor) + 'fold' + cfg.fold + '.pth.tar')

def main(cfg):

    # train_set = TrainSetLoader(dataset_dir = [os.path.join('data', 'train', cfg.trainset_name + '_patches')], cfg=cfg)
    #  
    # K-fold
    dataset_dirs = []
    folds = cfg.fold
    for idx in range(len(folds)):
        dataset_dirs.append(os.path.join('data', 'train', cfg.trainset_name + '_patches_fold' + folds[idx]))
    train_set = TrainSetLoader(dataset_dir = dataset_dirs, cfg=cfg)
    print('train set is ready')
    train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=cfg.batch_size, shuffle=True)
    print('train loader is ready')
    train(train_loader, cfg)

if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)
