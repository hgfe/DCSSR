from models import *
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from utils import *
import argparse
import os
from torchvision import transforms
import pandas as pd
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--testset_dir', type=str, default='data/test')
    parser.add_argument('--dataset', type=str, default='Medtronic_fold1_test')
    parser.add_argument('--scale_factor', type=int, default=2)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--fold', type=str, default='0000')
    parser.add_argument('--epoch', type=int, default=40)
    return parser.parse_args()

def test(test_loader, cfg):
    net = DCSSR(cfg.scale_factor).to(cfg.device)
    cudnn.benchmark = True
    pretrained_dict = torch.load('model/x' + str(cfg.scale_factor) + '/' + net.model_name + '_x' + str(cfg.scale_factor) + '_epoch' + str(cfg.epoch) + 'fold' + cfg.fold + '.pth.tar', map_location=cfg.device)
    net.load_state_dict(pretrained_dict['state_dict'])

    l_psnr_list = []
    r_psnr_list = []
    l_ssim_list = []
    r_ssim_list = []
    file_list = []

    with torch.no_grad():
        for idx_iter, (HR_left, HR_right, LR_left, LR_right) in tqdm(enumerate(test_loader)):
            b, c, h, w = LR_left.shape
            HR_left, HR_right, LR_left, LR_right = Variable(HR_left).to(cfg.device), Variable(HR_right).to(cfg.device), Variable(LR_left).to(cfg.device), Variable(LR_right).to(cfg.device)
            file_name = test_loader.dataset.file_list[idx_iter]
         
            SR_left, SR_right = net(LR_left, LR_right, is_training=0)
            SR_left = torch.clamp(SR_left, 0, 1)
            SR_right = torch.clamp(SR_right, 0, 1)

            #print(HR_left.shape, SR_left.shape)
            l_psnr_curr = cal_psnr(HR_left[:,:,:,64:], SR_left[:,:,:,64:])
            r_psnr_curr = cal_psnr(HR_right[:,:,:,:-64], SR_right[:,:,:,:-64])
            #print(psnr_curr)
            l_psnr_list.append(l_psnr_curr)
            r_psnr_list.append(r_psnr_curr)
            l_ssim_list.append(cal_ssim(HR_left[:,:,:,64:], SR_left[:,:,:,64:]))
            r_ssim_list.append(cal_ssim(HR_right[:,:,:,:-64], SR_right[:,:,:,:-64]))
            file_list.append(file_name)

            ## save results
            if not os.path.exists('results/'+cfg.dataset):
                os.mkdir('results/'+cfg.dataset)
            if not os.path.exists('results/'+cfg.dataset+'/'+file_name):
                os.mkdir('results/'+cfg.dataset+'/'+file_name)
            SR_left_img = transforms.ToPILImage()(torch.squeeze(SR_left.data.cpu(), 0))
            SR_left_img.save('results/'+cfg.dataset+'/'+file_name+'_L.png')
            SR_right_img = transforms.ToPILImage()(torch.squeeze(SR_right.data.cpu(), 0))
            SR_right_img.save('results/'+cfg.dataset+'/'+file_name+'_R.png')

        #print results
        df = pd.DataFrame({"file": pd.Series(np.array(file_list)), "lpsnr": pd.Series(np.array(l_psnr_list)), "lssim": pd.Series(np.array(l_ssim_list)), "rpsnr": pd.Series(np.array(r_psnr_list)), "rssim": pd.Series(np.array(r_ssim_list))})
        df.to_csv('results/'+cfg.dataset + '_' + net.model_name + '_x' + str(cfg.scale_factor) + '_epoch40fold' + cfg.fold + '_psnr.csv')
        print(cfg.dataset + ' mean lpsnr: ', float(np.array(l_psnr_list).mean()))
        print(cfg.dataset + ' std lpsnr: ', float(np.array(l_psnr_list).std()))
        print(cfg.dataset + ' mean rpsnr: ', float(np.array(r_psnr_list).mean()))
        print(cfg.dataset + ' std rpsnr: ', float(np.array(r_psnr_list).std()))
        print(cfg.dataset + ' mean lssim: ', float(np.array(l_ssim_list).mean()))
        print(cfg.dataset + ' std lssim: ', float(np.array(l_ssim_list).std()))
        print(cfg.dataset + ' mean rssim: ', float(np.array(r_ssim_list).mean()))
        print(cfg.dataset + ' std rssim: ', float(np.array(r_ssim_list).std()))


def main(cfg):
    test_set = TestSetLoader(dataset_dir=cfg.testset_dir + '/' + cfg.dataset, scale_factor=cfg.scale_factor)
    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)
    result = test(test_loader, cfg)
    return result

if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)
    