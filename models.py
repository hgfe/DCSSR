import torch
import torch.nn as nn
import numpy as np
from math import *
import matplotlib.pyplot as plt
from skimage import morphology

### DCSSR_IPCAI2022
class DCSSR(nn.Module):

    def __init__(self, upscale_factor):
        super(DCSSR, self).__init__()
        self.upscale_factor = upscale_factor
        self.model_name = 'DCSSR_standard'
       
        ### feature extraction
        ResBlock = RCAB # or ResB
        ASPPBlock = DenseASPPB # ResASPPB 
        self.init_feature_lr_left = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            ResBlock(64),
            ASPPBlock(64),
            ResBlock(64),
            ASPPBlock(64),
            ResBlock(64),
        )
        self.init_feature_lr_right = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            ResBlock(64),
            ASPPBlock(64),
            ResBlock(64),
            ASPPBlock(64),
            ResBlock(64),
        )
        self.init_feature_sr_left = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            ResBlock(64),
            ASPPBlock(64),
            ResBlock(64),
            ASPPBlock(64),
            ResBlock(64),
        )
        self.init_feature_sr_right = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            ResBlock(64),
            ASPPBlock(64),
            ResBlock(64),
            ASPPBlock(64),
            ResBlock(64),
        )
        
        ### paralax attention
        self.pam_lr_left = APAM(64)
        self.pam_lr_right = APAM(64)
        self.pam_sr_left = APAM(64)
        self.pam_sr_right = APAM(64)
        ### upscaling
        self.upscale_left = nn.Sequential(
            ResBlock(64),
            ResBlock(64),
            ResBlock(64),
            ResBlock(64)
        )
        self.upscale_left_2 = nn.Sequential(
            nn.Conv2d(64 + 3, 64 * upscale_factor ** 2, 1, 1, 0, bias=False),
            nn.PixelShuffle(upscale_factor),
            nn.Conv2d(64, 3, 3, 1, 1, bias=False),
            nn.Conv2d(3, 3, 3, 1, 1, bias=False)
        )
        self.upscale_right = nn.Sequential(
            ResBlock(64),
            ResBlock(64),
            ResBlock(64),
            ResBlock(64)
        )    
        self.upscale_right_2 = nn.Sequential(
            nn.Conv2d(64 + 3, 64 * upscale_factor ** 2, 1, 1, 0, bias=False),
            nn.PixelShuffle(upscale_factor),
            nn.Conv2d(64, 3, 3, 1, 1, bias=False),
            nn.Conv2d(3, 3, 3, 1, 1, bias=False)
        )
        self.upscale_for_mask_left = nn.Sequential(
            ResBlock(64),
            ResBlock(64),
            ResBlock(64),
            ResBlock(64),
            nn.Conv2d(64, 64 * upscale_factor ** 2, 1, 1, 0, bias=False),
            nn.PixelShuffle(upscale_factor),
            nn.Conv2d(64, 3, 3, 1, 1, bias=False),
            nn.Conv2d(3, 3, 3, 1, 1, bias=False)
        )
        self.upscale_for_mask_right = nn.Sequential(
            ResBlock(64),
            ResBlock(64),
            ResBlock(64),
            ResBlock(64),
            nn.Conv2d(64, 64 * upscale_factor ** 2, 1, 1, 0, bias=False),
            nn.PixelShuffle(upscale_factor),
            nn.Conv2d(64, 3, 3, 1, 1, bias=False),
            nn.Conv2d(3, 3, 3, 1, 1, bias=False)
        )
    def forward(self, x_left, x_right, is_training):
        ### feature extraction
        buffer_lr_left = self.init_feature_lr_left(x_left)
        buffer_lr_right = self.init_feature_lr_right(x_right)
        if is_training == 1:
            ### parallax attention
            buffer_0, (M_2_to_1, M_1_to_2), (M_1_2_1, M_2_1_2), \
            (V_1_to_2, V_2_to_1) = self.pam_lr_left(buffer_lr_left, buffer_lr_right, is_training)
            
            buffer_1, (_M_2_to_1, _M_1_to_2), (_M_1_2_1, _M_2_1_2), \
            (_V_1_to_2, _V_2_to_1) = self.pam_lr_right(buffer_lr_right, buffer_lr_left, is_training)
            ### upscaling
            sr_left = self.upscale_left(buffer_0)
            sr_left = torch.cat((sr_left, x_left), 1)
            sr_left = self.upscale_left_2(sr_left)

            sr_right = self.upscale_right(buffer_1)   
            sr_right = torch.cat((sr_right, x_right), 1)
            sr_right = self.upscale_right_2(sr_right)
            
            ### parallax attention HR
            buffer_sr_left = self.init_feature_sr_left(sr_left)
            buffer_sr_right = self.init_feature_sr_right(sr_right)
            
            _buffer_0, (M_2_to_1_sr, M_1_to_2_sr), (M_1_2_1_sr, M_2_1_2_sr), \
            (V_1_to_2_sr, V_2_to_1_sr) = self.pam_sr_left(buffer_sr_left, buffer_sr_right, is_training)
            
            _buffer_1, (_M_2_to_1_sr, _M_1_to_2_sr), (_M_1_2_1_sr, _M_2_1_2_sr), \
            (_V_1_to_2_sr, _V_2_to_1_sr) = self.pam_sr_right(buffer_sr_right, buffer_sr_left, is_training)
            
            ### upscaling_for_mask
            buffer_M_2_to_1 = M_2_to_1.unsqueeze(1)
            buffer_M_1_to_2 = M_1_to_2.unsqueeze(1)
            M_2_to_1_interp = torch.nn.functional.interpolate(buffer_M_2_to_1, scale_factor = self.upscale_factor, mode = 'trilinear')
            M_1_to_2_interp = torch.nn.functional.interpolate(buffer_M_1_to_2, scale_factor = self.upscale_factor, mode = 'trilinear')
            M_2_to_1_interp = M_2_to_1_interp.squeeze()
            M_1_to_2_interp = M_1_to_2_interp.squeeze()
            
            _buffer_M_2_to_1 = _M_2_to_1.unsqueeze(1)
            _buffer_M_1_to_2 = _M_1_to_2.unsqueeze(1)
            _M_2_to_1_interp = torch.nn.functional.interpolate(_buffer_M_2_to_1, scale_factor = self.upscale_factor, mode = 'trilinear')
            _M_1_to_2_interp = torch.nn.functional.interpolate(_buffer_M_1_to_2, scale_factor = self.upscale_factor, mode = 'trilinear')
            _M_2_to_1_interp = _M_2_to_1_interp.squeeze()
            _M_1_to_2_interp = _M_1_to_2_interp.squeeze()

            
            return sr_left, sr_right, (M_2_to_1, M_1_to_2), (M_1_2_1, M_2_1_2), \
                   (V_1_to_2, V_2_to_1), (M_2_to_1_sr, M_1_to_2_sr), (M_2_to_1_interp, M_1_to_2_interp), \
                   (_M_2_to_1, _M_1_to_2), (_M_1_2_1, _M_2_1_2), \
                   (_V_1_to_2, _V_2_to_1), (_M_2_to_1_sr, _M_1_to_2_sr), (_M_2_to_1_interp, _M_1_to_2_interp)                         
                   
        if is_training == 0:
            ### parallax attention
            buffer_test_0, V_1 = self.pam_lr_left(buffer_lr_left, buffer_lr_right, is_training)
            buffer_test_1, V_2 = self.pam_lr_right(buffer_lr_right, buffer_lr_left, is_training)
            ### upscaling
            result_0 = self.upscale_left(buffer_test_0)
            result_0 = torch.cat((result_0, x_left), 1)
            result_0 = self.upscale_left_2(result_0)
            result_1 = self.upscale_right(buffer_test_1)
            result_1 = torch.cat((result_1, x_right), 1)
            result_1 = self.upscale_right_2(result_1)            

            return result_0, result_1, V_1, V_2


class ResB(nn.Module):
    def __init__(self, channels):
        super(ResB, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
        )
    def __call__(self,x):
        out = self.body(x)
        return out + x

class ResASPPB(nn.Module):
    def __init__(self, channels):
        super(ResASPPB, self).__init__()
        self.conv1_1 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1, 1, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv2_1 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 4, 4, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv3_1 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 8, 8, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv1_2 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1, 1, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv2_2 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 4, 4, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv3_2 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 8, 8, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv1_3 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1, 1, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv2_3 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 4, 4, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv3_3 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 8, 8, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.b_1 = nn.Conv2d(channels * 3, channels, 1, 1, 0, bias=False)
        self.b_2 = nn.Conv2d(channels * 3, channels, 1, 1, 0, bias=False)
        self.b_3 = nn.Conv2d(channels * 3, channels, 1, 1, 0, bias=False)
    def __call__(self, x):
        buffer_1 = []
        buffer_1.append(self.conv1_1(x))
        buffer_1.append(self.conv2_1(x))
        buffer_1.append(self.conv3_1(x))
        buffer_1 = self.b_1(torch.cat(buffer_1, 1))

        buffer_2 = []
        buffer_2.append(self.conv1_2(buffer_1))
        buffer_2.append(self.conv2_2(buffer_1))
        buffer_2.append(self.conv3_2(buffer_1))
        buffer_2 = self.b_2(torch.cat(buffer_2, 1))

        buffer_3 = []
        buffer_3.append(self.conv1_3(buffer_2))
        buffer_3.append(self.conv2_3(buffer_2))
        buffer_3.append(self.conv3_3(buffer_2))
        buffer_3 = self.b_3(torch.cat(buffer_3, 1))

        return x + buffer_1 + buffer_2 + buffer_3

class DenseASPPB(nn.Module):
    def __init__(self, channels):
        super(DenseASPPB, self).__init__()
        self.conv1_1 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1, 1, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv2_1 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 4, 4, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv3_1 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 8, 8, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv1_2 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1, 1, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv2_2 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 4, 4, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv3_2 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 8, 8, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv1_3 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1, 1, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv2_3 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 4, 4, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv3_3 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 8, 8, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.b_1 = nn.Conv2d(channels * 3, channels, 1, 1, 0, bias=False)
        self.b_2 = nn.Conv2d(channels * 3, channels, 1, 1, 0, bias=False)
        self.b_3 = nn.Conv2d(channels * 3, channels, 1, 1, 0, bias=False)
    def __call__(self, x):
        buffer_1 = []
        buffer_1.append(self.conv1_1(x))
        buffer_1.append(self.conv2_1(x))
        buffer_1.append(self.conv3_1(x))
        buffer_1 = self.b_1(torch.cat(buffer_1, 1))
        buffer_11 = buffer_1 + x

        buffer_2 = []
        buffer_2.append(self.conv1_2(buffer_11))
        buffer_2.append(self.conv2_2(buffer_11))
        buffer_2.append(self.conv3_2(buffer_11))
        buffer_2 = self.b_2(torch.cat(buffer_2, 1))
        buffer_22 = buffer_2 + x

        buffer_3 = []
        buffer_3.append(self.conv1_3(buffer_22))
        buffer_3.append(self.conv2_3(buffer_22))
        buffer_3.append(self.conv3_3(buffer_22))
        buffer_3 = self.b_3(torch.cat(buffer_3, 1))

        return x + buffer_1 + buffer_2 + buffer_3


class ASPPB(nn.Module):
    def __init__(self, channels):
        super(ASPPB, self).__init__()
        self.conv1_1 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1, 1, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv2_1 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 4, 4, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv3_1 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 8, 8, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.b_1 = nn.Conv2d(channels * 3, channels, 1, 1, 0, bias=False)
    def __call__(self, x):
        buffer_1 = []
        buffer_1.append(self.conv1_1(x))
        buffer_1.append(self.conv2_1(x))
        buffer_1.append(self.conv3_1(x))
        buffer_1 = self.b_1(torch.cat(buffer_1, 1))

        return x + buffer_1

class PAM(nn.Module):
    def __init__(self, channels):
        super(PAM, self).__init__()
        self.b1 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.b2 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.b3 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.softmax = nn.Softmax(-1)
        self.rb = ResB(64)
        self.fusion = nn.Conv2d(channels * 2 + 1, channels, 1, 1, 0, bias=True)
    def __call__(self, x_left, x_right, is_training):
        b, c, h, w = x_left.shape
        buffer_left = self.rb(x_left)
        buffer_right = self.rb(x_right)

        ### M_{right_to_left}
        Q = self.b1(buffer_left).permute(0, 2, 3, 1)                                                # B * H * W * C
        S = self.b2(buffer_right).permute(0, 2, 1, 3)                                               # B * H * C * W
        score = torch.bmm(Q.contiguous().view(-1, w, c),
                          S.contiguous().view(-1, c, w))                                            # (B*H) * W * W
        M_right_to_left = self.softmax(score)

        ### M_{left_to_right}
        Q = self.b1(buffer_right).permute(0, 2, 3, 1)                                               # B * H * W * C
        S = self.b2(buffer_left).permute(0, 2, 1, 3)                                                # B * H * C * W
        score = torch.bmm(Q.contiguous().view(-1, w, c),
                          S.contiguous().view(-1, c, w))                                            # (B*H) * W * W
        M_left_to_right = self.softmax(score)

        ### valid masks
        V_left_to_right = torch.sum(M_left_to_right.detach(), 1) > 0.1
        V_left_to_right = V_left_to_right.view(b, 1, h, w)                                          #  B * 1 * H * W
        V_left_to_right = morphologic_process(V_left_to_right)
        if is_training==1:
            V_right_to_left = torch.sum(M_right_to_left.detach(), 1) > 0.1
            V_right_to_left = V_right_to_left.view(b, 1, h, w)                                      #  B * 1 * H * W
            V_right_to_left = morphologic_process(V_right_to_left)

            M_left_right_left = torch.bmm(M_right_to_left, M_left_to_right)
            M_right_left_right = torch.bmm(M_left_to_right, M_right_to_left)

        ### fusion
        buffer = self.b3(x_right).permute(0,2,3,1).contiguous().view(-1, w, c)                      # (B*H) * W * C
        buffer = torch.bmm(M_right_to_left, buffer).contiguous().view(b, h, w, c).permute(0,3,1,2)  #  B * C * H * W
        out = self.fusion(torch.cat((buffer, x_left, V_left_to_right), 1))

        ## output
        if is_training == 1:
            return out, \
               (M_right_to_left.contiguous().view(b, h, w, w), M_left_to_right.contiguous().view(b, h, w, w)), \
               (M_left_right_left.view(b,h,w,w), M_right_left_right.view(b,h,w,w)), \
               (V_left_to_right, V_right_to_left)
        if is_training == 0:
            return out

class APAM(nn.Module):
    def __init__(self, channels):
        super(APAM, self).__init__()
        self.b1 = ASPPB(64)
        self.b2 = ASPPB(64)
        self.b3 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.softmax = nn.Softmax(-1)
        self.rb = ResB(64)
        self.fusion = nn.Conv2d(channels * 2, channels, 1, 1, 0, bias=True)
        #self.fusion = nn.Conv2d(channels * 2 + 1, channels, 1, 1, 0, bias=True)
    def __call__(self, x_1, x_2, is_training):
        b, c, h, w = x_1.shape
        buffer_1 = self.rb(x_1)
        buffer_2 = self.rb(x_2)

        ### M_{right_to_left} if input = x_left, x_right
        Q = self.b1(buffer_1).permute(0, 2, 3, 1)                                                # B * H * W * C
        S = self.b2(buffer_2).permute(0, 2, 1, 3)                                               # B * H * C * W
        score = torch.bmm(Q.contiguous().view(-1, w, c),
                          S.contiguous().view(-1, c, w))                                            # (B*H) * W * W
        M_2_to_1 = self.softmax(score)

        ### M_{left_to_right}
        Q = self.b1(buffer_2).permute(0, 2, 3, 1)                                               # B * H * W * C
        S = self.b2(buffer_1).permute(0, 2, 1, 3)                                                # B * H * C * W
        score = torch.bmm(Q.contiguous().view(-1, w, c),
                          S.contiguous().view(-1, c, w))                                            # (B*H) * W * W
        M_1_to_2 = self.softmax(score)

        ### valid masks
        V_1_to_2 = torch.sum(M_1_to_2.detach(), 1) > 0.1  # gateway = 0.1 
        V_1_to_2 = V_1_to_2.view(b, 1, h, w)                                          #  B * 1 * H * W
        
        if is_training==1:
            V_2_to_1 = torch.sum(M_2_to_1.detach(), 1) > 0.1
            V_2_to_1 = V_2_to_1.view(b, 1, h, w)                                      #  B * 1 * H * W

            M_1_2_1 = torch.bmm(M_2_to_1, M_1_to_2)
            M_2_1_2 = torch.bmm(M_1_to_2, M_2_to_1)

        ### fusion
        buffer = self.b3(x_2).permute(0,2,3,1).contiguous().view(-1, w, c)                      # (B*H) * W * C
        buffer = torch.bmm(M_2_to_1, buffer).contiguous().view(b, h, w, c).permute(0,3,1,2)  #  B * C * H * W
        out = self.fusion(torch.cat((buffer, x_1), 1))

        ## output
        if is_training == 1:
            return out, \
               (M_2_to_1.contiguous().view(b, h, w, w), M_1_to_2.contiguous().view(b, h, w, w)), \
               (M_1_2_1.view(b,h,w,w), M_2_1_2.view(b,h,w,w)), \
               (V_1_to_2, V_2_to_1)
        if is_training == 0:
            return out, M_2_to_1.contiguous().view(b, h, w, w)


class CALayer(nn.Module):
    def __init__(self, channels, reduction = 16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
                nn.Conv2d(channels, channels // reduction, 1, padding = 0, bias = True),
                nn.ReLU(inplace = True),
                nn.Conv2d(channels // reduction, channels, 1, padding = 0, bias = True),
                nn.Sigmoid()
        )
    
    def forward(self, x):
        out = self.avg_pool(x)
        out = self.conv_du(out)
        return x * out
    
class RCAB(nn.Module):
    def __init__(self, channels, reduction = 4, bias = True, bn = False):
        super(RCAB, self).__init__()
        self.body = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding = 1, bias = bias),
                nn.ReLU(inplace = True),
                nn.Conv2d(channels, channels, 3, padding = 1, bias = bias),
        )
        self.ca = CALayer(channels, reduction)
        #add 0809
    def forward(self, x):
        res = self.body(x)
        res = self.ca(res)
        out = x + res
        return out

def morphologic_process(mask):
    device = mask.device
    b, _, _, _ = mask.shape
    mask = 1 - mask
    # mask = ~mask
    mask_np = mask.cpu().numpy().astype(bool)
    mask_np = morphology.remove_small_objects(mask_np, 20, 2)
    mask_np = morphology.remove_small_holes(mask_np, 10, 2)
    for idx in range(b):
        buffer = np.pad(mask_np[idx,0,:,:],((3,3),(3,3)),'constant')
        buffer = morphology.binary_closing(buffer, morphology.disk(3))
        mask_np[idx,0,:,:] = buffer[3:-3,3:-3]
    # mask_np = ~ mask_np
    mask_np = 1 - mask_np
    mask_np = mask_np.astype(float)

    return torch.from_numpy(mask_np).float().to(device)
    

    def __init__(self):
        super(DRRN, self).__init__()
        self.model_name = 'DRRN_standard'
        self.input = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.output = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

        # weights initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))

    def forward(self, x):
        residual = x
        inputs = self.input(self.relu(x))
        out = inputs
        for _ in range(25):
            out = self.conv2(self.relu(self.conv1(self.relu(out))))
            out = torch.add(out, inputs)

        out = self.output(self.relu(out))
        out = torch.add(out, residual)
        return out


    def __init__(self):
        super(VDSR, self).__init__()
        self.model_name = 'VDSR_standard'
        self.residual_layer = self.make_layer(Conv_ReLU_Block, 18)
        self.input = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.output = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
    
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                
    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)
    
    def forward(self, x):
        residual = x
        out = self.relu(self.input(x))
        out = self.residual_layer(out)
        out = self.output(out)
        out = torch.add(out,residual)
        return out
        


    def __init__(self, upscale_factor):
        super(DCSSR_a1, self).__init__()
        self.upscale_factor = upscale_factor
        self.model_name = 'DCSSR_a1'
        ### feature extraction
        ResBlock = RCAB # or ResB, or RCASAB
        ASPPBlock = DenseASPPB #ResASPPB 
        self.init_feature_lr_left = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            ResBlock(64),
            ASPPBlock(64),
            ResBlock(64),
            ASPPBlock(64),
            ResBlock(64),
        )
        ### upscaling
        self.upscale_left = nn.Sequential(
            ResBlock(64),
            ResBlock(64),
            ResBlock(64),
            ResBlock(64)
        )
        self.upscale_left_2 = nn.Sequential(
            nn.Conv2d(64 + 3, 64 * upscale_factor ** 2, 1, 1, 0, bias=False),
            nn.PixelShuffle(upscale_factor),
            nn.Conv2d(64, 3, 3, 1, 1, bias=False),
            nn.Conv2d(3, 3, 3, 1, 1, bias=False)
        )
        
    def forward(self, x_left, is_training):
        ### feature extraction
        buffer_lr_left = self.init_feature_lr_left(x_left)

        ### parallax attention
        buffer_0 = buffer_lr_left
        ### upscaling
        sr_left = self.upscale_left(buffer_0)
        sr_left = torch.cat((sr_left, x_left), 1)
        sr_left = self.upscale_left_2(sr_left)

        return sr_left                      

