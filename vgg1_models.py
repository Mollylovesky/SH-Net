import torch#定义网络
import torch.nn as nn#定义网络
import numpy as np
import cv2
from model.vgg import B2_VGG

import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize

import numpy as np
from PIL import Image
import numpy as np

from data import get_loader2

import torch.utils.model_zoo
import torch
from torch.autograd import Variable

#from module.BaseBlocks import BasicConv2d
from utils.tensor_ops import cus_sample, upsample_add


from torch.utils.checkpoint import checkpoint


use_jit = False #torch.cuda.device_count() <= 1
# if not use_jit:
#     print('Multiple GPUs detected! Turning off JIT.')
ScriptModuleWrapper = torch.jit.ScriptModule if use_jit else nn.Module
script_method_wrapper = torch.jit.script_method if use_jit else lambda fn, _rcn=None: fn

#model.cuda()
#params = model.parameters()#optimizer = torch.optim.Adam(params, opt.lr)

#opt = parser.parse_args()

'''''
image_root = 'D:/Scribble_Saliency-master/data/data1/img/'
gt_root = 'D:/Scribble_Saliency-master/data/data1/gt/'  #二值图
mask_root = 'D:/Scribble_Saliency-master/data/data1/mask/'
edge_root = 'D:/Scribble_Saliency-master/data/data1/edge/'
grayimg_root = 'D:/Scribble_Saliency-master/data/data1/gray/'
train_loader = get_loader2(image_root, gt_root, mask_root, grayimg_root, edge_root, batchsize=1, trainsize=352)



for i, pack in enumerate(train_loader, start=1):  # 用于可迭代\可遍历的数据对象组合为一个索引序列
    #optimizer.zero_grad()
    images, gts, masks, grays, edges = pack
    images = Variable(images)
    gts = Variable(gts)
    masks = Variable(masks)
    grays = Variable(grays)
    edges = Variable(edges)

    images = images.cuda()

    gts = gts.cuda()
    masks = masks.cuda()
    grays = grays.cuda()
    edges = edges.cuda()

    gts = F.interpolate(gts, size=(22, 22), mode='bilinear', align_corners=True)
    images = F.interpolate(images, size=(22, 22), mode='bilinear', align_corners=True)
    grays = F.interpolate(grays, size=(22, 22), mode='bilinear', align_corners=True)
    gts = gts.bool().cuda()

'''''

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()#调用父类的构造函数，传入类名和self
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        reduction_dim = 256
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()#激活函数（0-1）
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self, n_feat, kernel_size=3, reduction=16,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(self.default_conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def default_conv(self, in_channels, out_channels, kernel_size, bias=True):
        return nn.Conv2d(in_channels, out_channels, kernel_size,padding=(kernel_size // 2), bias=bias)

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
###########################################################
class Edge_Module(nn.Module):

    def __init__(self, in_fea=[64, 256, 512], mid_fea=32):
        super(Edge_Module, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_fea[0], mid_fea, 1)
        self.conv4 = nn.Conv2d(in_fea[1], mid_fea, 1)
        self.conv5 = nn.Conv2d(in_fea[2], mid_fea, 1)
        self.conv5_2 = nn.Conv2d(mid_fea, mid_fea, 3, padding=1)
        self.conv5_4 = nn.Conv2d(mid_fea, mid_fea, 3, padding=1)
        self.conv5_5 = nn.Conv2d(mid_fea, mid_fea, 3, padding=1)

        self.classifer = nn.Conv2d(mid_fea * 3, 1, kernel_size=3, padding=1)
        self.rcab = RCAB(mid_fea * 3)


    def forward(self,f_x_4_3_2_1,  f_x_4_3_2,  f_x_4_3):#x2,x4,x5   128 512 512
        _, _, h, w = f_x_4_3_2_1.size()

        edge2_fea = self.relu(self.conv2(f_x_4_3_2_1))

        edge2 = self.relu(self.conv5_2(edge2_fea))

        edge4_fea = self.relu(self.conv4(f_x_4_3_2))
        edge4 = self.relu(self.conv5_4(edge4_fea))

        edge5_fea = self.relu(self.conv5(f_x_4_3))
        edge5 = self.relu(self.conv5_5(edge5_fea))


        edge4 = F.interpolate(edge4, size=(h, w), mode='bilinear', align_corners=True)#指定输出形状的上采样torch.Size（[7,32,352,352]）

        edge5 = F.interpolate(edge5, size=(h, w), mode='bilinear', align_corners=True)#指定输出形状的上采样#用双线性插值的方法进行上采样torch。Size（[7,32,352,352]）


        edge = torch.cat([ edge2, edge4, edge5], dim=1)
        edge = self.rcab(edge)#96

        edge = self.classifer(edge)#1

        return edge


#Dense ASPP
class _DenseAsppBlock(nn.Sequential):
    """ ConvNet block for building DenseASPP. """

    def __init__(self, input_num, num1, num2, dilation_rate, drop_out, bn_start=True):
        super(_DenseAsppBlock, self).__init__()
        self.asppconv = torch.nn.Sequential()
        if bn_start:
            self.asppconv = nn.Sequential(
                nn.BatchNorm2d(input_num),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=input_num, out_channels=num1, kernel_size=1),
                nn.BatchNorm2d(num1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=num1, out_channels=num2, kernel_size=3,
                          dilation=dilation_rate, padding=dilation_rate)
            )
        else:
            self.asppconv = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=input_num, out_channels=num1, kernel_size=1),
                nn.BatchNorm2d(num1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=num1, out_channels=num2, kernel_size=3,
                          dilation=dilation_rate, padding=dilation_rate)
            )
        self.drop_rate = drop_out

    def forward(self, _input):
        feature = self.asppconv(_input)

        if self.drop_rate > 0:
            feature = F.dropout2d(feature, p=self.drop_rate, training=self.training)

        return feature
    #####################################################################################残留代码

class Classifier_Module(nn.Module):
    def __init__(self,dilation_series,padding_series,NoLabels, input_channel):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation,padding in zip(dilation_series,padding_series):
            self.conv2d_list.append(nn.Conv2d(input_channel,NoLabels,kernel_size=3,stride=1, padding =padding, dilation = dilation,bias = True))
        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list)-1):
            out += self.conv2d_list[i+1](x)
        return out
##############################################################################################
class _AtrousSpatialPyramidPoolingModule(nn.Module):#ASPP代码模块（融合edge的信息）
    '''
    operations performed:
      1x1 x depth
      3x3 x depth dilation 6
      3x3 x depth dilation 12
      3x3 x depth dilation 18
      image pooling
      concatenate all together
      Final 1x1 conv
    '''
#将backnone和最后输出的edge信息在一起输入了aspp
    def __init__(self, in_dim, reduction_dim=256, output_stride=16, rates=[6, 12, 18]):
        super(_AtrousSpatialPyramidPoolingModule, self).__init__()

        # Check if we are using distributed BN and use the nn from encoding.nn
        # library rather than using standard pytorch.nn

        if output_stride == 8:
            rates = [2 * r for r in rates]
        elif output_stride == 16:
            pass
        else:
            raise 'output stride of {} not supported'.format(output_stride)

        self.features = []
        # 1x1
        self.features.append(
            nn.Sequential(nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                          nn.ReLU(inplace=True)))
        # other rates（其他的速率）
        for r in rates:
            self.features.append(nn.Sequential(
                nn.Conv2d(in_dim, reduction_dim, kernel_size=3,
                          dilation=r, padding=r, bias=False),
                nn.ReLU(inplace=True)
            ))
        self.features = torch.nn.ModuleList(self.features)

        # img level features
        self.img_pooling = nn.AdaptiveAvgPool2d(1)
        self.img_conv = nn.Sequential(
            nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
            nn.ReLU(inplace=True))
        self.edge_conv = nn.Sequential(
            nn.Conv2d(1, reduction_dim, kernel_size=1, bias=False),
            nn.ReLU(inplace=True))

    def forward(self, x, edge):

        x_size = x.size()

        img_features = self.img_pooling(x)
        img_features = self.img_conv(img_features)
        img_features = F.interpolate(img_features, x_size[2:],
                                     mode='bilinear', align_corners=True)
        out = img_features###torch.Size([7, 32, 22, 22])


        edge_features = F.interpolate(edge, x_size[2:],
                                      mode='bilinear', align_corners=True)

        edge_features = self.edge_conv(edge_features)

        out = torch.cat((out, edge_features), 1)######torch.Size([7, 64, 22, 22])



        for f in self.features:
            y = f(x)
            out = torch.cat((out, y), 1)#二者concate一起经过aspp[上采样得到的edge_features和out concate到一起]
        return out


##################################################################################################################
class FPN(ScriptModuleWrapper):


    def __init__(self, in_channels):
        super().__init__()
        self.interpolation_mode = 'bilinear'  # 'bilinear'

        # num_features = 256
        self.lat_layers4 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1),#512->
            nn.ReLU(True)
        )
        self.lat_layers3 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),#256->128
            nn.ReLU(True)
        )
        self.lat_layers2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),#128->64
            nn.ReLU(True)
        )
        self.lat_layers1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1),#64->64
            nn.ReLU(True)
        )

        self.interpolation_mode = 'bilinear'


    @script_method_wrapper
    def forward(self,  x1, x2, x3, x4):

        up_x4 = F.interpolate(x4, size=(44, 44), mode=self.interpolation_mode, align_corners=False)
        f_x_4_3 = self.lat_layers4(up_x4)+x3

        up_x4_x3 = F.interpolate(f_x_4_3, size=(88, 88), mode=self.interpolation_mode, align_corners=False)
        f_x_4_3_2 = self.lat_layers3(up_x4_x3)+x2

        up_x4_x3_x2 = F.interpolate(f_x_4_3_2, size=(176, 176), mode=self.interpolation_mode, align_corners=False)
        f_x_4_3_2_1 = self.lat_layers2(up_x4_x3_x2) + x1
        f_x_4_3_2_1 = self.lat_layers1(f_x_4_3_2_1)
       # up_x4_x3_x2_x1 = F.interpolate(f_x_4_3_2_1, size=(352, 352), mode=self.interpolation_mode, align_corners=False)



        return  f_x_4_3 , f_x_4_3_2 , f_x_4_3_2_1   #88_256,176_128,352_64###



class Back_VGG(nn.Module):
    def __init__(self, channel=32):
        super(Back_VGG, self).__init__()
        self.vgg = B2_VGG()

#        self.freeze_bn()
        self.selected_layers = list(range(0, 4))

        self.rcab3 = RCAB(512)
        self.fpn = FPN(channel)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)#上采样
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        #self.downsample2 = nn.Downsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.layer5 = self._make_pred_layer(Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], channel, 512)
        self.relu = nn.ReLU(True)
        self.edge_layer = Edge_Module()
        self.aspp = _AtrousSpatialPyramidPoolingModule(512, 32,
                                                       output_stride=16)#512
        self.classification = nn.Sequential(
            nn.Dropout2d(p=0.1),
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channel, out_channels=1, kernel_size=3, padding=1)

        )

        self.rcab_feat = RCAB(channel * 6)#192
        self.sal_conv = nn.Conv2d(1, channel, kernel_size=3, padding=1, bias=False)#3*3卷积
        self.edge_conv = nn.Conv2d(1, channel, kernel_size=3, padding=1, bias=False)#3*3卷积
        self.rcab_sal_edge = RCAB(channel*2)#64
        self.layer6 = self._make_pred_layer(Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], 1, channel*2)
        self.after_aspp_conv5 = nn.Conv2d(channel * 6, channel, kernel_size=1, bias=False)
        self.after_aspp_conv2 = nn.Conv2d(128, channel, kernel_size=1, bias=False)#128
        self.final_sal_seg = nn.Sequential(
            nn.Conv2d(channel*2, channel, kernel_size=3, padding=1, bias=False),#3*3卷积
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=3, padding=1, bias=False),#3*3卷积
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, 1, kernel_size=1, bias=False))#1*1卷积
        self.fuse_canny_edge = nn.Conv2d(2, 1, kernel_size=1, padding=0, bias=False)#1*1卷积
        self.fused_edge_sal = nn.Conv2d(64, 1, kernel_size=3, padding=1, bias=False)#3*3卷积



        '''
        def sum_and_avge(gt_bin):  # qupingjun
            count = 0
            sum = 0
            img = np.copy(gt_bin)
            rows, cols = img.shape[:2]
            for row in range(rows):
                for col in range(cols):
                    if img[row, col] != 0:
                        sum = sum + np.abs(img[row, col])
                        count = count + 1
            aver = sum / count
            aver = np.rint(aver)
            return aver

        self.sumandaver = sum_and_avge(gt_bin=gts).cpu()
        '''



    def _make_pred_layer(self, block, dilation_series, padding_series, NoLabels, input_channel):
        return block(dilation_series, padding_series, NoLabels, input_channel)

    def input_transform(crop_size, upscale_factor):
        return Compose([
            CenterCrop(crop_size),
            Resize(crop_size // upscale_factor),
            ToTensor(),
        ])

    def upsample_add(*xs):
        y = xs[-1]
        for x in xs[:-1]:
            y = y + F.interpolate(x, size=y.size()[2:], mode="bilinear", align_corners=False)
        return y
    '''
    def sum_and_avge(self, gt_bin):  #qupingjun
        count = 0
        sum = 0
        img = np.copy(gt_bin)
        rows, cols = img..shape[:2]
        for row in range(rows):
            for col in range(cols):
                if img[row,col] !=0:
                    sum = sum + np.abs(img[row,col])
                    count = count + 1
        aver = sum / count 
        aver = np.rint(aver)
        return aver
    '''

    def forward(self, input):

        x_size = input.size()

        x = self.vgg.conv1(input) ## 352*352*64
        x1 = self.vgg.conv2(x)  ## 176*176*128
        x2 = self.vgg.conv3(x1)   ## 88*88*256
        x3 = self.vgg.conv4(x2)  ## 44*44*512
        x4 = self.vgg.conv5(x3)  ## 22*22*512

        f_x_4_3, f_x_4_3_2, f_x_4_3_2_1= self.fpn(x1, x2, x3, x4)  # 22*22_256,44*44_128,88*88_64
        f_x_4_3_2_1 = F.interpolate(f_x_4_3_2_1, size=(352, 352), mode='bilinear', align_corners=True)

        f_x_4_3_2 = F.interpolate(f_x_4_3_2, size=(176, 176), mode='bilinear', align_corners=True)

        f_x_4_3 = F.interpolate(f_x_4_3, size=(88, 88), mode='bilinear', align_corners=True)
       # f_x_4_3 = self.rcab3(f_x_4_3)



        edge_map = self.edge_layer(f_x_4_3_2_1, f_x_4_3_2,  f_x_4_3)#64，256，512


        edge_out = torch.sigmoid(edge_map)#176


        ####
        #im_arr = x.cpu().numpy().transpose((0, 2, 3, 1)).astype(np.uint8)
        im_arr = input.detach().cpu().numpy().transpose((0, 2, 3, 1)).astype(np.uint8)
        ########################################################################canny检测图像边缘
        canny = np.zeros((x_size[0], 1, x_size[2], x_size[3]))
        for i in range(x_size[0]):
            canny[i] = cv2.Canny(im_arr[i], 10, 100)
        canny = torch.from_numpy(canny).cuda().float()



        ##################################################
        #edge_out = F.interpolate(edge_out, size=(352, 352), mode='bilinear', align_corners=True)
        cat = torch.cat((edge_out, canny), dim=1)  # 按维数为1横拼接torch.Size([7, 2, 352, 352])

        acts = self.fuse_canny_edge(cat)  # 2-1

        acts = torch.sigmoid(acts)  # torch.Size([6, 1, 352, 352])

        '''
        #gi = x4 * gts
        #count = 0
        #sum = 0
        #img = np.copy(gi.cpu())
        #rows, cols = img.shape[:2]

        #for row in range(rows):
           #for col in range(cols):
                #if img[row, col].any() != 0:
                    #sum = sum + np.abs(img[row, col])
                    #count = count + 1
        #aver = sum / count
        #aver = np.rint(aver)
        #gii = aver * x4
        '''

        #batchsize = 6
        #means = []

        #gi = torch.mul(x4, gts)

        #gi = gi.view(x.size(0), x.size(1), -1)





        #for i in range(batchsize):

            #gi = gi[i, :, :, :].detach()
            #kernel = torch.masked_select(x4[i, :, :, :], gts)
            #means = torch.mean(kernel)
            #mean = torch.mean(gi)
            #####################################################
            #print(mean.size())
            #x4 = torch.mul(x4, means)
            #x4[i, :, :, :] = x4[i, :, :, :] * means

            ###################################################
            #means.append(means.unsqueeze(0))

        #means = torch.cat(means, dim=0)
        # a = torch.mul(x4, means)

#        means.append(means.unsqueeze(0))

        #h, w = x4.shape[2:4]
        #x4 = x4.view(x4.size(0), x4.size(1), h*w)# 1 * (b*c) * k * k
        #gts = gts.view(kernel.size(0), kernel.size.size(1), h*w)  # (b*c) * 1 * H * W

        #out = F.conv2d(x4, kernel, groups=h * w)
        #out = out.view(out.size(0), out.size(0), out.size(2), out.size(3))



        #print(means)

        #x4 = torch.mul(x4, means)

#################################################
        #gi = torch.mul(x4, gts)  #前景
        #gm = torch.mul(x4, masks-gts) #背景
        #mask1 = gi.bool()

        #kernel1 = torch.masked_select(gi, mask1)


        #kernel.unsqueeze(1).unsqueeze(2)
        #means1 = torch.mean(kernel1)

        #x4_1 = torch.mul(x4, means1)





        #h, w = x4.shape[2:]
        #x4 = x4.view(x4.size(0), x4.size(1), h * w)



        #print(means.size())





        #means.unsqueeze(0)

        #back = torch.neg(masks - gts)
        #summ = back + gts
        #x4 = torch.cat((grays, x4), dim=1)



        #x4 = x4 * masks + x4
        cats = self.aspp(x4, acts)#torch.Size([6, 192, 22, 22])


        x_conv5 = self.after_aspp_conv5(cats)
        x1 = self.after_aspp_conv2(x1)
        x_conv5_up = F.interpolate(x_conv5, x1.size()[2:], mode='bilinear', align_corners=True)#x2.size()[2:]



        #x_conv2 = self.downsample2(x_conv2)

        feat_fuse = torch.cat([x_conv5_up, x1], 1)

        sal_init = self.final_sal_seg(feat_fuse)
        sal_init = F.interpolate(sal_init, x_size[2:], mode='bilinear')

        sal_feature = self.sal_conv(sal_init)
        edge_feature = self.edge_conv(edge_map)
        #edge_feature = F.interpolate(edge_feature, size=(352, 352), mode='bilinear', align_corners=True)
        sal_edge_feature = self.relu(torch.cat((sal_feature, edge_feature), 1))##########
        sal_edge_feature = self.rcab_sal_edge(sal_edge_feature)
        sal_ref = self.fused_edge_sal(sal_edge_feature)
        return sal_init, edge_map, sal_ref

