import torch
import torch.nn.functional as F

import numpy as np
import pdb, os, argparse
from scipy import misc
import cv2
# from model.fpn_model import Yolact
from model.vgg2_models import Back_VGG
from data import test_dataset
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--batchsize', type=int, default=2, help='testing batch size')
model = Back_VGG(channel=32)
model.load_state_dict(torch.load('./models/scribble_50.pth',map_location=torch.device('cpu')))
model.cuda()
model.eval()
opt = parser.parse_args()

dataset_path = './testing/img/'


#model = Yolact(8,32)



#model.train()
test_datasets = ['EORSSD']

for dataset in test_datasets:
    save_path = './results/ResNet50/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = dataset_path + dataset + '/'
    test_loader = test_dataset(image_root, opt.testsize)
    for i in range(test_loader.size):
        print  (i)
        image, HH, WW, name = test_loader.load_data()
        image = image.cuda()
        res0, res1, res1= model(image)
        res=res1[0]
        # print(res.shape)
        x_visualize=res1[1]
        x_visualize = x_visualize.cpu().detach().numpy() #用
        # print(x_visualize.shape)
        x_visualize = np.max(x_visualize,axis=1).reshape(352,352) #
        x_visualize = (((x_visualize - np.min(x_visualize))/(np.max(x_visualize)-np.min(x_visualize)))*255).astype(np.uint8) #归一化并映射到0-255的整数，方便伪彩色化
        savedir = './feature/'
        if not os.path.exists(savedir):
            os.mkdir(savedir)   
        x_visualize = cv2.applyColorMap(x_visualize, cv2.COLORMAP_JET)  # 伪彩色处理  
        cv2.imwrite('./feature/'+name,x_visualize) #保存可视化图像 
        res = F.upsample(res, size=[WW, HH], mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        misc.imsave(save_path+name, res)