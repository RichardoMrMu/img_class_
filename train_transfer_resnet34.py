# -*- coding:utf-8 -*-
# @Time     : 2019-09-26 8:45
# @Author   : Richardo Mu
# @FILE     : train_transfer.PY
# @Software : PyCharm
# -*- coding:utf-8 -*-
# @Time     : 2019-09-25 16:01
# @Author   : Richardo Mu
# @FILE     : train_scratch.PY
# @Software : PyCharm
import torch
from torch import optim,nn
from torch.utils.data import DataLoader
import os
from torchvision.models import resnet34
from utils import Flatten
from HustData_LOAD import HustData_LOAD
# import argparse
# import time
import sys 

# time_start = time.time()
# parser = argparse.ArgumentParser(description="Image Classification -ResNet18")
# parser.add_argument('--epochs',type=int,default=10,
#                     help='epochs limit (default 10)')
# parser.add_argument('--lr',type=float,default=1e-3,
#                     help='initial learning rate (default :1e-3)')
# parser.add_argument('--seed',type=int,default=1234,
#                     help='random seed (default: 1234)')
# parser.add_argument('--batchsz',type=int,default=32,
#                     help='data batch_size(default=32)')
# parser.add_argument('--num_class',type=int,default=11,
#                     help='number of classification u want to ')
# args = parser.parse_args()
# torch.manual_seed(args.seed)
# cwd = os.getcwd()

def test_img():
    model.eval()
    img = sys.argv[1]
    test_img_db = HustData_LOAD(img,resize=224)
    test_img_dataloder = DataLoader(test_img_db,batch_size=32,shuffle=False,
                                    num_workers=2)
    for x in test_img_dataloder:
        with torch.no_grad():
            logits = model(x)
            # print(logits)
            layer = nn.Softmax()
            logits = layer(logits)
            # print(logits)
            pred = logits.argmax(dim=1)[0]
            poss = logits.max()
            if poss<0:
                pred=11
                print(int(pred))
                return pred
            else:
                print(int(pred))
                return int(pred)



    
if __name__ == '__main__':
    trained_model = resnet34()
    model = nn.Sequential(*list(trained_model.children())[:-1],  # [b,512,1,1]
                          Flatten(),  # [b,512,1,1]=>[b,512]
                          nn.Linear(512, 11)
                          )
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criteon = nn.CrossEntropyLoss()

    model.load_state_dict(torch.load('best_transfer_resnet34.mdl',map_location='cpu'))
    # print('loaded from ckpt!')
    re = test_img()
    # end = time.time()
    # print(end-time_start)
    # os.system("python send.py %s"%re)