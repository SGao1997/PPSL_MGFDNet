# -*- coding: utf-8 -*-
import torch
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
import os
import numpy as np
import Models
import dataset_half_bz24
from tqdm import tqdm
import random
from PIL import Image
import torch.utils.data as Data
from torch.optim.lr_scheduler import _LRScheduler
import torch.nn as nn

class PolyScheduler(_LRScheduler):
    r"""
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        total_steps (int): The total number of steps in the cycle. Note that
            if a value is not provided here, then it must be inferred by providing
            a value for epochs and steps_per_epoch.
            Default: None

    """

    def __init__(self,
                 optimizer,
                 total_steps,
                 power=1.0,
                 min_lr=0,
                 verbose=False):

        self.optimizer = optimizer
        # self.by_epoch = by_epoch
        self.min_lr = min_lr
        self.power = power
        self.total_steps = total_steps

        super(PolyScheduler, self).__init__(optimizer, -1, False)
    def _format_param(self, name, optimizer, param):
        """Return correctly formatted lr/momentum for each param group."""
        if isinstance(param, (list, tuple)):
            if len(param) != len(optimizer.param_groups):
                raise ValueError("expected {} values for {}, got {}".format(
                    len(optimizer.param_groups), name, len(param)))
            return param
        else:
            return [param] * len(optimizer.param_groups)
    def get_lr(self):
        step_num = self.last_epoch
        coeff = (1 - step_num / self.total_steps) ** self.power
        return [(base_lr - self.min_lr) * coeff + self.min_lr for base_lr in self.base_lrs]


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ConstractiveLoss(pred, target):
    constractive_loss = torch.mean((1-target)*torch.pow(pred, 2) +  target * torch.pow(torch.clamp(2 - pred, min=0.0),2))
    return constractive_loss

def main():
    #set random seed
    seed_torch(42)

    #training and val data
    image1_dir = r'E:\gs\BE2BCD\dataset\WHU_Building_dataset\train\image_rename'
    label1_dir = r'E:\gs\BE2BCD\dataset\WHU_Building_dataset\train\label_rename'

    imageval1_dir = r'E:\gs\BE2BCD\dataset\WHUBCD_full\val\A'
    imageval2_dir = r'E:\gs\BE2BCD\dataset\WHUBCD_full\val\B'
    labelval1_dir = r'E:\gs\BE2BCD\dataset\WHUBCD_full\val\OUT' 

    save_dir = r'./model'

    #training parameters. Accroding to the proposed PPSL, the batchsize here is actually equivalent to 16
    batch_size = 24
    epochs = 120
    LR = 5e-3

    #training dataloader
    train_data = dataset_half_bz24.changeDatasets(image1_dir, label1_dir, is_Transforms=True)
    train_dataloader = Data.DataLoader(train_data, num_workers=4, batch_size=batch_size, shuffle=True, collate_fn=train_data.be2bcd_sup, drop_last=True)

    #test dataloader
    test_data = dataset_half_bz24.testDatasets(imageval1_dir, imageval2_dir, labelval1_dir, is_Transforms=False)
    test_dataloader = Data.DataLoader(test_data, num_workers=4, batch_size=batch_size, shuffle=False)

    #model init
    net = Models.changeNet(backbone='resnet18').cuda()
    optimizer = torch.optim.SGD(net.parameters(), lr=LR, momentum=0.9, dampening=0, weight_decay=1e-3, nesterov=False)
    scheduler = PolyScheduler(optimizer, epochs*len(train_dataloader), power=0.9)

    #Records
    jilu = open(os.path.join(save_dir, "jilu.txt"), 'w')

    #loss
    bceloss_train = torch.nn.BCELoss()
    bceloss_test = torch.nn.BCELoss()
    
    for epoch in range(epochs):

        Loss = np.array([0, 0], dtype=np.float32)  # [train_loss, val_loss]
        tfpn_t1 = np.array([0, 0, 0, 0])  #[tp, tn, fp, fn]
        LB = 0
        LC = 0
        LD = 0

        net.train()
        for i, data in tqdm(enumerate(train_dataloader), desc="train, epoch {}".format(epoch)):
            
            image_t1 = data['image1'].cuda()
            image_t2 = data['image2'].cuda()
            building = data['image1_label'].cuda()
            change = data['change_label'].cuda()

            net.zero_grad()
            bout, cout1, metric = net(image_t1, image_t2)
            building = nn.functional.interpolate(building, size=(128, 128),mode='nearest')
            loss_b = bceloss_train(bout, building)
            loss_c = bceloss_train(cout1, change)
            loss_d = 0.25*ConstractiveLoss(metric[0], nn.functional.interpolate(change, size=(128, 128),mode='nearest')) +\
                     0.25*ConstractiveLoss(metric[1], nn.functional.interpolate(change, size=(128//2, 128//2),mode='nearest')) +\
                     0.25*ConstractiveLoss(metric[2], nn.functional.interpolate(change, size=(128//4, 128//4),mode='nearest')) +\
                     0.25*ConstractiveLoss(metric[3], nn.functional.interpolate(change, size=(128//8, 128//8),mode='nearest'))

            loss = loss_b + loss_c + loss_d
            loss.backward()
            optimizer.step()
            scheduler.step()

            Loss[0] += loss.item()
            LB += loss_b.item()
            LC += loss_c.item()
            LD += loss_d.item()
            
        torch.save(net.state_dict(), os.path.join(save_dir, 
                    'model{}.pth'.format(Loss[0] / len(train_dataloader))))

        net.eval()
        for j, testdata in tqdm(enumerate(test_dataloader), desc="val, epoch {}".format(epoch)):
            with torch.no_grad():
                
                test_t1 = testdata['image1']
                test_t2 = testdata['image2']
                tlabel_t1 = testdata['change']

                _, tout1, _ = net(test_t1.cuda(), test_t2.cuda())
                tloss = bceloss_test(tout1, tlabel_t1.cuda())

                Loss[1] += tloss.item()

                tout1 = (tout1.cpu().numpy() > 0.5).astype('uint8')*255          

                tlabel_t1 = (tlabel_t1.cpu().numpy()*255).astype('uint8')

                tfpn_t1[0] += float(len(np.where((tout1 == 255) & (tlabel_t1 == 255))[0]))
                tfpn_t1[1] += float(len(np.where((tout1 == 0) & (tlabel_t1 == 0))[0]))
                tfpn_t1[2] += float(len(np.where((tout1 == 255) & (tlabel_t1 == 0))[0]))
                tfpn_t1[3] += float(len(np.where((tout1 == 0) & (tlabel_t1 == 255))[0]))
  
        precision_t1 = (tfpn_t1[0] + 1e-8) / (tfpn_t1[0] + tfpn_t1[2] + 1e-8)
        recall_t1 = (tfpn_t1[0] + 1e-8) / (tfpn_t1[0] + tfpn_t1[3] + 1e-8)
        f1_t1 = 2 * ((precision_t1 * recall_t1 + 1e-8) / (precision_t1 + recall_t1 + 1e-8))

        print('lr:',optimizer.param_groups[0]['lr'])
        print("Epoch:", epoch, "; train loss:", Loss[0] / len(train_dataloader),
                                "\n",
                                "; building loss:", LB / len(train_dataloader),
                                "\n",
                                "; change loss:", LC / len(train_dataloader),
                                "\n",
                                "; distance loss:", LD / len(train_dataloader),
                                "\n",
                                "; val loss:", Loss[1] / len(test_dataloader),
                                "\n",
                                '; val precison:', precision_t1,
                                "\n",  
                                '; val recall:', recall_t1,
                                "\n",  
                                '; val f1:', f1_t1)
                
        jilu.write('epoch:{}\t, train loss:{}\t, val loss:{}\t, val pre:{}\t, val recall:{}\t, val f1:{}\n'.format(
                        epoch, Loss[0] / len(train_dataloader), 
                               Loss[1] / len(test_dataloader), 
                               precision_t1, 
                               recall_t1, 
                               f1_t1, ))

    jilu.close()

if __name__ == '__main__':
    main()

