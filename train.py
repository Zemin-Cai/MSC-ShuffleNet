import argparse
import os
from collections import OrderedDict
from glob import glob
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
import MSC_ShuffleNet
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm import tqdm
import losses
from dataset import Dataset
from metrics import iou_score
from utils import AverageMeter, str2bool
from metrics import DiceLoss
from val import _one_hot_encoder
from utils import dice_coefficient_in_train



ARCH_NAMES = MSC_ShuffleNet.__all__
LOSS_NAMES = losses.__all__
LOSS_NAMES.append('BCEWithLogitsLoss')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=MSC_ShuffleNet,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=1200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=20, type=int,
                        metavar='N', help='mini-batch size (default: 16)')

    # model
    parser.add_argument('--arch', '-a', metavar='ARCH', default='MSC_ShuffleNet')
    parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--input_channels', default=3, type=int,
                        help='input channels')
    parser.add_argument('--num_classes', default=5, type=int,
                        help='number of classes')
    parser.add_argument('--input_w', default=256, type=int,
                        help='image width')
    parser.add_argument('--input_h', default=256, type=int,
                        help='image height')

    # loss
    parser.add_argument('--loss', default='CrossEntropyLoss',
                        choices=LOSS_NAMES,
                        help='loss: ' +
                             ' | '.join(LOSS_NAMES) +
                             ' (default: BCEDiceLoss)')

    # dataset
    parser.add_argument('--dataset', default='isic',
                        help='dataset name')
    parser.add_argument('--img_ext', default='.png',
                        help='image file extension')
    parser.add_argument('--mask_ext', default='.png',
                        help='mask file extension')

    # optimizer
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                             ' | '.join(['Adam', 'SGD']) +
                             ' (default: Adam)')
    parser.add_argument('--lr', '--learning_rate', default=1e-2, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    # scheduler
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-9, type=float,
                        help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2 / 3, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int,
                        metavar='N', help='early stopping (default: -1)')
    parser.add_argument('--cfg', type=str, metavar="FILE", help='path to config file', )

    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--direc', default='./medt', type=str,
                        help='directory to save')
    config = parser.parse_args()

    return config


# args = parser.parse_args()

def train(config, train_loader, model, criterion, optimizer):
    avg_meters = {'loss': AverageMeter(),
                  'dice': AverageMeter()}
    num_classes = config['num_classes']
    model.train()

    pbar = tqdm(total=len(train_loader))
    for input, target, _ in train_loader:
        input = input.cuda()
        target = target.cuda()
        # compute output
        if config['deep_supervision']:
            outputs = model(input)
            ce_loss = 0
            for output in outputs:
                loss += criterion(output, target)
            loss /= len(outputs)
            iou, dice = iou_score(outputs[-1], target)
        else:
            output= model(input)
            ce_loss = criterion(output, target.long())
            # output = torch.argmax(output, dim=1)
            dice_loss = DiceLoss(num_classes)
            loss_dice = dice_loss(output, target, softmax=True)
            loss = 0.4 * ce_loss + 0.6 * loss_dice
            dice = 1 - loss_dice

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['dice'].update(dice.item(), input.size(0))
        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('dice', avg_meters['dice'].avg),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('dice', avg_meters['dice'].avg)])


def validate(config, val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(),
                  'dice': AverageMeter()
                  }

    # switch to evaluate mode
    DSC = 0
    model.eval()
    num_classes = config['num_classes']
    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, _ in val_loader:
            input = input.cuda()
            target = target.cuda()
            # compute output
            if config['deep_supervision']:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                iou, dice = iou_score(outputs[-1], target)
            else:
                output = model(input)
                ce_loss = criterion(output, target.long())
                # output = torch.argmax(output, dim=1)
                dice_loss = DiceLoss(num_classes)
                loss_dice = dice_loss(output, target, softmax=True)
                loss = 0.4 * ce_loss + 0.6 * loss_dice
                dice = 1 - loss_dice
                dice_out = torch.argmax(output, dim=1)
                dice_out = _one_hot_encoder(dice_out).permute(0, 2, 3, 1)
                target = _one_hot_encoder(target).permute(0, 2, 3, 1)
                dice_out = dice_out.cpu().numpy()
                target = target.cpu().numpy()
                dice_coeff = dice_coefficient_in_train(target, dice_out)
                for index in range(dice_coeff.shape[0]):
                    DSC += dice_coeff[index][0]
            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['dice'].update(dice, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('dice', avg_meters['dice'].avg)
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('dice', avg_meters['dice'].avg)]), a


def main():
    config = vars(parse_args())

    if config['name'] is None:
        if config['deep_supervision']:
            config['name'] = '%s_%s_wDS' % (config['dataset'], config['arch'])
        else:
            config['name'] = '%s_%s_woDS' % (config['dataset'], config['arch'])

    os.makedirs('models/%s' % config['name'], exist_ok=True)

    print('-' * 20)
    for key in config:
        print('%s: %s' % (key, config[key]))
    print('-' * 20)

    with open('models/%s/config.yml' % config['name'], 'w') as f:
        yaml.dump(config, f)

    # define loss function (criterion)
    if config['loss'] == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = losses.__dict__[config['loss']]().cuda()

    cudnn.benchmark = True

    # create model

    model = MSC_ShuffleNet.__dict__[config['arch']](config['num_classes'])

    model = model.cuda()
    model = nn.DataParallel(model, device_ids=[0, 1,2])
    print('parameters:', sum([p.numel() for p in model.parameters() if p.requires_grad]))
    params = filter(lambda p: p.requires_grad, model.parameters())
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(
            list(model.parameters()), lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'],
                              nesterov=config['nesterov'], weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError

    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'],
                                                   verbose=1, min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')],
                                             gamma=config['gamma'])
    elif config['scheduler'] == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError

    # Data loading code
    img_ids = glob(os.path.join('inputs', config['dataset'], 'images', '*' + config['img_ext']))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
    train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=42)
    '''
    train_transform = Compose([
        RandomRotate90(),
        transforms.Flip(),
        Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])
    '''
    train_dataset = Dataset(
        img_ids=train_img_ids,
        img_dir=os.path.join('inputs', config['dataset'], 'images'),
        mask_dir=os.path.join('inputs', config['dataset'], 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=None)
    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join('inputs', config['dataset'], 'images'),
        mask_dir=os.path.join('inputs', config['dataset'], 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=None
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('loss', []),
        ('val_loss', []),
        ('val_dice', []),
    ])
    best_iou = 0
    trigger = 0
    for epoch in range(config['epochs']):
        print('Epoch [%d/%d]' % (epoch, config['epochs']))

        # train for one epoch
        train_log = train(config, train_loader, model, criterion, optimizer)
        # evaluate on validation set
        val_log, val_dice_cof = validate(config, val_loader, model, criterion)
        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_log['loss'])

        print('loss %.4f - dice %.4f - val_loss %.4f - val_dice %.4f'
              % (train_log['loss'], train_log['dice'], val_log['loss'], val_log['dice']))

        log['epoch'].append(epoch)
        log['lr'].append(config['lr'])
        log['loss'].append(train_log['loss'])
        log['val_loss'].append(val_log['loss'])
        log['val_dice'].append(val_log['dice'])
        pd.DataFrame(log).to_csv('models/%s/log.csv' %
                                 config['name'], index=False)

        trigger += 1

        if (epoch % 10) == 0:
            torch.save(model.state_dict(), 'models/model_%s.pth' % epoch)
        if val_dice_cof > best_iou:
            best_iou = val_dice_cof
            torch.save(model.state_dict(), 'model_MSC_ShuffleNet_42.pth')
            print("=> saved best model")
            trigger = 0

        # early stopping
        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("=> early stopping")
            break

        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
