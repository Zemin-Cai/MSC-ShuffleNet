import argparse
import os
from glob import glob
import torch.nn as nn
import torch.backends.cudnn as cudnn
import yaml
import MSC_ShuffleNet
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from dataset import Dataset
from PIL import Image
import numpy as np
import torch


from scipy.spatial.distance import directed_hausdorff
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='MSC_ShuffleNet',
                        help='model name')

    args = parser.parse_args()

    return args

def _one_hot_encoder(input_tensor):
    tensor_list = []
    for i in range(5):
        temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
        tensor_list.append(temp_prob.unsqueeze(1))
    output_tensor = torch.cat(tensor_list, dim=1)
    return output_tensor.float()


def main():
    args = parse_args()

    with open('models/%s/config.yml' % args.name, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-'*20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-'*20)

    cudnn.benchmark = True

    print("=> creating model %s" % config['arch'])
    model = MSC_ShuffleNet.__dict__[config['arch']](5,
                                      config['deep_supervision'])


    # Data loading code
    img_ids = glob(os.path.join('inputs', config['dataset'], 'images', '*' + config['img_ext']))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    _, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=42)

    model = nn.DataParallel(model, device_ids=[0, 1, 2])
    model = model.cuda()

    model_state_dict = torch.load('models/model_MSC_ShuffleNet_42.pth')
    model.load_state_dict(model_state_dict)
    model.eval()
    print('parameters:', sum([p.numel() for p in model.parameters() if p.requires_grad]))

    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join('inputs', config['dataset'], 'images'),
        mask_dir=os.path.join('inputs', config['dataset'], 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=None)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    for c in range(config['num_classes']):
        os.makedirs(os.path.join('outputs', 'MSC-ShuffleNet_42', str(c)), exist_ok=True)
    with torch.no_grad():
        for input, target, meta in tqdm(val_loader, total=len(val_loader)):
            input = input.cuda()
            model = model.cuda()
            # compute output
            output = model(input)
            output = output.cpu().numpy()
            # iou_avg_meter.update(iou, input.size(0))
            # dice_avg_meter.update(dice_coeff, input.size(0))
            output=np.argmax(output, axis=1)
            torch.cuda.empty_cache()
            for i in range(output.shape[0]):
                Image.fromarray((output[i, :, :] * 255 / 4).astype(np.uint8)).save(
                    os.path.join('outputs', 'MSC-ShuffleNet_42', meta['img_id'][i] + '.png'))
if __name__ == '__main__':
    main()
