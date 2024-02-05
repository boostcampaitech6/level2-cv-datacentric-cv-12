import os
import os.path as osp
import time
import math
from datetime import timedelta
from argparse import ArgumentParser

import torch
from torch import cuda
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm
from random import shuffle
import pickle

from east_dataset import EASTDataset
from dataset import SceneTextDataset
from model import EAST
import torchvision.transforms as transform
from PIL import Image

import wandb
        
def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--data_dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', '../data/medical'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR',
                                                                        'trained_models'))

    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--image_size', type=int, default=2048)
    parser.add_argument('--input_size', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--max_epoch', type=int, default=150)
    parser.add_argument('--save_interval', type=int, default=5)
    parser.add_argument('--ignore_tags', type=list, default=['masked', 'excluded-region', 'maintable', 'stamp'])

    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args


def do_training(data_dir, model_dir, device, image_size, input_size, num_workers, batch_size,
                learning_rate, max_epoch, save_interval, ignore_tags):
    pickle_filepath = 'east_dataset'
    east = []

    dataset = SceneTextDataset(
            data_dir,
            split='train',
            image_size=image_size,
            crop_size=input_size,
            ignore_tags=ignore_tags
        )
    dataset = EASTDataset(dataset)
    num_batches = math.ceil(len(dataset) / batch_size)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    if os.path.exists(f'{pickle_filepath}.pkl'):
        with open(f'{pickle_filepath}.pkl', 'rb') as pickle_file:
            east = pickle.load(pickle_file)
    else:
        for b in train_loader:
            east.append(b)

        # 데이터를 피클 파일로 저장
        with open(pickle_filepath, 'wb') as pickle_file:
            pickle.dump(east, pickle_file)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST()
    model.load_state_dict(torch.load('/data/ephemeral/home/code/code/trained_models/MJ.pth'))
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max= 10,eta_min=learning_rate*0.1)

    '''wandb.init(
        # set the wandb project where this run will be logged
        project="data_centric",
        entity='cv-12',
        name = 'SJ'
    )'''

    best_loss = 1000
    model.train()
    for epoch in range(max_epoch):
        epoch_loss, epoch_start = 0, time.time()
        with open(f'{pickle_filepath}{epoch%20}.pkl', 'rb') as pickle_file:
            east = pickle.load(pickle_file)
        shuffle(east)
        with tqdm(total=num_batches) as pbar:
            for img, gt_score_map, gt_geo_map, roi_mask in east:
                #image = transform.ToPILImage()(img[0])
                #image.save(f'{epoch}.jpg')
                pbar.set_description('[Epoch {}]'.format(epoch + 1))
                loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_val = loss.item()
                epoch_loss += loss_val

                pbar.update(1)
                val_dict = {
                    'Cls loss': extra_info['cls_loss'], 'Angle loss': extra_info['angle_loss'],
                    'IoU loss': extra_info['iou_loss']
                }
                #wandb.log({'Cls loss': extra_info['cls_loss'], 'Angle loss': extra_info['angle_loss'], 'IoU loss': extra_info['iou_loss']})
                pbar.set_postfix(val_dict)

        scheduler.step()
        if best_loss > epoch_loss / num_batches:
            best_loss = epoch_loss / num_batches
            if not osp.exists(model_dir):
                os.makedirs(model_dir)
            ckpt_fpath = osp.join(model_dir, 'best.pth')
            torch.save(model.state_dict(), ckpt_fpath)
            print(f'Best loss : {best_loss}')

        print('Mean loss: {:.4f} | Elapsed time: {}'.format(
            epoch_loss / num_batches, timedelta(seconds=time.time() - epoch_start)))

        if (epoch + 1) % save_interval == 0:
            if not osp.exists(model_dir):
                os.makedirs(model_dir)

            ckpt_fpath = osp.join(model_dir, 'latest.pth')
            torch.save(model.state_dict(), ckpt_fpath)


def main(args):
    do_training(**args.__dict__)


if __name__ == '__main__':
    args = parse_args()
    main(args)
