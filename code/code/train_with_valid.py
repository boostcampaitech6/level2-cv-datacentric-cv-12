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

from east_dataset import EASTDataset
from dataset import SceneTextDataset
from model import EAST

import logging

import cv2
from detect import detect
import json
from deteval import calc_deteval_metrics

import wandb


# 로그 설정
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
log_file = osp.join(log_dir, 'train_log.txt')
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# wnadb 설정
wandb.init(
    project="data_centric",
    entity='cv-12',
    name = 'wandb_name'
)

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
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--max_epoch', type=int, default=150)
    parser.add_argument('--save_interval', type=int, default=5)
    parser.add_argument('--ignore_tags', type=list, default=['masked', 'excluded-region', 'maintable', 'stamp'])

    parser.add_argument('--pretrained', type=str, default=None) # pretrained 모델 경로 입력
    parser.add_argument('--valid_interval', type=int, default=5) # 몇 epoch마다 valid할 것인지
    parser.add_argument('--valid_start', type=int, default=0) # 몇 epoch부터 valid할 것인지

    parser.add_argument('--train_json', type=str, default='train_instances_default_ufo')
    parser.add_argument('--valid_json', type=str, default='val_instances_default_ufo')
    parser.add_argument('--train_folder', type=str, default='train')
    parser.add_argument('--valid_folder', type=str, default='train') # valid 이미지도 train 폴더 내에 있는 게 defalut

    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args


def do_training(data_dir, model_dir, device, image_size, input_size, num_workers, batch_size,
                learning_rate, max_epoch, save_interval, ignore_tags,
                pretrained, valid_interval, valid_start,
                train_json, valid_json, train_folder, valid_folder):

    dataset = SceneTextDataset(
        data_dir,
        split=train_folder,
        image_size=image_size,
        crop_size=input_size,
        ignore_tags=ignore_tags,
        json_name = train_json
    )
    dataset = EASTDataset(dataset)
    num_batches = math.ceil(len(dataset) / batch_size)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    # valid set은 직접 작성한 json을 통해 가져옴

    # JSON 파일 경로
    json_file_path = data_dir + '/ufo/' + valid_json + '.json'

    # JSON 파일 읽어오기
    with open(json_file_path, 'r') as json_file:
        valid_data = json.load(json_file)

    # gt bbox 계산
    gt_bboxes_dict = {}
    for file_name in valid_data['images']:
        gt_bboxes_dict[file_name] = []
        for word_id in valid_data['images'][file_name]['words']:
            gt_bboxes_dict[file_name].append(valid_data['images'][file_name]['words'][word_id]['points'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EAST()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[max_epoch // 2], gamma=0.1)

    # 저장된 모델 불러오기
    if pretrained:
        # checkpoint = torch.load('/data/ephemeral/home/code/trained_models/test_latest.pth')
        checkpoint = torch.load(pretrained)
        model.load_state_dict(checkpoint)

    max_f1 = [0, 0] # 최대 f1 score와 그 epoch을 저장해두는 값

    for epoch in range(max_epoch):
        print('epoch -', epoch + 1)
        model.train()
        epoch_loss, epoch_start = 0, time.time()
        # with tqdm(total=num_batches) as pbar:
        #     for img, gt_score_map, gt_geo_map, roi_mask in train_loader:
        #         pbar.set_description('[Epoch {}]'.format(epoch + 1))

        #         loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
        #         optimizer.zero_grad()
        #         loss.backward()
        #         optimizer.step()

        #         loss_val = loss.item()
        #         epoch_loss += loss_val

        #         pbar.update(1)
        #         val_dict = {
        #             'Cls loss': extra_info['cls_loss'], 'Angle loss': extra_info['angle_loss'],
        #             'IoU loss': extra_info['iou_loss']
        #         }
        #         pbar.set_postfix(val_dict)

        # tqdm pbar가 보기 불편해서 변경하였고 그외 코드는 동일합니다.
        i = 0
        for img, gt_score_map, gt_geo_map, roi_mask in train_loader:
            i += 1
            loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            epoch_loss += loss_val

            val_dict = {
                'Cls loss': extra_info['cls_loss'], 'Angle loss': extra_info['angle_loss'],
                'IoU loss': extra_info['iou_loss']
            }

            logging.info(f'[Epoch {epoch + 1}] no.{i} - {val_dict}')

            wandb.log({'Epoch': epoch + 1,'Cls loss': extra_info['cls_loss'], 'Angle loss': extra_info['angle_loss'], 'IoU loss': extra_info['iou_loss']})

        scheduler.step()

        logging.info('Mean loss: {:.4f} | Elapsed time: {}'.format(
            epoch_loss / num_batches, timedelta(seconds=time.time() - epoch_start)))

        wandb.log({'Epoch': epoch + 1,'Mean loss': epoch_loss / num_batches})
        
        if (epoch + 1) % save_interval == 0:
            if not osp.exists(model_dir):
                os.makedirs(model_dir)

            ckpt_fpath = osp.join(model_dir, 'latest.pth')
            torch.save(model.state_dict(), ckpt_fpath)

        model.eval()
        valid_time = time.time()

        if (epoch + 1) % valid_interval == 0 and epoch + 1 >= valid_start:

            image_fnames, by_sample_bboxes = [], []

            images = []

            for file_name in valid_data['images']:
                image_fpath = data_dir + '/img/' + valid_folder + '/' + file_name
                image_fnames.append(osp.basename(image_fpath))

                images.append(cv2.imread(image_fpath)[:, :, ::-1])
                if len(images) == batch_size:
                    by_sample_bboxes.extend(detect(model, images, input_size))
                    images = []

            if len(images):
                by_sample_bboxes.extend(detect(model, images, input_size))

            ufo_result = dict(images=dict())
            for image_fname, bboxes in zip(image_fnames, by_sample_bboxes):
                words_info = {idx: dict(points=bbox.tolist()) for idx, bbox in enumerate(bboxes)}
                ufo_result['images'][image_fname] = dict(words=words_info)

            pred_bboxes_dict = {}
            for file_name in ufo_result['images']:
                pred_bboxes_dict[file_name] = []
                for word_id in ufo_result['images'][file_name]['words']:
                    pred_bboxes_dict[file_name].append(ufo_result['images'][file_name]['words'][word_id]['points'])

            result = calc_deteval_metrics(pred_bboxes_dict, gt_bboxes_dict)

            logging.info('-' * 30 + f' VALID in {epoch + 1} epoch ' + '-' * 30)
            logging.info('%s | Elapsed time: %s', result['total'], timedelta(seconds=time.time() - valid_time))
            logging.info('-' * 80)

            wandb.log({'Epoch': epoch + 1,'precision': result['total']['precision'], 'recall': result['total']['recall'], 'hmean': result['total']['hmean']})

            # f1 score (hmean)이 최대일때 best 저장
            if result['total']['hmean'] > max_f1[0]:
                if not osp.exists(model_dir):
                    os.makedirs(model_dir)

                ckpt_fpath = osp.join(model_dir, 'best.pth')
                torch.save(model.state_dict(), ckpt_fpath)

                max_f1 = [result['total']['hmean'], epoch + 1]

                logging.info(f'{epoch + 1} epoch model is best')

def main(args):
    do_training(**args.__dict__)


if __name__ == '__main__':
    args = parse_args()
    main(args)
