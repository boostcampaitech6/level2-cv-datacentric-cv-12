import json
import torch
from torch import cuda
from torch.optim import lr_scheduler

# 이하는 code 내의 경로로 맞춰주셔야합니다.
from model import EAST
from inference import do_inference
from deteval import calc_deteval_metrics


def take_deteval(trained_pth): # 4개의 json에 대해 deteval의 calc_deteval_metrics로 gt bbox를 계산하는 함수
    data_dir = '/data/ephemeral/home/data/medical'
    valid_20_json = 'valid_images'
    valid_40_json = 'val_instances_default_ufo'
    train_json = 'train'
    train_noise_json = 'train_noise'
    ignore_tags = ['masked', 'excluded-region', 'maintable', 'stamp']
    # trained_pth = '/data/ephemeral/home/code/trained_models/train_latest.pth'

    # JSON 파일 경로
    file_path_1 = data_dir + '/ufo/' + valid_20_json + '.json'
    file_path_2 = data_dir + '/ufo/' + valid_40_json + '.json'
    file_path_3 = data_dir + '/ufo/' + train_json + '.json'
    file_path_4 = data_dir + '/ufo/' + train_noise_json + '.json'

    def gt_bbox(json_file_path):
        # JSON 파일 읽어오기
        with open(json_file_path, 'r') as json_file:
            valid_data = json.load(json_file)

        # gt bbox 계산
        gt_bboxes_dict = {}
        for file_name in valid_data['images']:
            gt_bboxes_dict[file_name] = []
            for word_id in valid_data['images'][file_name]['words']:
                for tag in valid_data['images'][file_name]['words'][word_id]['tags']:
                    if tag in ignore_tags:
                        break
                else:
                    gt_bboxes_dict[file_name].append(valid_data['images'][file_name]['words'][word_id]['points'])

        return gt_bboxes_dict

    gt_bboxes_dict_1 = gt_bbox(file_path_1)
    gt_bboxes_dict_2 = gt_bbox(file_path_2)
    gt_bboxes_dict_3 = gt_bbox(file_path_3)
    gt_bboxes_dict_4 = gt_bbox(file_path_4)

    # # 딕셔너리를 파일로 저장
    # with open(f'gt_bboxes_dict.json', 'w', encoding='utf-8') as json_file:
    #     json.dump(gt_bboxes_dict, json_file, ensure_ascii=False, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    model = EAST(pretrained=False).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    ufo_result = do_inference(model, trained_pth, data_dir, input_size=2048, batch_size=5, split='train_noise')

    # # 딕셔너리를 파일로 저장
    # with open(f'ufo_result.json', 'w', encoding='utf-8') as json_file:
    #     json.dump(ufo_result, json_file, ensure_ascii=False, indent=2)

    # # JSON 파일 읽어오기
    # with open('/data/ephemeral/home/code/ufo_result.json', 'r') as json_file:
    #     ufo_result = json.load(json_file)

    def pred_bbox(gt_bboxes_dict): # gt bbox의 이미지들에 대해 pred bbox를 유추하는 함수
        pred_bboxes_dict = {}
        for file_name in ufo_result['images']:
            if file_name in gt_bboxes_dict:
                pred_bboxes_dict[file_name] = []
                for word_id in ufo_result['images'][file_name]['words']:
                    pred_bboxes_dict[file_name].append(ufo_result['images'][file_name]['words'][word_id]['points'])

        return pred_bboxes_dict

    pred_bboxes_dict_1 = pred_bbox(gt_bboxes_dict_1)
    pred_bboxes_dict_2 = pred_bbox(gt_bboxes_dict_2)
    pred_bboxes_dict_3 = pred_bbox(gt_bboxes_dict_3)
    pred_bboxes_dict_4 = pred_bbox(gt_bboxes_dict_4)

    # # 딕셔너리를 파일로 저장
    # with open(f'pred_bboxes_dict.json', 'w', encoding='utf-8') as json_file:
    #     json.dump(pred_bboxes_dict, json_file, ensure_ascii=False, indent=2)

    print('pth', trained_pth)
    print('val20', calc_deteval_metrics(pred_bboxes_dict_1, gt_bboxes_dict_1)['total'])
    print('val40', calc_deteval_metrics(pred_bboxes_dict_2, gt_bboxes_dict_2)['total'])
    print('train', calc_deteval_metrics(pred_bboxes_dict_3, gt_bboxes_dict_3)['total'])
    print('noise', calc_deteval_metrics(pred_bboxes_dict_4, gt_bboxes_dict_4)['total'])

# 원하는 pth들로 수행
take_deteval('/data/ephemeral/home/code/trained_models/train2_latest_109.pth')
take_deteval('/data/ephemeral/home/code/trained_models/train2_latest.pth')
take_deteval('/data/ephemeral/home/code/trained_models/train3_latest.pth')
take_deteval('/data/ephemeral/home/code/trained_models/test3_latest.pth')
take_deteval('/data/ephemeral/home/code/trained_models/test2_latest.pth')
take_deteval('/data/ephemeral/home/code/trained_models/test2_best.pth')
take_deteval('/data/ephemeral/home/code/trained_models/test_latest.pth')
take_deteval('/data/ephemeral/home/code/trained_models/test_best.pth')
