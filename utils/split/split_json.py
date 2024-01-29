import json
from sklearn.model_selection import train_test_split

# split할 JSON 파일 경로
json_file_path = 'data/medical/ufo/train.json'

# JSON 파일 불러오기
with open(json_file_path, 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

# 이미지 정보를 훈련 세트와 검증 세트로 나누는 함수
def split_train_valid(data, test_size=0.2, random_state=42):
    image_dict = data.get("images", {})

    # 이미지 이름과 이미지 정보를 각각 리스트로 추출
    image_names, image_infos = zip(*image_dict.items())

    # train과 valid로 이미지를 나눔
    train_names, valid_names, train_infos, valid_infos = train_test_split(
        image_names, image_infos, test_size=test_size, random_state=random_state
    )

    # 나뉜 이미지들을 딕셔너리로 만듦
    train_images = {name: info for name, info in zip(train_names, train_infos)}
    valid_images = {name: info for name, info in zip(valid_names, valid_infos)}

    return train_images, valid_images

# 이미지 정보를 훈련 세트와 검증 세트로 나눔
train_images, valid_images = split_train_valid(data)

# # 결과 확인
# print(f"Train Set: {list(train_images.keys())}")
# print(f"Valid Set: {list(valid_images.keys())}")

# 나뉜 이미지 정보를 각각의 JSON 파일로 저장
with open('train_images.json', 'w', encoding='utf-8') as json_file:
    json.dump({"images": train_images}, json_file, ensure_ascii=False, indent=2)

with open('valid_images.json', 'w', encoding='utf-8') as json_file:
    json.dump({"images": valid_images}, json_file, ensure_ascii=False, indent=2)