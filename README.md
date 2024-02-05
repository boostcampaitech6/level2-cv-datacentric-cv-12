# OCR 글자 검출 프로젝트

![image](https://github.com/boostcampaitech6/level2-cv-datacentric-cv-12/assets/149575938/fc5f11af-dacd-4e73-b271-018a2993dfd2)


- 2024.01.24 ~ 2024.02.01
- 네이버 커넥트 재단 및 Upstage에서 주관하는 비공개 대회


## Members

> 공통 : EDA, Annotation 가이드 제작, 리라벨링, 모델 학습
>
>[김세진](https://github.com/Revabo): 학습데이터 피클화를 통한 모델 학습시간 경량화
>
>[박혜나](https://github.com/heynapark): 추론 결과 분석, Noise Data Augmentation
>
>[이동우](https://github.com/Dong-Uri): Valid set, DetEval 연구 및 코드 제작
>
>[진민주](https://github.com/freenozero): CVAT 세팅, 외부 데이터셋 학습 진행
>
>[허재영](https://github.com/jae-heo): Pepper noise Augmentation, Noise Reduction
 


## 문제 정의(대회소개) & Project Overview

![image](https://github.com/boostcampaitech6/level2-cv-datacentric-cv-12/assets/149575938/b7614573-b6dc-4fea-9ea8-521de882d6cd)




스마트폰으로 카드를 결제하거나, 카메라로 카드를 인식할 경우 자동으로 카드 번호가 입력되는 경우가 있습니다. 또 주차장에 들어가면 차량 번호가 자동으로 인식되는 경우도 흔히 있습니다. 이처럼 OCR (Optimal Character Recognition) 기술은 사람이 직접 쓰거나 이미지 속에 있는 문자를 얻은 다음 이를 컴퓨터가 인식할 수 있도록 하는 기술로, 컴퓨터 비전 분야에서 현재 널리 쓰이는 대표적인 기술 중 하나입니다. 

이번 프로젝트에서는 OCR을 이용해 의료 영수증의 글자 영역을 Detecting하는 것이 목표입니다.



## 대회 결과

Public 5등 | Private 2등

![image](https://github.com/boostcampaitech6/level2-cv-datacentric-cv-12/assets/149575938/9b802e1f-ccdd-4d20-a6c2-fe11042e1866)



![image](https://github.com/boostcampaitech6/level2-cv-datacentric-cv-12/assets/149575938/7f106459-0eff-450f-960b-688215bb0995)



## Dataset

- 전체 이미지 개수 : Train set 100장, Test set 100장
- 이미지 종류 : 진료비 영수증
- 이미지 크기 : Various

## Metric

- F1 score

![image](https://github.com/boostcampaitech6/level2-cv-datacentric-cv-12/assets/149575938/6226821f-fcef-49b6-a33e-79a6a7726ba2)



## Model

- EAST


## Tools
- Github
- Notion
- Slack
- Wandb

## Project Outline
![image](https://github.com/boostcampaitech6/level2-cv-datacentric-cv-12/assets/149575938/5043cf0d-ffa2-4991-91fa-9278dd116268)



## Data Augmentations

- CIE Ich Noise
- Median Blur
- 명도 확산
- Pepper Noise
