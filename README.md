# BERT
- tensorflow==1.15

## 사용법
### 1. 전처리
`script/create_shard_tfrecord.sh` 혹은 `script/create_tfrecord.sh`
### 2. 사전학습
`script/run_train.sh`
### 3. 파인튜닝
#### 1) 사전학습모델 성능 검증용 Benchmark 데이터 (NSMC, KorQuAD)
`script/tune_nsmc.sh`, `script/tune_korquad.sh`
#### 2) 의도 분석
`script/tune_intent.sh`
#### 3) 개체명 인식
`script/tune_ner.sh`
### 4. 배포 (Docker 방식)
`script/run_docker.sh`

## TODO
- [ ] requirements.txt 작성
- [ ] 전처리 코드 검수
- [ ] 사전학습 코드 검수
- [x] 파인튜닝 코드 검수
- [ ] GPU 병렬처리 코드 (horovod) 검수
