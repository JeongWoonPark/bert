#!/usr/bin/env bash

DATA_PATH="../data/finetuning_data/PVoT/multi-intent"
MODEL_PATH="../storage/model/BERT_SP_Whole_100m_lr25"
OUTPUT_PATH="finetuning_models/pvot_multi-intent/v${1}_1e_25lr"

  CUDA_VISIBLE_DEVICES=1 python src/BERT_NER.py\
    --task_name=ner \
    --do_lower_case=False \
    --crf=True \
    --lstm=False \
    --do_train=True \
    --do_eval=True \
    --do_predict=True \
    --data_dir=${DATA_PATH} \
    --vocab_file=${MODEL_PATH}/vocab.txt \
    --bert_config_file=${MODEL_PATH}/bert_config.json \
    --init_checkpoint=${MODEL_PATH}/model.ckpt-1000000 \
    --max_seq_length=128 \
    --train_batch_size=32 \
    --learning_rate=2e-5 \
    --num_train_epochs=1 \
    --output_dir=${OUTPUT_PATH}


perl src/conlleval.pl -d '\t' < ${OUTPUT_PATH}/label_test.txt
