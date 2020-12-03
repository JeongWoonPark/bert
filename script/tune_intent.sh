#!/bin/bash

DATA_PATH="../../data/finetuning_data/PVoT/intent"
MODEL_PATH="../../../storage/model/BERT_SP_Whole_100m_lr25"
OUTPUT_PATH="../../BERT_finetuningmodels/pvot_intent/v${1}_2e_25lr"

#mpirun -np 2 \
#    -H localhost:2 \
#    -bind-to none -map-by slot \
#    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
#    -mca pml ob1 -mca btl ^openib \
    CUDA_VISIBLE_DEVICES=0 python src/run_classifier.py \
      --task_name=pvot \
      --do_train=True \
      --do_eval=True \
      --do_predict=True \
      --do_lower_case=False \
      --data_dir=${DATA_PATH} \
      --vocab_file=${MODEL_PATH}/vocab.txt \
      --bert_config_file=${MODEL_PATH}/bert_config.json \
      --init_checkpoint=${MODEL_PATH}/model.ckpt-1000000 \
      --max_seq_length=128 \
      --train_batch_size=32 \
      --learning_rate=2e-5 \
      --num_train_epochs=1 \
      --output_dir=${OUTPUT_PATH}
