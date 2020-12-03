MODEL="BERT_Char_PVo"
MODEL_PATH="storage/model/${MODEL}"
DATA_PATH="fine_tuning_model/nsmc"
OUTPUT_DIR="${DATA_PATH}/${MODEL}"

#mpirun -np 2 \
#    -H localhost:2 \
#    -bind-to none -map-by slot \
#    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
#    -mca pml ob1 -mca btl ^openib \
    python src/run_classifier_hvd.py \
      --task_name=nsmc \
      --do_train=True \
      --do_eval=True \
      --data_dir=${DATA_PATH} \
      --vocab_file=${MODEL_PATH}/vocab.txt \
      --bert_config_file=${MODEL_PATH}/bert_config.json \
      --init_checkpoint=${MODEL_PATH}/model.ckpt-1000000 \
      --max_seq_length=128 \
      --train_batch_size=30 \
      --learning_rate=2e-5 \
      --num_train_epochs=3.0 \
      --output_dir=${OUTPUT_DIR}
