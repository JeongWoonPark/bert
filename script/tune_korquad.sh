MODEL="BERT_Korean_YeonTaek"
MODEL_PATH="storage/model/${MODEL}"
DATA_PATH="fine_tuning_model/korquad"
OUTPUT_PATH="${DATA_PATH}/${MODEL}"

#mpirun -np 2 \
#    -H localhost:2 \
#    -bind-to none -map-by slot \
#    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
#    -mca pml ob1 -mca btl ^openib \
    python src/run_squad_hvd.py \
      --do_train=True \
      --train_file=${DATA_PATH}/KorQuAD_v1.0_train.json \
      --do_predict=True \
      --predict_file=${DATA_PATH}/KorQuAD_v1.0_dev.json \
      --vocab_file=${MODEL_PATH}/vocab.txt \
      --bert_config_file=${MODEL_PATH}/bert_config.json \
      --init_checkpoint=${MODEL_PATH}/bert_model.ckpt \
      --max_seq_length=128 \
      --train_batch_size=28 \
      --learning_rate=5e-5 \
      --num_train_epochs=3.0 \
      --output_dir=${OUTPUT_PATH}
