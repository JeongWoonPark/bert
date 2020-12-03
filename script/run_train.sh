INPUT_PATH="storage/tfrecord/corpus_wiki_news/len512/train"
OUTPUT_PATH="pre_training_model/corpus_wiki_news_lr14"

mpirun -np 8 \
    -H localhost:8 \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python src/run_pretraining_hvd.py \
      --input_file=${INPUT_PATH} \
      --output_dir=${OUTPUT_PATH} \
      --do_train=True \
      --do_eval=False \
      --bert_config_file=bert_config.json \
      --max_seq_length=512 \
      --train_batch_size=4 \
      --max_predictions_per_seq=20 \
      --num_train_steps=1000000 \
      --num_warmup_steps=10000 \
      --learning_rate=2e-5 \
      --save_checkpoints_steps=10000 \
      --init_checkpoint=${OUTPUT_PATH}/model.ckpt-930000
