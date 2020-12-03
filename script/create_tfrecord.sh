##### Customize #####
CORPUS="news_all"
VOCAB_SIZE="70000"
#####################

INPUT_PATH="../corpus"
OUTPUT_PATH="../tfrecord/${CORPUS}"
VOCAB_PATH="../vocab/${CORPUS}"

python create_pretraining_data.py \
  --input_file=${INPUT_PATH}/${CORPUS}.txt \
  --output_file=${OUTPUT_PAT}/${CORPUS}_whole.tfrecord \
  --vocab_file=${VOCAB_PATH}/vocab_${VOCAB_SIZE}.txt \
  --do_lower_case=False \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=5 \
  --do_whole_word_mask=True
