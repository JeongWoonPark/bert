#!/bin/sh

VOCAB_PATH="rsc/vocab/corpus_wiki_news_7m"
INPUT_PATH="rsc/corpus/shard"
OUTPUT_PATH="storage/tfrecord/corpus_wiki_news/len512"

begin=94
end=94

while [ ${begin} -le ${end} ]; do

  python src/create_pretraining_data.py \
    --vocab_file=${VOCAB_PATH}/vocab.txt \
    --input_file=${INPUT_PATH}/corpus_wiki_news_${begin}.txt \
    --output_file=${OUTPUT_PATH}/${begin}.tfrecord \
    --do_lower_case=False \
    --do_whole_word_mask=True \
    --max_seq_length=512 \
    --max_predictions_per_seq=20 \
    --masked_lm_prob=0.15 \
    --random_seed=12345 \
    --dupe_factor=5

  (( begin += 1 ))
done

