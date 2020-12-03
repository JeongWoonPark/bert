#!/bin/bash

#INTENT_PATH="$(pwd)/finetuning_models/pvot_intent/v${1}_2e_25lr/export_model"
#NER_PATH="$(pwd)/finetuning_models/pvot_ner/v${1}_1e_25lr/export_model"
MULTI_PATH="$(pwd)/finetuning_models/pvot_multi-intent/v1.1_1e_25lr/export_model"

#docker run -t --rm -p 6501:8501 \
#  -v "${INTENT_PATH}:/models/PVoT_CLS" \
#  -e MODEL_NAME=PVoT_CLS \
#  --name "pvot_cls" \
#  tensorflow/serving &

#docker run -t --rm -p 7501:8501 \
#  -v "${NER_PATH}:/models/PVoT_NER" \
#  -e MODEL_NAME=PVoT_NER \
#  --name "pvot_ner" \
#  tensorflow/serving &

docker run -t --rm -p 8501:8501 \
  -v "${MULTI_PATH}:/models/PVoT_Multi" \
  -e MODEL_NAME=PVoT_Multi \
  --name "pvot_multi" \
  tensorflow/serving &
