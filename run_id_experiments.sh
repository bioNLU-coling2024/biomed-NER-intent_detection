#!/bin/bash
#MODELS="roberta pubmedbert bert"
MODEL="roberta"
DATASET="mergedNER"
DATASET_PATH="./ID_dataset/"
OUTPUT_PATH="./outputs/"

for DATA in $DATASET
do
    echo "STARTING " $DATA
    TRAIN_FILE=$DATASET_PATH/train_$DATA.csv
    TEST_FILE=$DATASET_PATH/test_$DATA.csv
    python model_transformer_id/main.py --model $MODEL --dataset $DATA --base_dir $OUTPUT_PATH > ${OUTPUT_PATH}/${DATA}.${MODEL}.txt
done