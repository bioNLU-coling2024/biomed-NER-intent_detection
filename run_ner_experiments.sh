#!/bin/bash
#MODEL="cnn lstm logreg xgb biobert pubmedbert biomedroberta bert roberta"
MODEL="cnn"
DATASET="ncbi jnlpba bc5cdr anatem DDI"
DATASET_PATH="./NER_dataset/"
OUTPUT_PATH="./outputs/"

for DATA in $DATASET
do
    echo "STARTING " $DATA
    TRAIN_FILE=$DATASET_PATH/train_$DATA.txt
    TEST_FILE=$DATASET_PATH/test_$DATA.txt

    if [$MODEL == "cnn"] || [$MODEL == "lstm"]; then
        python model_cnn_lstm_ner/main.py --feature_extractor MODEL --train_path $TRAIN_FILE --test_path $TEST_FILE > ${OUTPUT_PATH}/${DATA}.${MODEL}.txt
    elif [$MODEL == "logreg"] || [$MODEL == "xgb"];; then
        python model_handcrafted_features_ner/main.py --base_path $OUTPUT_PATH  --train_path $TRAIN_FILE --test_path $TEST_FILE --dataset $DATA > ${OUTPUT_PATH}/${DATA}.${MODEL}.txt
    else
        python model_transformer_ner/main.py --model $MODEL --dataset $DATA --base_dir $OUTPUT_PATH > ${OUTPUT_PATH}/${DATA}.${MODEL}.txt
    fi
done