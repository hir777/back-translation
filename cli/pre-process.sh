#!/bin/bash
set -ex

# レポジトリの絶対パスをコマンドライン引数REPO_APTHとして渡す
for ARGUMENT in "$@"
do
    KEY=$(echo $ARGUMENT | cut -f1 -d=)

    KEY_LENGTH=${#KEY}
    VALUE="${ARGUMENT:$KEY_LENGTH+1}"

    export "$KEY"="$VALUE"
done

# 学習用・検証用・テスト用データのPATHを指定
TRAIN_EN="$REPO_PATH/corpus/data/train.en"
TRAIN_JA="$REPO_PATH/corpus/data/train.ja"
VALID_EN="$REPO_PATH/corpus/data/valid.en"
VALID_JA="$REPO_PATH/corpus/data/valid.ja"
TEST_EN="$REPO_PATH/corpus/data/test.en"
TEST_JA="$REPO_PATH/corpus/data/test.ja"
TRAIN_SP="$REPO_PATH/src/train_sp.py"
ENCODE="$REPO_PATH/src/encode.py"

# 学習用データセットを用いてSentencePieceを学習させる
cat $TRAIN_EN $TRAIN_JA > train.enja
python $TRAIN_SP --input train.enja --prefix bpe --vocab_size 8000 --character_coverage 0.9995

# 学習済みのSentencePieceを用いて各データセットをエンコードする
encode () {
    python $ENCODE --model bpe.model
}

encode < $TRAIN_EN > train.en
encode < $TRAIN_JA > train.ja
encode < $VALID_EN > valid.en
encode < $VALID_JA > valid.ja
encode < $TEST_EN > test.en
encode < $TEST_JA > test.ja

# fairseqの前処理用コマンドを実行する
fairseq-preprocess -s en -t ja \
    --trainpref train \
    --validpref valid \
    --destdir data-bin \
    --joined-dictionary \
    --workers 4
