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
TRAIN_EN="$REPO_PATH/corpus/genuine_bilingual/train.en"
TRAIN_JA="$REPO_PATH/corpus/genuine_bilingual/train.ja"
VALID_EN="$REPO_PATH/corpus/genuine_bilingual/valid.en"
VALID_JA="$REPO_PATH/corpus/genuine_bilingual/valid.ja"
TEST_EN="$REPO_PATH/corpus/genuine_bilingual/test.en"
TEST_JA="$REPO_PATH/corpus/genuine_bilingual/test.ja"
TRAIN_SP="$REPO_PATH/scripts/train_sp.py"
ENCODE="$REPO_PATH/scripts/encode.py"

TEST_EN_MONO="$REPO_PATH/corpus/monolingual/test.en"
TEST_JA_MONO="$REPO_PATH/corpus/monolingual/test.ja"

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

encode < $TEST_EN_MONO > test_mono.en
encode < $TEST_JA_MONO > test_mono.ja

# fairseqの前処理用コマンドを実行する
fairseq-preprocess -s en -t ja \
    --trainpref train \
    --validpref valid \
    --destdir data-bin \
    --joined-dictionary \
    --workers 4
