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
TRAIN_SP="$REPO_PATH/scripts/train_sp.py"
ENCODE="$REPO_PATH/scripts/encode.py"

#TEST_EN_MONO="$REPO_PATH/corpus/monolingual/test.en"
TEST_JA_MONO="$REPO_PATH/corpus/monolingual/test.ja"

# 学習済みのSentencePieceを用いて各データセットをエンコードする
encode () {
    python $ENCODE --model bpe.model
}

#encode < $TEST_EN_MONO >  $REPO_PATH/corpus/monolingual/test_mono_00.en
encode < $TEST_JA_MONO > $REPO_PATH/corpus/monolingual/test_mono_00.ja
