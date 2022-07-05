import random as rd
import tqdm as t
import typing
import numpy as np
import os


def replace_all(text, pattern: typing.Dict[str, str]):
    """
    patternで指定された置き換えをまとめて実行する関数
    """
    for old, new in pattern.items():
        text = text.replace(old, new)
    return text


def check_ratio(split_ratio: typing.Dict[str, float]):
    ratio = split_ratio.items()
    vals = [r[1] for r in ratio]
    sum = np.sum(vals)
    eps = 0.01

    return False if len(ratio) != 3 or not all([val > 0.0 for val in vals]) or not (1.0 - eps < sum < 1.0 + eps) else True


def write_ds(f_name, f_path, bitexts, div_size):
    """
    作成したデータセットをファイルに書き込む関数
    複数ファイルへの分割書き込みに対応 (大きなデータセットの場合に有効)
    """
    total = len(bitexts)
    num_split = 1 if div_size >= total else int(total / div_size)
    size = total if div_size >= total else div_size
    num_split = num_split if num_split * size == total else num_split+1

    for idx in range(num_split):
        en_path = os.path.join(f_path, "{}{}.en".format(f_name, idx+1))
        ja_path = os.path.join(f_path, "{}{}.ja".format(f_name, idx+1))
        with open(en_path, 'w') as f_en, open(ja_path, 'w') as f_ja:
            print("\nWriting {}{}.en and {}{}.ja ...".format(
                f_name, idx+1, f_name, idx+1))
            head = idx * size
            tail = total if idx == (num_split-1) else (idx+1) * size
            for bitext in t.tqdm(bitexts[head:tail]):
                en, ja = bitext.split('\t')
                f_en.write(en + '\n')
                f_ja.write(ja + '\n')
            print("Finished writing {}{}.en and {}{}.ja   ({} sents)".format(
                f_name, idx+1, f_name, idx+1, tail-head))


def split_dataset(en_sents, ja_sents, split_ratio: typing.Dict[str, float], repo_path, div_size=1000000, div_train=False, div_valid=False, div_test=False):
    data_path = os.path.join(repo_path, "corpus/data/")
    pattern = {'\t': '', '\n': ''}
    en_ja = [replace_all(en, pattern) + '\t' + replace_all(ja, pattern)
             for en, ja in zip(en_sents, ja_sents)]
    rd.shuffle(en_ja)

    total = len(en_ja)
    if check_ratio:
        train_size = int(split_ratio["train"] * total)
        valid_size = int(split_ratio["valid"] * total)
        test_size = total - (train_size + valid_size)
        train = en_ja[:train_size]
        valid = en_ja[train_size:train_size + valid_size]
        test = en_ja[train_size + valid_size:]

        # 各データセットを分割する場合は、分割後のサイズを指定する。
        # 分割後の各ファイルのサイズが、分割前のサイズ(例 len(valid) や len(test)など)
        # を上回る場合は分割前のサイズに合わせて保存される
        _size = div_size if div_train else train_size
        write_ds('train', data_path, train, _size)
        _size = div_size if div_valid else valid_size
        write_ds('valid', data_path, valid, _size)
        _size = div_size if div_test else test_size
        write_ds('test', data_path, test, _size)
