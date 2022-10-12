from audioop import mul
import dl_tatoeba as tatoeba
import filter as fl
import split_dataset as spl
import argparse
import dl_WikiMatrix as wiki
import tokenize_enja as tkn
import sys
import time
from cleaning import clean
import gc


def print_bitexts(en_sents, ja_sents):
    for idx, (en, ja) in enumerate(zip(en_sents, ja_sents)):
        print("\nEN{}:   {}".format(idx+1, en))
        print("JA{}:   {}".format(idx+1, ja))


def check_workers(workers, name, min, max):
    if workers < min or workers > max:
        print("The value workers_{} {} is invalid. ".format(name, workers))
        print("It is replaced by %d." % min)
        return min
    else:
        return workers


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='usage')
    parser.add_argument("--repo_path", type=str,
                        help="absolute path of Machine_Translation_Proto repository")
    parser.add_argument("--cleaning", action="store_true",
                        help="turn on/off the cleaning feature.")
    parser.add_argument("--tatoeba", action="store_true",
                        help="use Tatoeba dataset")
    parser.add_argument("--WikiMatrix", action="store_true",
                        help="use WikiMatrix dataset.")
    parser.add_argument("--len_filter", action="store_true",
                        help="turn on/off the length filter")
    parser.add_argument("--min_len", type=int, default=4,
                        help="valid minimum length of sentences in a dataset")
    parser.add_argument("--max_len", type=int, default=256,
                        help="valid maximum length of sentences in a dataset")
    parser.add_argument("--overlap_filter", action="store_true",
                        help="turn on/off the length filter")
    parser.add_argument("--ratio_filter", action="store_true",
                        help="turn on/off the ratio filter")
    parser.add_argument("--freq_filter", action="store_true",
                        help="turn on/off the freq filter")
    parser.add_argument("--freq_thld", type=int, default=3,
                        help="threshold for filtering words by frequency")
    parser.add_argument("--workers_tkn", type=int, default=1,
                        help="the number of processes to accelerate tokenization\nDefault: 1   Valid range: 1 <= workers_tkn <= 12")
    parser.add_argument("--workers_freq", type=int, default=1,
                        help="the number of processes to accelerate creating frequency dictionaries\nDefault: 1   Valid range: 1 <= workers_freq <= 8")
    parser.add_argument("--workers_clean", type=int, default=1,
                        help="the number of processes to accelerate cleaning downloaded datasets\nDefault: 1   Valid range: 1 <= workers_clean <= 8")
    parser.add_argument("--div_size", type=int, default=250000,
                        help="the number of sentences contained in each divided file if division of a dataset is enabled")
    parser.add_argument("--div_train", action="store_true",
                        help="divide a train dataset into several pieces when this optional parameter is given.")
    parser.add_argument("--div_valid", action="store_true",
                        help="divide a valid dataset into several pieces when this optional parameter is given.")
    parser.add_argument("--div_test", action="store_true",
                        help="divide a test dataset into several pieces when this optional parameter is given.")

    args = parser.parse_args()
    repo_path = args.repo_path

    en_tmp_ls, ja_tmp_ls = [], []

    # Tatoebaデータセットをダウンロードしてリスト化する
    if args.tatoeba:
        tatoeba.dl_tatoeba(repo_path)
        tatoeba_en, tatoeba_ja = tatoeba.json2list(repo_path)
        en_tmp_ls.append(tatoeba_en)
        ja_tmp_ls.append(tatoeba_ja)

    # WikiMatrixデータセットをダウンロードしてリスト化する
    if args.WikiMatrix:
        wiki_en, wiki_ja = wiki.dl_WikiMatrix(repo_path)

        # 後で各データセットを結合する時のために小分けにしてリストに保存しておく。
        # それによって、結合時のメモリの使用率を下げることができる。
        total = min(len(wiki_en), len(wiki_ja))
        _size = 10000
        num_split = 390
        for idx in range(num_split):
            head = idx * _size
            tail = (idx+1) * _size if idx != (num_split-1) else total
            en_tmp_ls.append(wiki_en[head:tail])
            ja_tmp_ls.append(wiki_ja[head:tail])

    if len(en_tmp_ls) == 0 or len(ja_tmp_ls) == 0:
        print("You need to specify at least one dataset to create a new dataset.")
        sys.exit()
    else:
        # 各データセットを一つのリストにまとめて保存する
        en_tmp_gen = (en_sents for en_sents in en_tmp_ls)
        ja_tmp_gen = (ja_sents for ja_sents in ja_tmp_ls)
        en_ls = [en_sent for en_sents in en_tmp_gen for en_sent in en_sents]
        ja_ls = [ja_sent for ja_sents in ja_tmp_gen for ja_sent in ja_sents]

        del en_tmp_ls[:]
        del ja_tmp_ls[:]
        gc.collect()

    if args.cleaning:
        workers_clean = args.workers_clean
        min_workers_clean = 1
        max_workers_clean = 20
        workers_clean = check_workers(
            workers_clean, "clean", min_workers_clean, max_workers_clean)

        start = time.time()
        en_ls, ja_ls = clean(en_ls, ja_ls, workers_clean)
        end = time.time()
        print("%d seconds for cleaning datasets" % int(end - start))

    print("\n{} sentences".format(min(len(en_ls), len(ja_ls))))

    # 英文と日本文をそれぞれトークン化する
    workers_tkn = args.workers_tkn
    min_workers_tkn = 1
    max_workers_tkn = 20
    workers_tkn = check_workers(
        workers_tkn, "tkn", min_workers_tkn, max_workers_tkn)

    print("\nTokenizing sentences...")
    start = time.time()
    tkn = tkn.Tokenization(workers=workers_tkn)
    en_ls, ja_ls = tkn.tokenize(en_ls, ja_ls)
    end = time.time()
    print("%d seconds for tokenizing sentences" % int(end - start))

    # フィルタリング
    if args.len_filter:
        min = args.min_len
        max = args.max_len
        if min < 1 or min > 16:
            print(
                "The minimum length of sentences should be in a range: 1 <= min_len <= 16")
            print("Specified min_len %d is replaced by %d" % (min, 5))
            min = 5
        if max < 16 or max > 256:
            print(
                "The maximum length of sentences should be in a range: 16 <= min_len <= 256")
            print("Specified max_len %d is replaced by %d" % (max, 32))
            max = 32
        en_ls, ja_ls = fl.len_filter(en_ls, ja_ls, min, max, truncate=True)

    if args.overlap_filter:
        en_ls, ja_ls = fl.overlap_filter(en_ls, ja_ls)

    if args.ratio_filter:
        en_ls, ja_ls = fl.ratio_filter(en_ls, ja_ls)

    if args.freq_filter:
        workers_freq = args.workers_freq
        min_workers_freq = 1
        max_workers_freq = 20
        workers_freq = check_workers(
            workers_freq, "freq", min_workers_freq, max_workers_freq)

        en_ls, ja_ls = fl.freq_filter(
            en_ls, ja_ls, args.freq_thld, workers=workers_freq)

    split_ratio = {"train": 0.98, "valid": 0.01, "test": 0.01}
    spl.split_dataset(en_ls, ja_ls, split_ratio, repo_path,
                      div_size=args.div_size, div_train=args.div_train, div_valid=args.div_valid, div_test=args.div_test)
