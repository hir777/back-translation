import re
import sacremoses as sm
import unicodedata
import MeCab
from typing import List
import multiprocessing as mp
import os
import gc


class Tokenization():
    def __init__(self, workers=1):
        self.workers = mp.Value('i', workers)

    def tokenize_en(self, en_sents: List[str]):
        mt = sm.MosesTokenizer(lang='en')
        for en in en_sents:
            en = unicodedata.normalize("NFKC", en)
            en = re.sub(
                mt.AGGRESSIVE_HYPHEN_SPLIT[0], r'\1 - ', en)
            en = mt.tokenize(en, escape=False)
            en = ' '.join(en).lower()
            yield en

    def tokenize_ja(self, ja_sents: List[str]):
        mecab = MeCab.Tagger("-Owakati")
        for ja in ja_sents:
            ja = unicodedata.normalize("NFKC", ja)
            ja = mecab.parse(ja)
            yield ja

    def tokenize_en_ja(self, queue, en_sents: List[str], ja_sents: List[str]):
        print("Tokenization (Process ID: {}) started.".format(
            os.getpid()))
        queue.put([en.replace('\t', '') + '\t' + ja.replace('\t', '')
                   for en, ja in zip(self.tokenize_en(en_sents), self.tokenize_ja(ja_sents))])
        print("Tokenization [Process ID: {}] has finished.".format(
            os.getpid()))

    def tokenize(self, en_sents: List[str], ja_sents: List[str]):
        """
        引数で与えられた英文と和文のリストをトークン化する関数
        処理の高速化のためにマルチプロセス処理を実装してある
        """
        queue = mp.Queue()
        num_sents = min(len(en_sents), len(ja_sents))
        min_workers = 1
        max_workers = 12

        self.workers.value = self.workers.value if min_workers <= self.workers.value and self.workers.value <= max_workers else min_workers
        self.workers.value = self.workers.value if self.workers.value <= num_sents else min_workers

        size = int(num_sents / self.workers.value)
        for idx in range(self.workers.value):
            head = idx * size
            tail = (idx+1) * size if idx != (self.workers.value-1) else num_sents
            proc = mp.Process(target=self.tokenize_en_ja, args=[queue,
                                                                en_sents[head: tail], ja_sents[head: tail]])
            proc.start()

        tmp = []
        for _ in range(self.workers.value):
            tmp.append(queue.get())
        tmp_gen = (sents for sents in tmp)
        bitexts = [sent for sents in tmp_gen for sent in sents]
        del tmp[:]
        gc.collect()
        en_ls, ja_ls = [], []
        for bitext in bitexts:
            en, ja = bitext.split('\t')
            en_ls.append(en.strip())
            ja_ls.append(ja.strip())

        return en_ls, ja_ls


# テスト用コード
if __name__ == "__main__":
    en_ls = ["I have to sleep.",
             "Michael is twenty years old today.",
             "The password is Muriel.",
             "I will come back soon."]
    ja_ls = ["私は眠らなければなりません。",
             "マイケルは今日２０歳になりました。",
             "パスワードは「Muiriel」です。",
             "まもなく私は戻って来ます。"
             ]

    tkn = Tokenization(workers=2)
    en_sents, ja_sents = tkn.tokenize(en_ls, ja_ls)
    print(en_sents)
    print(ja_sents)
