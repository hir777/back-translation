from inspect import cleandoc
import unicodedata
from pkg_resources import yield_lines
from tqdm import tqdm
import re
import regex
import multiprocessing as mp
import os


def is_en(sents):
    en = re.compile("""[a-zA-Z   # アルファベット
                        0-9      # アラビア数字
                        \u2160-\u2188   # ローマ数字
                        \u0020-\u002F\u003A-\u0040\u005B-\u0060\u007B-\u007E   # ASCII記号の半角版
    ]+""")

    en_tf_ls = []
    for sent in sents:
        if en.fullmatch(sent) is None:
            en_tf_ls.append(False)
        else:
            en_tf_ls.append(True)
    return en_tf_ls


def is_ja(sents):
    ja = regex.compile("""[\u3041-\u309F                    # ひらがな
                            \u30A1-\u30FF\uFF66-\uFF9F      # カタカナ
                            0-9０-９                        # アラビア数字
                            \p{Numeric_Type=Numeric}        # 漢数字、ローマ数字
                            \p{Script_Extensions=Han}       # 漢字
                            # ASCII文字(記号)の半角版
                            \u0020-\u002F\u003A-\u0040\u005B-\u0060\u007B-\u007E
                            # ASCII文字(記号)全角版と日本語の記号の半角版
                            \uFF01-\uFF0F\uFF1A-\uFF20\uFF3B-\uFF40\uFF5B-\uFF65\u3000-\u303F
    ]+""")

    ja_tf_ls = []
    for sent in sents:
        if ja.fullmatch(sent) is None:
            ja_tf_ls.append(False)
        else:
            ja_tf_ls.append(True)
    return ja_tf_ls


def rm_noise(en_sents, ja_sents, en_q, ja_q):
    """
    正規表現を用いてデータセットに含まれるノイズ(記号, URL, メールアドレス, etc...)を除去する関数
    高速化のため、正規表現のパターンを事前にコンパイルしておく。

    同じ機能を実現するための正規表現のパターンは一通りではなく、いくつも考えられる。
    しかし、パターンによってはプログラムを意図せず停止させてしまうことがあるから、
    新しい機能をこの関数に追加するときは、十分にテストする。
    """
    brackets = re.compile(r"""\<.*?\>|\{.*?\}|\(.*?\)|\[.*?\]|   # 括弧（半角）
                            |【.*?】|（.*?）|〈.*?〉|《.*?》|「.*?」|『.*?』|【.*?】|                # 括弧（全角）
                            |〔.*?〕|〖.*?〗|〘.*?〙|〚.*?〛|｛.*?｝|＜.*?＞|｛.*?｝|｟.*?｠|＜.*?＞  # 括弧（全角）
                            """)
    unwanted = re.compile(
        r"[*#^\「\」\『\』\〈\〉:;\<\>\{\}\"\(\)\[\]]+")   # 間違って 空白を入れてしまわないように注意する
    msc = re.compile(r"\\\\|\t|\\\\t|\r|\\\\r")
    newlines = re.compile(r"\\\n|\n")
    urls = re.compile(
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+")
    email = re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    )
    encoding_err = re.compile("0000,0000,0000,\w*?")
    multi_space = re.compile("[ 　]{2,}")
    emoji = regex.compile("\p{Emoji_Presentation=Yes}+")
    hiragana_rare = re.compile(
        "[\U0001B001-\U0001B11F\U0001B150-\U0001B152\U0001F200]+")
    katakana_rare = re.compile(
        "[\u31F0-\u31FF\u32D0-\u32FE\u3300-\u3357\U0001AFF0-\U0001AFFE\U0001B000\U0001B120-\U0001B122\U0001B164-\U0001B167]+")

    cleaned_en, cleaned_ja = [], []
    print("Start denoising sentences... (Process ID: {})".format(os.getpid()))

    for en_sent, ja_sent in zip(en_sents, ja_sents):
        en_sent = unicodedata.normalize("NFKC", en_sent).strip()
        ja_sent = unicodedata.normalize("NFKC", ja_sent).strip()
        en_sent = urls.sub('', en_sent)
        ja_sent = urls.sub('', ja_sent)

        en_sent = email.sub('', en_sent)
        ja_sent = email.sub('', ja_sent)

        en_sent = msc.sub(' ', en_sent)
        ja_sent = msc.sub(' ', ja_sent)

        en_sent = newlines.sub('', en_sent)
        ja_sent = newlines.sub('', ja_sent)

        en_sent = emoji.sub('', en_sent)
        ja_sent = emoji.sub('', ja_sent)

        en_sent = brackets.sub('', en_sent)
        ja_sent = brackets.sub('', ja_sent)

        en_sent = unwanted.sub('', en_sent)
        ja_sent = unwanted.sub('', ja_sent)

        ja_sent = hiragana_rare.sub('', ja_sent)
        ja_sent = katakana_rare.sub('', ja_sent)

        en_sent = multi_space.sub(' ', en_sent)
        ja_sent = multi_space.sub(' ', ja_sent)

        en_sent = encoding_err.sub('', en_sent)
        ja_sent = encoding_err.sub('', ja_sent)

        cleaned_en.append(en_sent.strip())
        cleaned_ja.append(ja_sent.strip())

    en_q.put(cleaned_en)
    ja_q.put(cleaned_ja)

    print("Finished denoising sentences... (Process ID: {})".format(os.getpid()))


def clean(en_sents, ja_sents, workers=1):
    """
    正規表現を用いてデータセットに含まれる各種のノイズ(記号, URL, メールアドレス, etc...)を除去するジェネレータ関数
    また、日英以外の言語の文も発見次第除去する。
    マルチプロセス対応済み(引数 workers を用いてプロセス数を指定する)
    """
    min_workers = 1
    max_workers = 8
    num_sents = min(len(en_sents), len(ja_sents))
    workers = 1 if workers < min_workers or workers > max_workers or workers > num_sents else workers
    size = int(num_sents/workers)

    en_q = mp.Queue()
    ja_q = mp.Queue()

    tgt_fun = rm_noise
    for idx in range(workers):
        head = idx * size
        tail = (idx+1) * size if idx != (workers-1) else num_sents
        proc = mp.Process(target=tgt_fun, args=[
                          en_sents[head:tail], ja_sents[head:tail], en_q, ja_q])
        proc.start()

    cleaned_en, cleaned_ja = [], []
    for _ in range(workers):
        for en_sent, ja_sent in zip(en_q.get(), ja_q.get()):
            cleaned_en.append(en_sent)
            cleaned_ja.append(ja_sent)

    print("\nChecking if downloaded sentences are truly English or Japanese sentences...")

    en_ls, ja_ls = [], []
    for idx, (en_tf, ja_tf) in enumerate(zip(is_en(cleaned_en), is_ja(cleaned_ja))):
        if en_tf and ja_tf:
            en_ls.append(cleaned_en[idx])
            ja_ls.append(cleaned_ja[idx])

    return en_ls, ja_ls


# テスト用コード
if __name__ == "__main__":
    # clean関数全体のテストコード
    en_sents = [
        "the college 🤩provided courses in science , engineering and art , and also established its own internal degree courses in science and engineering , which were ratified by the university of london .",
        "he won the \t fossati prize . \n https://en.wikipedia.org/wiki/Giampiero_Fossati https://www.w3resource.com/python-exercises/re/python-re-exercise-42.php",
        "he was born in haarlem as the son of aart jansz and became a lawyer .",
        "He said (私は日系アメリカ人二世です。).",
        "I    have a 💯(pen). timo.33@gmail.com"
    ]

    ja_sents = [
        "当時 の 😀😃学校 は 科学 、 工学 、 🤩芸術 の コース を 開講 し 、 それ ら の コース は ロンドン 大学 より 学位 の 承認 を 受け て い た 。",
        "^ ピタリ賞 を 獲得 し た 。http://en.wikipedia.org/wiki/Giampiero_Fossati",
        "er wurde in haarlem als sohn von aart jansz geboren und wurde rechtsanwalt.",
        "彼は「日系アメリカ人二世です。０IV」と言った。",
        "私はペン　　を持っています。*💯"
    ]

    # is_en関数とis_ja関数のテストコード
    #en_tf_ls, ja_tf_ls = [], []
    # for en_tf, ja_tf in zip(is_en(en_sents), is_ja(ja_sents)):
    #  en_tf_ls.append(en_tf)
    #  ja_tf_ls.append(ja_tf)
    # print(en_tf_ls)
    # print(ja_tf_ls)

    # clean関数全体のテストコード

    en_ls, ja_ls = clean(en_sents=en_sents, ja_sents=ja_sents, workers=4)
    print(en_ls)
    print(ja_ls)

    # rm_noise関数で用いられている正規表現のテストコード
    # en = "^(Hello Nice to meet# you.) ljsadfjl:kashttps://www.w3resource.com/python-exercises/re/python-re-exercise-42.php"
    # ja = "や***あ**＜こんに 𛀁ちは＞^#『みんな』ポケモン<>080- ㋐ 	㍈ 4482-1811;𛀂  𛀃 https://en.wikipedia.org/wiki/Giampiero_Fossati "
    # hiragana_rare = re.compile(
    #    "[\U0001B001-\U0001B11F\U0001B150-\U0001B152\U0001F200]+")
    # katakana_rare = re.compile(
    #    "[\u31F0-\u31FF\u32D0-\u32FE\u3300-\u3357\U0001AFF0-\U0001AFFE\U0001B000\U0001B120-\U0001B122\U0001B164-\U0001B167]+")
    #urls = re.compile("http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+")
    #cleaned_en = hiragana_rare.sub('', en)
    #cleaned_ja = hiragana_rare.sub('', ja)
    #cleaned_en = katakana_rare.sub('', en)
    #cleaned_ja = katakana_rare.sub('', ja)
    #cleaned_en = urls.sub('', en)
    #cleaned_ja = urls.sub('', ja)
    #print("BF: {}\nAF: {}".format(en, cleaned_en))
    #print("BF: {}\nAF: {}".format(ja, cleaned_ja))
