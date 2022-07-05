from tqdm import tqdm
import os


def dl_WikiMatrix(repo_path):
    num_sents = 3895992
    print("\nDownloading WikiMatrix dataset...")
    os.system(
        "wget --progress=bar https://dl.fbaipublicfiles.com/laser/WikiMatrix/v1/WikiMatrix.en-ja.tsv.gz")
    os.system("gunzip WikiMatrix.en-ja.tsv.gz")

    en_ls, ja_ls = [], []
    with open("WikiMatrix.en-ja.tsv", 'r') as f:
        for line in tqdm(f, total=num_sents):
            _, en, ja = line.rstrip().split('\t')
            en_ls.append(en + '\n')
            ja_ls.append(ja + '\n')
    os.system("rm WikiMatrix.en-ja.tsv")
    return en_ls, ja_ls


# テストコード
if __name__ == "__main__":
    en_ls, ja_ls = dl_WikiMatrix("/home/hiroshi/Machine_Translation_Proto/")
    print("The number of sentences in WikiMatrix dataset:   {} sents".format(
        min(len(en_ls), len(ja_ls))))
    for en, ja in zip(en_ls[2000:2010], ja_ls[2000:2010]):
        print("en: %s" % en)
        print("ja: %s \n" % ja)
