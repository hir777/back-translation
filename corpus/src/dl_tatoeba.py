import datasets 
import os
import tqdm as t
import json


def dl_tatoeba(repo_path):
    ds_path = os.path.join(repo_path, "corpus/data")
    print(ds_path)
    ds_dict = datasets.load_dataset("tatoeba", lang1="en", lang2="ja")
    ds = ds_dict["train"]
    ds.info.write_to_directory(ds_path)
    ds.to_json(os.path.join(ds_path, "en-ja.json"))


def reform_json(file1, file2):
    with open(file1, "r", encoding="utf-8") as fr, open(file2, "w", encoding="utf-8") as fw:
        num_sents = int(os.popen("wc -l %s" % file1).read().strip().split()[0])
        fw.write('{\n  "bitexts-en-ja": [\n')
        print("\nReformating a json file downloaded from tatoeba...")
        for i in t.tqdm(range(0, num_sents)):
            line = fr.readline()
            if i != num_sents-1:
                fw.write("  " + line[:-1] + ",\n")
            else:
                fw.write("  " + line[:-1] + "\n")
        fw.write(' ]\n}')


def json2list(repo_path):
    ds_path = os.path.join(repo_path, "corpus/data/en-ja.json")
    tmp_path = os.path.join(repo_path, "corpus/data/dataset.json")
    en_ls, ja_ls = [], []

    # jsonファイルをフォーマットし直す
    reform_json(ds_path, tmp_path)
    with open(tmp_path, 'r', encoding='utf-8') as fr, open(ds_path, "w", encoding="utf-8") as fw:
        data = json.load(fr)
        json.dump(data, fw, indent=2, ensure_ascii=False)

        print("\nConverting a json file into list...")
        for bitext in t.tqdm(data["bitexts-en-ja"]):
            bitext = bitext["translation"]
            en_ls.append(bitext["en"])
            ja_ls.append(bitext["ja"])

    os.system("rm {}".format(tmp_path))
    return en_ls, ja_ls


if __name__ == "__main__":
    dl_tatoeba("/home/hiroshi/tmp/Machine_Translation_Proto/")
    en_ls, ja_ls = json2list("/home/hiroshi/tmp/Machine_Translation_Proto/")

    for en, ja in zip(en_ls[:10], ja_ls[:10]):
        print(en + '\t' + ja)
