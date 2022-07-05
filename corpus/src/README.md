pip3 install datasets

pip3 install sacremoses

pip3 install spacy

データセット作成

python3 create_dataset.py --repo_path PATH_TO_REPOSITORY


MeCab　インストール　使い方

pip install mecab-python3

pip install unidic

python -m unidic download

https://atmarkit.itmedia.co.jp/ait/articles/2102/05/news027.html

python3 create_dataset.py --repo_path /home/hiroshi/Machine_Translation_Proto/ --tatoeba --len_filter --overlap_filter --ratio_filter --freq_filter --threading --num_threads 4
