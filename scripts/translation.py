import re
import unicodedata
import sentencepiece as spm
from fairseq.models.transformer import TransformerModel
from sacremoses import MosesTokenizer
import MeCab
import unidic


class Translation:
    def __init__(self, src: str, tgt: str):
        self.src = src
        self.tgt = tgt
        self.ln_ls = ["en", "ja"]
        if src == "en":
            self.tokenizer = MosesTokenizer(lang="en")
        elif src == "ja":
            self.tokenizer = MeCab.Tagger("-Owakati")
        else:
            raise ValueError(
                "Error: Source language %s is not supported." % src)

        if tgt not in self.ln_ls:
            raise ValueError(
                "Error: Target language %s is not supported." % tgt)

    def load(self, checkpoint_dir, checkpoint_file, data_name_or_path, path_bpe_model):
        self.model = TransformerModel.from_pretrained(
            checkpoint_dir,
            checkpoint_file=checkpoint_file,
            data_name_or_path=data_name_or_path
        )
        self.sp = spm.SentencePieceProcessor(model_file=path_bpe_model)

    def preproc_en(self, en):
        en = unicodedata.normalize("NFKC", en)
        en = re.sub(self.tokenizer.AGGRESSIVE_HYPHEN_SPLIT[0], r'\1 - ', en)
        en = self.tokenizer.tokenize(en, escape=False)
        en = ' '.join(en).lower()
        en = ' '.join(self.sp.encode(en, out_type="str"))
        return en

    def preproc_ja(self, ja):
        ja = unicodedata.normalize("NFKC", ja)
        ja = self.tokenizer.parse(ja)
        ja = ' '.join(self.sp.encode(ja, out_type="str"))
        return ja

    def translate(self, src_sent, beam, lenpen):
        src_sent = self.preproc_en(
            src_sent) if self.src == "en" else self.preproc_ja(src_sent)
        tgt_sent = self.model.translate(src_sent, beam, lenpen)
        tgt_sent = ''.join(tgt_sent.split()).replace(' ', '').replace('_', '').strip()
        return tgt_sent
