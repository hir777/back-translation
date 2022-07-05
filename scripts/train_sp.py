import sentencepiece as spm
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help=""" one-sentence-per-line raw corpus file. 
                            No need to run tokenizer, normalizer or preprocessor. """)
    parser.add_argument("--prefix", type=str,
                        help="output model name prefix. <model_name>.model and <model_name>.vocab are generated.")
    parser.add_argument("--vocab_size", type=int, default=8000,
                        help="vocabulary size, e.g., 8000, 16000, or 32000")
    parser.add_argument("--character_coverage", type=float, default=0.9995,
                        help="""amount of characters covered by the model, good defaults are: 0.9995 for languages with rich character set like Japanese or Chinese. 
                            1.0 for other languages with small character set.""")
    parser.add_argument("--model_type", type=str, default="unigram",
                        help="Choose from unigram (default), bpe, char, or word. The input sentence must be pretokenized when using word type.")
    args = parser.parse_args()

    spm.SentencePieceTrainer.Train(
        input=args.input,
        model_prefix=args.prefix,
        vocab_size=args.vocab_size,
        character_coverage=args.character_coverage,
        model_type=args.model_type)
