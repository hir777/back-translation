fairseq-interactive \
    /home/tasaki/Machine_Translation_Proto/data-bin \
    --buffer-size 1024 \
    --batch-size 128 \
    --path /home/tasaki/Machine_Translation_Proto/checkpoints/checkpoint_last.pt \
    --beam 3 \
    --lenpen 0.6 \
    < test.en \
    | grep '^H' \
    | cut -f 3 \
    | python /home/tasaki/Machine_Translation_Proto/src/decode.py \
    | tee output.txt \
    | sacrebleu /home/tasaki/Machine_Translation_Proto/corpus/data/test.ja

head /home/tasaki/Machine_Translation_Proto/output.txt /home/tasaki/Machine_Translation_Proto/corpus/data/test.ja
