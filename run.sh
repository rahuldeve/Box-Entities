python main.py \
--model saved_models/m4.pt \
--data_dir ../data/zeshel-full \
--logfile_path logs/m4.txt \
--pretrained_model bert-base-uncased \
--B 16 \
--gradient_accumulation_steps 4 \
--logging_steps 100 \
--k 64 \
--epochs 4 \
--lr 0.00002 \
--type_cands hard_and_random_negative \
--gpus 0,1,2,3,4,5,6,7 \
--warmup_proportion 0.2 \
--num_cands 64 \
--cands_ratio 0.5 \
--vol_temp 1 \
--vol_int_temp 0.00001 \
--int_temp 0.00001
