# Fix for 2 cls
python eval.py \
--model saved_models/m3.pt \
--data_dir ../data/zeshel-full \
--B 64 \
--k 64 \
--gpus 0,1,2,3 \
--eval_method micro \
--vol_temp 0.5 \
--vol_int_temp 0.00001 \
--int_temp 0.00001
