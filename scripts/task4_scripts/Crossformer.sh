export CUDA_VISIBLE_DEVICES=0

#cd ..

product_list=('cu' 'zn' 'ni' 'au' 'ag')

for product in "${product_list[@]}";
do

python -u run.py \
  --is_training_and_testing [1,1] \
  --model 'Crossformer' \
  --product $product \
  --train_set ['20220301', '20220505'] \
  --vali_set ['20220505', '20220601'] \
  --test_set ['20220601', '20220701'] \
  --test_day_list ['20220601', '20220602'] \
  --data 'task4' \
  --target 'ret' \
  --itr 3 \
  --tran_epochs 10

done