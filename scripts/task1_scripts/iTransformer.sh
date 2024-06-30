export CUDA_VISIBLE_DEVICES=0

#cd ..

product_list=('cu' 'zn' 'ni' 'au' 'ag')

for product in "${product_list[@]}";
do

python -u run.py \
  --is_training_and_testing [1,1] \
  --model 'iTransformer' \
  --product $product \
  --train_set ['202203', '202205'] \
  --vali_set ['202205', '202206'] \
  --test_set ['202206', '202207'] \
  --test_day_list ['20220601', '20220602'] \
  --data 'task1' \
  --target 'weight_price' \
  --itr 3 \
  --tran_epochs 10

done