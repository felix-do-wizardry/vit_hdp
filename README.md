# vit_hdp
HDP experiments on xcit (and other vision transformer networks to come)

> xcit_small_12_p16_224_h8 | non-hdp 8 heads
```
python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --model xcit_small_12_p16 \
    --data-path /host/ubuntu/data/imagenet2012 \
    --batch-size 64 --drop-path 0.1 --output_dir ./experiments/xcit_small_12_p16/ --epochs 400 \
    --hdp_num_heads 4 --wandb 1 \
```

> xcit_small_12_p16_224_hdp4to8q | hdp 4->8 heads for q
```
python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --model xcit_small_12_p16 \
    --data-path /host/ubuntu/data/imagenet2012 \
    --batch-size 64 --drop-path 0.1 --output_dir ./experiments/xcit_small_12_p16/ --epochs 400 \
    --hdp q --hdp_num_heads 4 --wandb 1 \
```

> xcit_small_12_p16_224_hdp4to8k | hdp 4->8 heads for k
```
python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --model xcit_small_12_p16 \
    --data-path /host/ubuntu/data/imagenet2012 \
    --batch-size 64 --drop-path 0.1 --output_dir ./experiments/xcit_small_12_p16/ --epochs 400 \
    --hdp k --hdp_num_heads 4 --wandb 1 \
```

> xcit_small_12_p16_224_hdp4to8qk | hdp 4->8 heads for qk
```
python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --model xcit_small_12_p16 \
    --data-path /host/ubuntu/data/imagenet2012 \
    --batch-size 64 --drop-path 0.1 --output_dir ./experiments/xcit_small_12_p16/ --epochs 400 \
    --hdp qk --hdp_num_heads 4 --wandb 1 \
```





> 3090 TEST: xcit_small_12_p16_224_h8 | non-hdp
```
python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --model xcit_small_12_p16 \
    --data-path /host/ubuntu/data/imagenet2012 \
    --batch-size 128 --drop-path 0.05 --output_dir ./experiments/xcit_small_12_p16/ --epochs 400 \
    --gpu 1x3090 --wandb 1 \



```
