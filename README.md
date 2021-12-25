# vit_hdp
HDP experiments on xcit (and other vision transformer networks to come)

> xcit_small_12_p16_224_hdp5to8q
```
python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --model xcit_small_12_p16 \
    --data-path /host/ubuntu/data/imagenet2012 \
    --batch-size 128 --drop-path 0.05 --output_dir ./experiments/xcit_small_12_p16/ --epochs 300 \
    --hdp q --hdp_num_heads 5 \


```