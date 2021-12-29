
> 3090 TEST: xcit_small_12_p16_224_h8 | non-hdp
```
python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --model xcit_small_12_p16 \
    --data-path /host/ubuntu/data/imagenet2012 \
    --batch-size 128 --drop-path 0.05 --output_dir ./experiments/xcit_small_12_p16/ --epochs 400 \
    --gpu 1x3090 --wandb 1 \


```