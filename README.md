# vit_hdp swin
HDP experiments on swin


## Installation
> python/conda env:
```
conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=10.1 -c pytorch

pip install timm==0.3.2

pip install opencv-python==4.4.0.46 termcolor==1.1.0 yacs==0.1.8
```

> [Optional] Install Apex:
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```


## Experiments
> A: swin_tiny_patch4_window7_224 | non-hdp baseline
```
python -m torch.distributed.launch --nproc_per_node=4 --use_env  --master_port 12345 \
    main.py --local_rank 0 \
    --data-path /host/ubuntu/data/imagenet2012 \
    --cfg configs/swin_tiny_patch4_window7_224.yaml \


```

> B: swin_tiny_patch4_window7_224_hdp2qk | hdp HALF heads for qk
```
python -m torch.distributed.launch --nproc_per_node=4 --use_env  --master_port 12345 \
    main.py --local_rank 0 \
    --data-path /host/ubuntu/data/imagenet2012 \
    --cfg configs/swin_tiny_patch4_window7_224_hdp2qk_nonlinear.yaml \


```


> 3090 TEST: swin_tiny_patch4_window7_224 | non-hdp
add `--amp-opt-level O0` to disable mixed-precision training
```
python -m torch.distributed.launch --nproc_per_node=1 --use_env  --master_port 12345 \
    main.py --local_rank 0 \
    --data-path /host/ubuntu/data/imagenet2012 \
    --cfg configs/swin_tiny_patch4_window7_224_3090.yaml \
    --batch-size 128 --amp-opt-level O0 \
    --wandb 1 \


python -m torch.distributed.launch --nproc_per_node=1 --use_env  --master_port 12345 \
    main.py --local_rank 0 \
    --data-path /host/ubuntu/data/imagenet2012 \
    --cfg configs/swin_tiny_patch4_window7_224_hdp2qk_nonlinear.yaml \
    --batch-size 128 --amp-opt-level O0 \
    --wandb 1 \


```
