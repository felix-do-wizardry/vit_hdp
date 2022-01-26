# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

from .swin_transformer import SwinTransformer, SwinTransformer_HDP
from .swin_mlp import SwinMLP

# default settings
# _C.MODEL.SWIN = CN()
# _C.MODEL.SWIN.PATCH_SIZE = 4
# _C.MODEL.SWIN.IN_CHANS = 3
# _C.MODEL.SWIN.EMBED_DIM = 96
# _C.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
# _C.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
# _C.MODEL.SWIN.WINDOW_SIZE = 7
# _C.MODEL.SWIN.MLP_RATIO = 4.
# _C.MODEL.SWIN.QKV_BIAS = True
# _C.MODEL.SWIN.QK_SCALE = None
# _C.MODEL.SWIN.APE = False
# _C.MODEL.SWIN.PATCH_NORM = True

def build_model(config):
    model_type = config.MODEL.TYPE
    if model_type == 'swin':
        hdp = config.MODEL.HDP.HDP
        if hdp:
            assert isinstance(hdp, str)
            assert hdp in ['q', 'k', 'qk'], hdp
            model = SwinTransformer_HDP(img_size=config.DATA.IMG_SIZE,
                                    patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                    in_chans=config.MODEL.SWIN.IN_CHANS,
                                    num_classes=config.MODEL.NUM_CLASSES,
                                    embed_dim=config.MODEL.SWIN.EMBED_DIM,
                                    depths=config.MODEL.SWIN.DEPTHS,
                                    num_heads=config.MODEL.SWIN.NUM_HEADS,
                                    window_size=config.MODEL.SWIN.WINDOW_SIZE,
                                    mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                                    qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                                    qk_scale=config.MODEL.SWIN.QK_SCALE,
                                    drop_rate=config.MODEL.DROP_RATE,
                                    drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                    ape=config.MODEL.SWIN.APE,
                                    patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                    use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                                    
                                    hdp=config.MODEL.HDP.HDP,
                                    hdp_ratios=config.MODEL.HDP.HDP_RATIOS,
                                    hdp_non_linear=config.MODEL.HDP.NON_LINEAR,
                                    hdp_stages=config.MODEL.HDP.STAGES,
                                    
                                    )
        else:
            model = SwinTransformer(img_size=config.DATA.IMG_SIZE,
                                    patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                    in_chans=config.MODEL.SWIN.IN_CHANS,
                                    num_classes=config.MODEL.NUM_CLASSES,
                                    embed_dim=config.MODEL.SWIN.EMBED_DIM,
                                    depths=config.MODEL.SWIN.DEPTHS,
                                    num_heads=config.MODEL.SWIN.NUM_HEADS,
                                    window_size=config.MODEL.SWIN.WINDOW_SIZE,
                                    mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                                    qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                                    qk_scale=config.MODEL.SWIN.QK_SCALE,
                                    drop_rate=config.MODEL.DROP_RATE,
                                    drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                    ape=config.MODEL.SWIN.APE,
                                    patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                    use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                                    )
    elif model_type == 'swin_mlp':
        raise ValueError('`swin_mlp` is not supported for HDP experiments')
        model = SwinMLP(img_size=config.DATA.IMG_SIZE,
                        patch_size=config.MODEL.SWIN_MLP.PATCH_SIZE,
                        in_chans=config.MODEL.SWIN_MLP.IN_CHANS,
                        num_classes=config.MODEL.NUM_CLASSES,
                        embed_dim=config.MODEL.SWIN_MLP.EMBED_DIM,
                        depths=config.MODEL.SWIN_MLP.DEPTHS,
                        num_heads=config.MODEL.SWIN_MLP.NUM_HEADS,
                        window_size=config.MODEL.SWIN_MLP.WINDOW_SIZE,
                        mlp_ratio=config.MODEL.SWIN_MLP.MLP_RATIO,
                        drop_rate=config.MODEL.DROP_RATE,
                        drop_path_rate=config.MODEL.DROP_PATH_RATE,
                        ape=config.MODEL.SWIN_MLP.APE,
                        patch_norm=config.MODEL.SWIN_MLP.PATCH_NORM,
                        use_checkpoint=config.TRAIN.USE_CHECKPOINT)
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
