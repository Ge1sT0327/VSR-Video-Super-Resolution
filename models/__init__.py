from .basicvsr import BasicVSR, BasicVSRLight
from .vsr_3dcnn import VSR3DCNN
from .vsr_convlstm import VSRConvLSTM

MODELS = {
    'basicvsr': BasicVSR,
    'basicvsr_light': BasicVSRLight,
    'vsr_3dcnn': VSR3DCNN,
    'vsr_convlstm': VSRConvLSTM,
}


def create_model(name, scale=4, **kwargs):
    """工厂函数：根据名称创建模型"""
    if name == 'basicvsr':
        return BasicVSR(scale=scale, **kwargs)
    elif name == 'basicvsr_light':
        return BasicVSRLight(scale=scale, **kwargs)
    elif name == 'vsr_3dcnn':
        return VSR3DCNN(scale=scale, **kwargs)
    elif name == 'vsr_convlstm':
        return VSRConvLSTM(scale=scale, **kwargs)
    else:
        raise ValueError(f"未知模型: {name}. 可选: {list(MODELS.keys())}")
