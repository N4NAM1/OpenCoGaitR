import collections
import torch
import numpy as np
import torchvision.transforms as T
import random

class BaseTransformer(object):
    def __init__(self):
        pass

    def __call__(self, x):
        return x

class Compose(BaseTransformer):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x

class ToTensor(BaseTransformer):
    def __call__(self, x):
        return T.functional.to_tensor(x)

class Normalize(BaseTransformer):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return T.functional.normalize(x, self.mean, self.std)

class Resize(BaseTransformer):
    def __init__(self, size, interpolation=T.InterpolationMode.BICUBIC):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, x):
        return T.functional.resize(x, self.size, self.interpolation)

class CLIPNormalize(BaseTransformer):
    """
    自动使用 OpenAI CLIP 的归一化参数
    """
    def __init__(self):
        # CLIP 官方标准参数
        self.mean = [0.48145466, 0.4578275, 0.40821073]
        self.std = [0.26862954, 0.26130258, 0.27577711]

    def __call__(self, x):
        return T.functional.normalize(x, self.mean, self.std)

class CLIPImageProcessor(BaseTransformer):
    """
    🔥 懒人专用：一键搞定 CLIP 的所有预处理
    Resize(224) -> ToTensor -> Normalize
    """
    def __init__(self, size=224):
        self.pipeline = T.Compose([
            T.Resize((size, size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            CLIPNormalize() # 复用上面的类
        ])

    def __call__(self, x):
        return self.pipeline(x)

# 更新工厂函数
def get_transform(transform_cfg):
    if transform_cfg is None: return lambda x: x
    
    tr_list = []
    for tr_s in transform_cfg:
        tr_name = tr_s['type']
        tr_args = tr_s.copy()
        tr_args.pop('type')

        if tr_name == 'Compose': continue
        
        # === 新增的自动类 ===
        if tr_name == 'CLIPNormalize':
            tr_list.append(CLIPNormalize())
        elif tr_name == 'CLIPImageProcessor':
            tr_list.append(CLIPImageProcessor(**tr_args))
            
        # === 原有的基础类 ===
        elif tr_name == 'Resize':
            if isinstance(tr_args.get('size'), int):
                tr_args['size'] = (tr_args['size'], tr_args['size'])
            tr_list.append(Resize(**tr_args))
        elif tr_name == 'ToTensor':
            tr_list.append(ToTensor())
        elif tr_name == 'Normalize':
            tr_list.append(Normalize(**tr_args))
        else:
            try:
                Cls = getattr(T, tr_name)
                tr_list.append(Cls(**tr_args))
            except AttributeError:
                print(f"⚠️ Warning: Transform {tr_name} not found.")

    return Compose(tr_list)