import torch
import numpy as np


# 默认的collate函数
# 参考torch.utils.data._utils.collate下的default_collate
def default_collate(batch):  # 这里batch就是data
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out=None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)  # 在0这个batch堆叠起来
    elif elem_type.__module__ == 'numpy':
        return default_collate([torch.as_tensor(b) for b in batch])
    else:
        raise NotImplementedError


def collate_fn(batch):
    """Only collate dict value in to a list. E.g. meta data dict and img_info
        dict will be collated."""
    elem = batch[0]
    if isinstance(elem, dict):
        return {key: collate_fn([d[key] for d in batch]) for key in elem}
    else:
        return batch
