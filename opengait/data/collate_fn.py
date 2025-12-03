import torch

def get_collate_fn(processor, mode='train'):
    """
    数据整理工厂函数
    Args:
        processor: CLIPProcessor 实例 (在 main.py 里初始化)
        mode: 'train' or 'test'
    """
    
    def train_collate_fn(batch):
        """
        训练时的 Collate:
        1. 视觉数据: 堆叠成 Tensor [B, T, C, H, W] (Dataset 保证了 T 一致)
        2. 文本数据: Tokenize 成 Tensor [B, 77]
        """
        # 解包 Dataset 返回的 tuple
        # batch item: (ref, tar, caption, caption_inv, task, meta)
        refs, tars, caps, invs, _, _ = zip(*batch)

        # 1. 视觉数据堆叠
        ref_batch = torch.stack(refs)
        tar_batch = torch.stack(tars)

        # 2. 文本 Tokenization (使用传入的 processor)
        # padding='max_length' 保证长度统一为 77 (CLIP 标准)
        txt_out = processor(
            text=list(caps), 
            padding="max_length", 
            truncation=True, 
            max_length=77, 
            return_tensors="pt"
        )
        
        inv_out = processor(
            text=list(invs), 
            padding="max_length", 
            truncation=True, 
            max_length=77, 
            return_tensors="pt"
        )

        # 返回顺序必须与 Model.forward 的输入对应
        # (ref, tar, txt_ids, txt_mask, inv_ids, inv_mask)
        return (
            ref_batch, 
            tar_batch,
            txt_out['input_ids'], 
            txt_out['attention_mask'],
            inv_out['input_ids'], 
            inv_out['attention_mask']
        )

    def test_collate_fn(batch):
        """
        测试时的 Collate:
        1. 视觉数据: 保持为 List[Tensor]，因为测试时帧数可能不同 (max_frames='all')
        2. 文本数据: Tokenize
        3. 元数据: 保留 task 和 meta 用于计算指标
        """
        # batch item: (ref, tar, caption, caption_inv, task, meta)
        refs, tars, caps, _, tasks, metas = zip(*batch)

        # 1. 视觉数据保持 List 状态 (处理变长序列的关键)
        ref_batch = list(refs)
        tar_batch = list(tars)

        # 2. 文本 Tokenization
        txt_out = processor(
            text=list(caps), 
            padding="max_length", 
            truncation=True, 
            max_length=77, 
            return_tensors="pt"
        )

        # 返回顺序与 Test Engine 对应
        # (ref_list, tar_list, txt_ids, txt_mask, task_list, meta_list)
        return (
            ref_batch, 
            tar_batch,
            txt_out['input_ids'], 
            txt_out['attention_mask'],
            list(tasks), 
            list(metas)
        )

    return train_collate_fn if mode == 'train' else test_collate_fn