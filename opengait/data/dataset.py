import os
import json
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

# 引入我们之前写好的 transform 模块
from .transform import get_transform

class GaitCIRDataset(Dataset):
    def __init__(self, data_cfg, mode='train'):       
        self.mode = mode
        self.data_cfg = data_cfg
        
        # === 1. 基本配置 ===
        self.dataset_name = data_cfg['dataset_name'].upper()
        self.dataset_root = data_cfg['dataset_root']
        self.use_features = data_cfg.get('use_features', False)
        self.feature_root = data_cfg.get('feature_root', self.dataset_root)
        self.rgb_subfolder = "RGB" 
        
        # === 2. 采样配置 ===
        if mode == 'train':
            self.max_frames = data_cfg.get('train_max_frames', 30)
        else:
            self.max_frames = data_cfg.get('test_max_frames', 'all')
            
        self.subject_token = "the person" 

        # === 3. 初始化 Transform ===
        tr_cfg = data_cfg.get('transform', [])
        self.transform = get_transform(tr_cfg)
        print(f"[{mode.upper()}] Transform Initialized.")

        # === 4. 加载索引文件 ===
        json_path = data_cfg['train_json']
        print(f"[{mode.upper()}] Loading Index: {json_path}")
        with open(json_path, 'r') as f:
            all_data = json.load(f)
            
        # === 5. 数据划分 ===
        split_config_path = data_cfg.get('split_config', None)
        if split_config_path and os.path.exists(split_config_path):
            with open(split_config_path, 'r') as f:
                split_cfg = json.load(f)
            subset_key = 'TRAIN_SET' if mode == 'train' else 'TEST_SET'
            allowed_ids = set(split_cfg[subset_key])
            self.data = [item for item in all_data if str(item['sid']) in allowed_ids]
            print(f"✅ Filter Applied ({subset_key}): {len(all_data)} -> {len(self.data)} triplets kept.")
        else:
            self.data = all_data

    def _load_sequence(self, rel_seq_path):
        """ 
        读取图像序列 (内存优化版)
        策略：先采样索引，再读取文件。避免全量加载。
        """
        # 构建路径
        base_path = os.path.join(self.dataset_root, self.rgb_subfolder, rel_seq_path)
        pkl_path = base_path if base_path.endswith('.pkl') else base_path + ".pkl"
        dir_path = base_path

        seq_data = None
        is_raw_images = False
        
        # --- 策略 A: PKL (通常无法避免全量加载，除非 pickle 结构特殊) ---
        if os.path.exists(pkl_path):
            try:
                with open(pkl_path, 'rb') as f:
                    seq_data = pickle.load(f)
            except Exception as e:
                print(f"⚠️ [Pickle Error] {pkl_path}: {e}")

        # --- 策略 B: 文件夹 (Raw Images) - 内存优化的关键 ---
        elif os.path.isdir(dir_path):
            # 1. 先只读文件名，不读图片内容！
            imgs = sorted([f for f in os.listdir(dir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            if len(imgs) > 0:
                seq_data = imgs  # 此时 seq_data 只是文件名列表，内存占用极小
                is_raw_images = True

        if seq_data is None or len(seq_data) == 0:
            return None

        total = len(seq_data)
        
        # --- 采样逻辑 (Sample First) ---
        if self.mode == 'train':
            frames_to_sample = self.max_frames if isinstance(self.max_frames, int) else 30
            replace = total < frames_to_sample
            indices = sorted(np.random.choice(total, frames_to_sample, replace=replace))
        else:
            if self.max_frames == "all" or self.max_frames is all: 
                indices = np.arange(total)
            else:
                frames_to_sample = int(self.max_frames)
                indices = np.linspace(0, total - 1, frames_to_sample, dtype=int)

        # --- 加载与 Transform (Load Later) ---
        final_imgs = []
        
        for idx in indices:
            # 分支 1: 原图模式 (按需读取，省内存！)
            if is_raw_images:
                img_name = seq_data[idx] # seq_data 是文件名列表
                img_path = os.path.join(dir_path, img_name)
                try:
                    # 直接用 PIL 读取，避免 CV2 转换开销
                    img = Image.open(img_path).convert('RGB')
                except:
                    # 坏图兜底：生成黑图
                    img = Image.new('RGB', (224, 224))
            
            # 分支 2: PKL 模式 (seq_data 已经是加载好的对象列表)
            else:
                img = seq_data[idx]
                if isinstance(img, np.ndarray):
                    if img.ndim == 3 and img.shape[0] == 3: img = img.transpose(1, 2, 0)
                    elif img.ndim == 3 and img.shape[0] == 1: img = img.squeeze(0)
                    if img.dtype != np.uint8: img = img.astype(np.uint8)
                    img = Image.fromarray(img)
            
            # 应用 Transform
            if self.transform:
                img = self.transform(img)
            
            final_imgs.append(img)

        if len(final_imgs) > 0 and isinstance(final_imgs[0], torch.Tensor):
            return torch.stack(final_imgs)
        return final_imgs

    def _load_features(self, rel_seq_path):
        # ... (Feature 模式保持不变，因为它本来就很小) ...
        path = os.path.join(self.feature_root, rel_seq_path + ".pt")
        if not os.path.exists(path): 
            path = os.path.join(self.feature_root, rel_seq_path)
        if not os.path.exists(path): return None
        data = torch.load(path, map_location='cpu')
        total = data.size(0)
        if total == 0: return None
        
        if self.mode == 'train':
            frames_to_sample = self.max_frames if isinstance(self.max_frames, int) else 30
            replace = total < frames_to_sample
            indices = sorted(np.random.choice(total, frames_to_sample, replace=replace))
        else:
            if self.max_frames == "all" or self.max_frames is all: indices = np.arange(total)
            else: frames_to_sample = int(self.max_frames); indices = np.linspace(0, total - 1, frames_to_sample, dtype=int)
        return data[indices]

    def __getitem__(self, idx):
        retries = 0
        max_retries = 10
        while True:
            if retries > max_retries:
                item = self.data[idx]
                print(f"💀 [Fatal] Failed: {item['sid']} - {item['ref']['seq_path']}")
                raise RuntimeError(f"❌ Max retries exceeded")

            item = self.data[idx]
            try:
                if self.use_features:
                    ref_out = self._load_features(item['ref']['seq_path'])
                    tar_out = self._load_features(item['tar']['seq_path'])
                else:
                    ref_out = self._load_sequence(item['ref']['seq_path'])
                    tar_out = self._load_sequence(item['tar']['seq_path'])

                if ref_out is None or tar_out is None:
                    raise ValueError(f"Data missing")

                caption = item['caption'].replace("{subject}", self.subject_token)
                caption_inv = item.get('caption_inv', "").replace("{subject}", self.subject_token)
                
                return ref_out, tar_out, caption, caption_inv, item['task'], {
                    "sid": str(item['sid']), 

                    # 显式重命名为评估器需要的 Key
                    "tar_cond": str(item['tar']['condition']), 
                    "tar_view": str(item['tar']['view']),

                    # 新增 Reference 信息
                    "ref_cond": str(item['ref']['condition']),
                    "ref_view": str(item['ref']['view']),

                }
            except Exception:
                if self.mode == 'train': idx = np.random.randint(len(self.data)); retries += 1
                else: raise 
    
    def __len__(self): return len(self.data)