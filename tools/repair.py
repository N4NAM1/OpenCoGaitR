import os
import pickle
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

# ================= 配置区域 =================
# 🚨 请确认这是你的 SUSTech1K 数据集路径
DATA_ROOT = "/root/autodl-tmp/SUSTech1K_Processed/RGB" 

# 解压后是否删除源文件？
DELETE_PKL = False 
# 并发线程数
NUM_THREADS = 16
# ===========================================

def process_one_pkl(pkl_path):
    file_name = os.path.basename(pkl_path)
    lower_name = file_name.lower()

    # 1. 过滤逻辑：跳过 Ratios, masks, pose 等非图像文件
    if "ratios" in lower_name or "mask" in lower_name or "pose" in lower_name:
        return False

    try:
        # 2. 读取 PKL
        with open(pkl_path, 'rb') as f:
            seq_data = pickle.load(f)
            
        # 3. 安全检查
        if seq_data is None: return False
        if isinstance(seq_data, (list, tuple)) and len(seq_data) == 0: return False
        if isinstance(seq_data, np.ndarray) and seq_data.size == 0: return False
            
        # 4. 🔥 核心修改：路径处理逻辑
        # 如果是 RGB_raw 文件，直接解压到【当前父目录】，不创建子文件夹
        if "rgb_raw" in lower_name:
            save_dir = os.path.dirname(pkl_path)
        else:
            # 其他情况（如标准 CASIA-B），还是创建同名子文件夹比较安全
            save_dir = pkl_path.replace('.pkl', '')
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
            
        # 5. 逐帧保存
        for i, frame in enumerate(seq_data):
            save_path = os.path.join(save_dir, f"{i:03d}.jpg")
            
            # 断点续传：如果已存在则跳过
            if os.path.exists(save_path): continue

            # 处理 Numpy Array
            if isinstance(frame, np.ndarray):
                # 维度修正 (3, H, W) -> (H, W, 3)
                if frame.ndim == 3 and frame.shape[0] == 3:
                    frame = frame.transpose(1, 2, 0)
                elif frame.ndim == 3 and frame.shape[0] == 1:
                    frame = frame.squeeze(0)
                
                # 类型修正
                if frame.dtype != np.uint8:
                    frame = frame.astype(np.uint8)
                
                # 保存 (转 BGR)
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imwrite(save_path, frame_bgr)
                
            # 处理 PIL Image
            elif isinstance(frame, Image.Image):
                frame.save(save_path, quality=95)
        
        # 6. (可选) 删除源文件
        if DELETE_PKL:
            os.remove(pkl_path)
            
        return True

    except Exception as e:
        print(f"❌ Error processing {file_name}: {e}")
        return False

def main():
    print(f"🔍 Scanning {DATA_ROOT} for .pkl files...")
    pkl_files = []
    # 递归扫描
    for root, dirs, files in os.walk(DATA_ROOT):
        for f in files:
            if f.endswith('.pkl'):
                pkl_files.append(os.path.join(root, f))
    
    if not pkl_files:
        print("✅ No .pkl files found.")
        return

    print(f"📦 Found {len(pkl_files)} PKL files. Filtering and unpacking...")
    
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        results = list(tqdm(executor.map(process_one_pkl, pkl_files), total=len(pkl_files)))
        
    success_count = sum(results)
    print(f"\n✅ Done! Successfully unpacked {success_count} files.")
    
    if not DELETE_PKL:
        print("\n💡 Tip: Please verify images.")
        print("   Then run this to delete all .pkl files:")
        print(f"   find {DATA_ROOT} -name '*.pkl' -delete")

if __name__ == '__main__':
    main()