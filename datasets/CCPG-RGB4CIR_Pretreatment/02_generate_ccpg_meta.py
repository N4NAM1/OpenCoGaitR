import os
import json
import argparse
from tqdm import tqdm

def generate_ccpg_meta_aligned(data_root, output_path):
    """
    生成 CCPG 元数据 (Folder-Based Version)
    
    逻辑升级：
    1. seq_path 指向文件夹 (e.g. "001/U0_D0_BG/01_0")
    2. 验证逻辑：只要文件夹里有图片(.jpg/.png) 或者 PKL(.pkl)，就被视为有效序列。
       (不再依赖具体的文件名 *-aligned-rgbs.pkl，防止因改名或解包导致生成失败)
    """
    meta_data = {}
    
    if not os.path.exists(data_root):
        raise FileNotFoundError(f"Data root not found: {data_root}")

    subjects = sorted([d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))])
    print(f"Found {len(subjects)} subjects in {data_root}")

    count = 0
    # 遍历所有 Subject
    for pid in tqdm(subjects, desc="Indexing CCPG"):
        pid_path = os.path.join(data_root, pid)
        if not os.path.isdir(pid_path): continue
        
        statuses = sorted(os.listdir(pid_path))
        
        # 遍历所有状态 (U0_D0_BG ...)
        for status in statuses:
            status_path = os.path.join(pid_path, status)
            if not os.path.isdir(status_path): continue
            
            view_seqs = sorted(os.listdir(status_path))
            
            # 遍历所有序列 (01_0, 02_0 ...)
            for vs_folder in view_seqs:
                # 过滤掉非序列文件夹
                if '_' not in vs_folder: continue
                try:
                    view_code, seq_num = vs_folder.split('_')
                except ValueError:
                    continue

                # 序列绝对路径
                seq_abs_path = os.path.join(status_path, vs_folder)
                
                # 🔥 核心修改：智能验证数据有效性
                # 不再硬编码检查 xxx-aligned-rgbs.pkl
                # 而是检查文件夹里是否有数据 (图片 或 PKL)
                has_valid_data = False
                if os.path.isdir(seq_abs_path):
                    try:
                        # 快速扫描文件夹内容
                        files = os.listdir(seq_abs_path)
                        # 只要包含 .jpg, .png 或 .pkl 任意一种，就算有效
                        if any(f.endswith(('.jpg', '.png', '.pkl', '.pkl.bak')) for f in files):
                            has_valid_data = True
                    except OSError:
                        pass
                
                if has_valid_data:
                    # === 构造元数据 ===
                    
                    # 1. 相对路径 (指向文件夹，Loader 会自己去找里面的文件)
                    rel_path = os.path.join(pid, status, vs_folder)
                    
                    # 2. 静态描述
                    base_cap = "{Subject} walking"
                    if "BG" in status:
                        base_cap += " carrying a bag"
                    if "U" in status and "U0" not in status:
                        base_cap += ", wearing a different upper outfit"
                    if "D" in status and "D0" not in status:
                        base_cap += ", wearing different pants"
                    base_cap += "."
                    
                    # 3. 构造 Key (Unique ID)
                    key = f"{pid}_{status}_{view_code}_{seq_num}"
                    
                    meta_data[key] = {
                        "sid": pid,
                        "condition": status,
                        "view": view_code,
                        "seq_path": rel_path,      # ✅ 纯文件夹路径
                        "static_caption": base_cap
                    }
                    count += 1

    # 保存结果
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(meta_data, f, indent=4)
    print(f"✅ Saved meta data to {output_path}. Total Sequences: {count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 你的 CCPG RGB 数据集根目录
    parser.add_argument('--data_root', type=str, required=True, help='CCPG_Processed RGB absolute path')
    parser.add_argument('--output', type=str, default='datasets/CCPG_RGB_JSON/CCPG/meta_ccpg.json')
    args = parser.parse_args()
    
    generate_ccpg_meta_aligned(args.data_root, args.output)