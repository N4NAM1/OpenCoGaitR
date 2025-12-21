import os
import argparse
from collections import Counter
from tqdm import tqdm

def analyze_sustech_structure(data_root):
    if not os.path.exists(data_root):
        print(f"❌ Error: Data root '{data_root}' does not exist.")
        return

    subjects = sorted([d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))])
    print(f"🔍 Found {len(subjects)} subjects. Scanning states...")

    # 用于统计完整的文件夹名 (e.g., "01-bg")
    state_folder_counter = Counter()
    
    # 用于统计拆解后的原子属性 (e.g., "bg", "nt")
    atomic_attr_counter = Counter()
    
    # 记录一些样本路径用于检查
    sample_paths = {} 

    for pid in tqdm(subjects):
        pid_path = os.path.join(data_root, pid)
        
        # 获取该人的所有状态文件夹
        states = sorted([d for d in os.listdir(pid_path) if os.path.isdir(os.path.join(pid_path, d))])
        
        for state in states:
            state_folder_counter[state] += 1
            
            # 保存一个示例路径，方便人工核对
            if state not in sample_paths:
                sample_paths[state] = os.path.join(pid, state)

            # 尝试拆解属性
            # 假设命名规则是 "00-nm", "01-bg-nt" 这种连字符分隔
            parts = state.lower().split('-')
            
            # 过滤掉纯数字编号 (如 '00', '01')，只保留语义部分
            semantic_parts = [p for p in parts if not p.isdigit()]
            
            for p in semantic_parts:
                atomic_attr_counter[p] += 1

    print("\n" + "="*50)
    print("📊 Analysis Report for SUSTech1K")
    print("="*50)

    print(f"\n1. Top 20 Common State Folders (Total Unique: {len(state_folder_counter)}):")
    print("-" * 30)
    for state, count in state_folder_counter.most_common(20):
        print(f"  - [{count:5d} occurrences] : {state}")

    print(f"\n2. Atomic Attributes Found (Split by '-'):")
    print("-" * 30)
    # 按频率排序
    for attr, count in atomic_attr_counter.most_common():
        print(f"  - {attr:<10} : {count:5d} times")

    print("\n3. Sample Paths for Verification:")
    print("-" * 30)
    # 随机展示几个原子属性对应的完整路径
    seen_atoms = set()
    for attr, _ in atomic_attr_counter.most_common():
        # 找到包含这个属性的一个文件夹示例
        for state_name, sample_path in sample_paths.items():
            if attr in state_name.lower():
                print(f"  - Attribute '{attr}': .../{sample_path}")
                break

    print("\n" + "="*50)
    print("💡 Suggestion for generate_meta.py:")
    print("Based on Section 2 above, update your 'parse_sustech_attributes' function to handle these specific keys.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 请修改为你的实际路径
    parser.add_argument('--data_root', type=str, default='/root/autodl-tmp/SUSTech1K_Processed/RGB', help='SUSTech1K data root')
    args = parser.parse_args()
    
    analyze_sustech_structure(args.data_root)