import torch
import numpy as np

# ==========================================
# 1. 模拟 metric.py 中的读取逻辑 (原封不动搬运)
# ==========================================
def simulate_vectorize_metadata(meta_list, dataset_name):
    print(f"\n🚀 开始模拟读取 (Dataset: {dataset_name})")
    print("-" * 60)
    
    # 模拟 CASIA-B / CCPG 的读取逻辑
    for i, m in enumerate(meta_list):
        print(f"📄 [样本 {i}] 原始字典: {m}")
        
        # --- 核心测试点 ---
        # 你的 metric.py 里就是这么写的：m.get('view', '000')
        read_view = m.get('view', 'MISSING_DEFAULT_000') 
        read_cond = str(m.get('cond', 'MISSING_DEFAULT_nm')).split('-')[0]
        
        # 打印读取结果
        if read_view == 'MISSING_DEFAULT_000':
            print(f"❌ [读取失败] View 读到了默认值！代码在找 'view'，但没找到。")
        else:
            print(f"✅ [读取成功] View = {read_view}")
            
        if read_cond == 'MISSING_DEFAULT_nm':
            print(f"❌ [读取失败] Cond 读到了默认值！代码在找 'cond'，但没找到。")
        else:
            print(f"✅ [读取成功] Cond = {read_cond}")
        print("-" * 60)

# ==========================================
# 2. 构造测试数据
# ==========================================

# 🔴 情况 1：你现在的字典结构 (带 tar_ 前缀)
wrong_data = [
    {
        "sid": "001",
        "tar_view": "090",       # <--- 只有 tar_view
        "tar_cond": "bg-01",     # <--- 只有 tar_cond
        "ref_view": "000",
        "ref_cond": "nm-01"
    }
]

# 🟢 情况 2：我建议修改后的字典结构 (标准 Key)
correct_data = [
    {
        "sid": "001",
        "view": "090",           # <--- 伪装成了标准 view
        "cond": "bg-01",         # <--- 伪装成了标准 cond
        "ref_view": "000",
        "ref_cond": "nm-01"
    }
]

# ==========================================
# 3. 运行测试
# ==========================================
print("\n🔥 测试 1：使用原始字典 (带 tar_ 前缀)")
simulate_vectorize_metadata(wrong_data, 'CASIA-B')

print("\n\n🔥 测试 2：使用修正字典 (标准 Key)")
simulate_vectorize_metadata(correct_data, 'CASIA-B')