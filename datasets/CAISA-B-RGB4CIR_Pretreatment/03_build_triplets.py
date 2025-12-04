import json
import random
import os
from tqdm import tqdm
from collections import defaultdict

# ================= 配置区域 =================
# 路径配置 (请根据你的实际目录结构调整)
META_FILE = '../CASIA-B_RGB_JSON/meta_casiab_static.json'
TEMPLATE_FILE = '../CASIA-B_RGB_JSON/templates_instruction.json'
SPLIT_FILE = '../CASIA-B_RGB_JSON/CASIA-B/CASIA-B.json' # 新增：划分文件路径
OUTPUT_TRAIN = '../CASIA-B_RGB_JSON/CASIA-B/casiab_cir_final.json'

# 🔥 核心修改：区分训练和测试的采样强度
TRAIN_MAX_PAIRS = 800   # 训练集采样深度 (保持高密度)
TEST_MAX_PAIRS = 200    # 测试集采样深度 (降低密度，加速评估)

# ===========================================

# === 1. 视角描述池 ===
VIEW_TEXT_POOL = {
    "000": ["a front view", "a frontal angle", "a 0-degree view", "a face-to-face view", "a view facing the camera"],
    "018": ["a front-side view", "an oblique angle", "a slight frontal angle", "a front-quarter view"],
    "036": ["a front-side view", "an oblique angle", "a front-quarter view", "a diagonal front view"],
    "054": ["a front-side view", "an oblique angle", "a sharp frontal angle", "a semi-frontal view"],
    "072": ["a side view", "a profile view", "a slight profile", "a near-side view"],
    "090": ["a side view", "a profile view", "a lateral view", "a 90-degree view", "a side-on view"],
    "108": ["a side view", "a profile view", "a 108-degree view", "a past-side view"],
    "126": ["a back-side view", "a rear-oblique view", "a rear-quarter view", "a view walking away at an angle"],
    "144": ["a back-side view", "a rear-oblique view", "a rear-quarter view", "an off-center back view"],
    "162": ["a back-side view", "a rear-oblique view", "a slight rear angle", "an almost back view"],
    "180": ["a back view", "a rear view", "a dorsal view", "a 180-degree view", "a view seen from behind"]
}

# === 2. 粗粒度映射 ===
COARSE_MAP = {
    "000": "front",
    "018": "front-side", "036": "front-side", "054": "front-side",
    "072": "side", "090": "side", "108": "side",
    "126": "back-side", "144": "back-side", "162": "back-side",
    "180": "back"
}

def safe_fill_view(template, view_text):
    return template.replace("{view}", view_text)

def get_instruction(src_item, tgt_item, templates):
    src_c, src_v = src_item['condition'], src_item['view']
    tgt_c, tgt_v = tgt_item['condition'], tgt_item['view']

    # --- A. 状态部分 ---
    state_instr = ""
    # 根据 Condition 组合选择对应模板
    # 假设 templates 键名与 templates_instruction.json 一致
    if src_c == 'nm' and tgt_c == 'bg': state_instr = random.choice(templates['source_nm_target_bg'])
    elif src_c == 'bg' and tgt_c == 'nm': state_instr = random.choice(templates['source_bg_target_nm'])
    elif src_c == 'nm' and tgt_c == 'cl': state_instr = random.choice(templates['source_nm_target_cl'])
    elif src_c == 'cl' and tgt_c == 'nm': state_instr = random.choice(templates['source_cl_target_nm'])
    elif src_c == 'bg' and tgt_c == 'cl': state_instr = random.choice(templates['source_bg_target_cl'])
    elif src_c == 'cl' and tgt_c == 'bg': state_instr = random.choice(templates['source_cl_target_bg'])
    
    # --- B. 视角部分 ---
    view_instr = ""
    src_coarse = COARSE_MAP.get(src_v, src_v)
    tgt_coarse = COARSE_MAP.get(tgt_v, tgt_v)
    
    if src_coarse != tgt_coarse:
        tpl = random.choice(templates['change_view'])
        tgt_angle = tgt_item['view']
        potential_texts = VIEW_TEXT_POOL.get(tgt_angle, [f"{tgt_angle} degree view"]) 
        view_text = random.choice(potential_texts)
        view_instr = safe_fill_view(tpl, view_text)

    # --- C. 组装 ---
    final_caption = ""
    task_type = "unknown"
    
    # Case 1: Composite
    if state_instr and view_instr:
        conn = random.choice(templates['connectors'])
        s_text = state_instr.rstrip('.')
        v_text = view_instr.rstrip('.')
        
        if random.random() > 0.5:
            p2_content = v_text[0].lower() + v_text[1:] if len(v_text) > 0 else ""
            final_caption = f"{s_text}{conn}{p2_content}."
        else:
            p2_content = s_text[0].lower() + s_text[1:] if len(s_text) > 0 else ""
            final_caption = f"{v_text}{conn}{p2_content}."
        task_type = "composite_change"
        
    # Case 2: Attribute Only
    elif state_instr:
        final_caption = state_instr
        task_type = "attribute_change"
        
    # Case 3: Viewpoint Only
    elif view_instr and src_c == tgt_c:
        final_caption = view_instr
        task_type = "viewpoint_change"
    
    return final_caption, task_type

def build():
    print("Loading metadata and templates...")
    try:
        with open(META_FILE, 'r') as f:
            meta_db = json.load(f)
        with open(TEMPLATE_FILE, 'r') as f:
            templates = json.load(f)
        # 🔥 加载划分文件
        with open(SPLIT_FILE, 'r') as f:
            split_cfg = json.load(f)
    except FileNotFoundError as e:
        print(f"❌ Error: File not found {e.filename}. Check paths.")
        return

    # 解析训练集和测试集 ID
    train_ids = set(split_cfg['TRAIN_SET'])
    test_ids = set(split_cfg['TEST_SET'])
    print(f"Split Loaded: {len(train_ids)} Train IDs, {len(test_ids)} Test IDs")

    # 1. 重建索引
    data_index = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for item in meta_db:
        # 这里转换 sid 为字符串以防万一
        data_index[str(item['sid'])][item['condition']][item['view']].append(item)
        
    all_triplets = []
    stats = defaultdict(int)
    
    sorted_ids = sorted(data_index.keys())
    print("🚀 Starting triplet generation...")
    
    for sid in tqdm(sorted_ids):
        # 🔥 核心修改：根据 ID 决定采样数量
        if sid in train_ids:
            current_max_pairs = TRAIN_MAX_PAIRS
        elif sid in test_ids:
            current_max_pairs = TEST_MAX_PAIRS
        else:
            # 如果有些 ID 不在划分里，可以选择跳过或给个默认值
            continue

        conds = data_index[sid]
        
        # 收集可用节点
        nodes = []
        for c in conds:
            for v in conds[c]:
                if len(conds[c][v]) > 0:
                    nodes.append((c, v))
        
        if len(nodes) < 2: continue

        # --- 采样循环 ---
        for _ in range(current_max_pairs):
            src_node = random.choice(nodes)
            tgt_node = random.choice(nodes)
            
            if src_node == tgt_node: continue
            
            src_c, src_v = src_node
            tgt_c, tgt_v = tgt_node
            
            ref_item = random.choice(conds[src_c][src_v])
            tar_item = random.choice(conds[tgt_c][tgt_v])
            
            if ref_item['seq_path'] == tar_item['seq_path']: continue

            # 生成指令
            fwd_caption, task_type = get_instruction(ref_item, tar_item, templates)
            if not fwd_caption: continue

            inv_caption, _ = get_instruction(tar_item, ref_item, templates)

            all_triplets.append({
                "sid": sid,
                "dataset": "CASIA-B",
                "task": task_type,
                "caption": fwd_caption,
                "caption_inv": inv_caption,
                "ref": ref_item,
                "tar": tar_item,
                # 可选：标记是训练还是测试样本，方便后续 debug
                "split": "train" if sid in train_ids else "test"
            })
            stats[task_type] += 1

    # 保存结果
    os.makedirs(os.path.dirname(OUTPUT_TRAIN), exist_ok=True)
    with open(OUTPUT_TRAIN, 'w') as f:
        json.dump(all_triplets, f, indent=4)
    
    print(f"✅ Done! Total samples: {len(all_triplets)}")
    print(f"   Train Sampling: {TRAIN_MAX_PAIRS}/ID")
    print(f"   Test Sampling : {TEST_MAX_PAIRS}/ID")
    print("📊 Task Distribution:", dict(stats))

if __name__ == '__main__':
    build()