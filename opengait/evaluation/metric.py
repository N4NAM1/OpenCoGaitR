import torch
import numpy as np
import torch.nn.functional as F

from prettytable import PrettyTable
from utils import is_tensor
from collections import defaultdict


def cuda_dist(x, y, metric='euc'):
    x = torch.from_numpy(x).cuda()
    y = torch.from_numpy(y).cuda()
    if metric == 'cos':
        x = F.normalize(x, p=2, dim=1)  # n c p
        y = F.normalize(y, p=2, dim=1)  # n c p
    num_bin = x.size(2)
    n_x = x.size(0)
    n_y = y.size(0)
    dist = torch.zeros(n_x, n_y).cuda()
    for i in range(num_bin):
        _x = x[:, :, i]
        _y = y[:, :, i]
        if metric == 'cos':
            dist += torch.matmul(_x, _y.transpose(0, 1))
        else:
            _dist = torch.sum(_x ** 2, 1).unsqueeze(1) + torch.sum(_y ** 2, 1).unsqueeze(
                0) - 2 * torch.matmul(_x, _y.transpose(0, 1))
            dist += torch.sqrt(F.relu(_dist))
    return 1 - dist/num_bin if metric == 'cos' else dist / num_bin


def mean_iou(msk1, msk2, eps=1.0e-9):
    if not is_tensor(msk1):
        msk1 = torch.from_numpy(msk1).cuda()
    if not is_tensor(msk2):
        msk2 = torch.from_numpy(msk2).cuda()
    n = msk1.size(0)
    inter = msk1 * msk2
    union = ((msk1 + msk2) > 0.).float()
    miou = inter.view(n, -1).sum(-1) / (union.view(n, -1).sum(-1) + eps)
    return miou


def compute_ACC_mAP(distmat, q_pids, g_pids, q_views=None, g_views=None, rank=1):
    num_q, _ = distmat.shape
    # indices = np.argsort(distmat, axis=1)
    # matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    all_ACC = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        q_idx_dist = distmat[q_idx]
        q_idx_glabels = g_pids
        if q_views is not None and g_views is not None:
            q_idx_mask = np.isin(g_views, q_views[q_idx], invert=True) | np.isin(
                g_pids, q_pids[q_idx], invert=True)
            q_idx_dist = q_idx_dist[q_idx_mask]
            q_idx_glabels = q_idx_glabels[q_idx_mask]

        assert(len(q_idx_glabels) >
               0), "No gallery after excluding identical-view cases!"
        q_idx_indices = np.argsort(q_idx_dist)
        q_idx_matches = (q_idx_glabels[q_idx_indices]
                         == q_pids[q_idx]).astype(np.int32)

        # binary vector, positions with value 1 are correct matches
        # orig_cmc = matches[q_idx]
        orig_cmc = q_idx_matches
        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1
        all_ACC.append(cmc[rank-1])

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()

        if num_rel > 0:
            num_valid_q += 1.
            tmp_cmc = orig_cmc.cumsum()
            tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
            tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
            AP = tmp_cmc.sum() / num_rel
            all_AP.append(AP)

    # all_ACC = np.asarray(all_ACC).astype(np.float32)
    ACC = np.mean(all_ACC)
    mAP = np.mean(all_AP)

    return ACC, mAP


def evaluate_rank(distmat, p_lbls, g_lbls, max_rank=50):
    '''
    Copy from https://github.com/Gait3D/Gait3D-Benchmark/blob/72beab994c137b902d826f4b9f9e95b107bebd78/lib/utils/rank.py#L12-L63
    '''
    num_p, num_g = distmat.shape

    if num_g < max_rank:
        max_rank = num_g
        print('Note: number of gallery samples is quite small, got {}'.format(num_g))

    indices = np.argsort(distmat, axis=1)

    matches = (g_lbls[indices] == p_lbls[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each probe
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_p = 0.  # number of valid probe

    for p_idx in range(num_p):
        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        raw_cmc = matches[p_idx]
        if not np.any(raw_cmc):
            # this condition is true when probe identity does not appear in gallery
            continue

        cmc = raw_cmc.cumsum()

        pos_idx = np.where(raw_cmc == 1)    # 返回坐标，此处raw_cmc为一维矩阵，所以返回相当于index
        max_pos_idx = np.max(pos_idx)
        inp = cmc[max_pos_idx] / (max_pos_idx + 1.0)
        all_INP.append(inp)

        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_p += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_p > 0, 'Error: all probe identities do not appear in gallery'

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_p

    return all_cmc, all_AP, all_INP


def evaluate_many(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)   # 对应位置变成从小到大的序号
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(
        np.int32)  # 根据indices调整顺序 g_pids[indices]
    # print(matches)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0.
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()

        pos_idx = np.where(orig_cmc == 1)
        max_pos_idx = np.max(pos_idx)
        inp = cmc[max_pos_idx] / (max_pos_idx + 1.0)
        all_INP.append(inp)

        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP)

    return all_cmc, mAP, mINP




def compute_rank_k_ap(sim_mat, gt_mask, k_list=[1, 5, 10]):
    """
    计算 R@K 和 mAP
    """
    device = sim_mat.device
    num_q = sim_mat.size(0)
    
    # 1. 排序
    scores, indices = torch.sort(sim_mat, dim=1, descending=True)
    
    # 2. 重排 Ground Truth
    gts = torch.gather(gt_mask.float(), 1, indices)
    
    results = {}
    
    # --- R@K ---
    for k in k_list:
        if k > gts.size(1):
            results[f'R{k}'] = 100.0
        else:
            hits = (gts[:, :k].sum(dim=1) > 0).float()
            results[f'R{k}'] = hits.mean().item() * 100.0
            
    # --- mAP ---
    cumsum = torch.cumsum(gts, dim=1)
    ranks = torch.arange(1, gts.size(1) + 1, device=device).unsqueeze(0)
    precision = cumsum / ranks
    
    ap_sum = (precision * gts).sum(dim=1)
    num_rel = gts.sum(dim=1)
    
    mask = num_rel > 0
    ap = torch.zeros(num_q, device=device)
    ap[mask] = ap_sum[mask] / num_rel[mask]
    
    results['mAP'] = ap.mean().item() * 100.0
    
    return results

# =========================================================
# 2. 向量化元数据 (View Groups 解析)
# =========================================================

def vectorize_metadata(meta_list, dataset_name, config, device):
    """
    将 meta 解析为 Tensor，核心：将角度映射为 Group ID
    """
    N = len(meta_list)
    vec_data = {}
    
    # ID
    raw_sids = [m['sid'] for m in meta_list]
    _, sid_indices = np.unique(raw_sids, return_inverse=True)
    vec_data['sid'] = torch.tensor(sid_indices, device=device)

    # === CASIA-B 逻辑 ===
    if 'CASIA-B' in dataset_name:
        # 1. 解析 View Groups
        # config 示例: {'view_groups': {'front': ['000'], 'side': ['090']...}}
        view_groups = config.get('view_groups', {})
        
        # 构建 Angle -> Group ID 映射
        angle_to_gid = {}
        if view_groups:
            # 排序保证 ID 稳定
            sorted_keys = sorted(view_groups.keys())
            for gid, gname in enumerate(sorted_keys):
                for angle in view_groups[gname]:
                    angle_to_gid[str(int(angle))] = gid
                    angle_to_gid[int(angle)] = gid
        
        def get_group_id(angle_val):
            ang_int = int(angle_val)
            # 如果有分组配置，返回组ID；否则返回角度本身(精确匹配)
            return angle_to_group_id.get(ang_int, ang_int)

        angle_to_group_id = angle_to_gid 

        # 2. 提取数据 (Target & Reference)
        tar_grp_ids = []
        ref_grp_ids = []
        
        ref_conds, tar_conds = [], []
        
        for m in meta_list:
            # Target View -> Group ID (用于 Strict 和 Soft)
            tar_grp_ids.append(get_group_id(m.get('tar_view')))
            
            # Target Attribute (Condition) -> 用于 Strict 和 Soft
            tar_conds.append(str(m.get('tar_cond')).split('-')[0]) 
            
            # Reference View -> Group ID (用于判断是否 Change)
            if 'ref_cond' in m:
                ref_grp_ids.append(get_group_id(m['ref_view']))
                ref_conds.append(str(m['ref_cond']).split('-')[0])
            else:
                ref_grp_ids.append(-1)
                ref_conds.append('none')

        # 3. 转 Tensor
        vec_data['tar_view_grp'] = torch.tensor(tar_grp_ids, device=device)
        vec_data['gal_view_grp'] = vec_data['tar_view_grp'] # Gallery 也是 Group ID
        vec_data['ref_view_grp'] = torch.tensor(ref_grp_ids, device=device)

        # Condition 处理
        # 建立统一的 Cond 映射字典
        all_conds = sorted(list(set(ref_conds + tar_conds)))
        cond_map = {c: i for i, c in enumerate(all_conds)}
        
        vec_data['tar_cond'] = torch.tensor([cond_map[c] for c in tar_conds], device=device)
        vec_data['gal_cond'] = vec_data['tar_cond']
        vec_data['ref_cond'] = torch.tensor([cond_map[c] for c in ref_conds], device=device)
        
        # 🔥 计算 GT 变化 (用于 Soft)
        # 1. View Group Changed? (Ref Group != Tar Group)
        vec_data['gt_view_grp_changed'] = (vec_data['ref_view_grp'] != vec_data['tar_view_grp'])
        
        # 2. Attribute Changed? (Ref Cond != Tar Cond)
        vec_data['gt_cond_changed'] = (vec_data['ref_cond'] != vec_data['tar_cond'])

    # === CCPG 逻辑 (保持不变) ===
    elif 'CCPG' in dataset_name:
        def parse(cond_str):
            parts = str(cond_str).split('_')
            u, d, bag = 0, 0, 0
            for p in parts:
                if p.startswith('U'): u = int(p[1:])
                elif p.startswith('D'): d = int(p[1:])
                elif p == 'BG': bag = 1
            return u, d, bag

        ref_u = torch.zeros(N, dtype=torch.long, device=device)
        ref_d = torch.zeros(N, dtype=torch.long, device=device)
        ref_bag = torch.zeros(N, dtype=torch.long, device=device)
        tar_u = torch.zeros(N, dtype=torch.long, device=device)
        tar_d = torch.zeros(N, dtype=torch.long, device=device)
        tar_bag = torch.zeros(N, dtype=torch.long, device=device)
        ref_views, tar_views = [], []

        for i, m in enumerate(meta_list):
            if 'ref_cond' in m:
                ru, rd, rb = parse(m['ref_cond'])
                ref_u[i], ref_d[i], ref_bag[i] = ru, rd, rb
                ref_views.append(m['ref_view'])
            else:
                ref_views.append(m.get('ref_view'))
            
            tu, td, tb = parse(m['tar_cond'])
            tar_u[i], tar_d[i], tar_bag[i] = tu, td, tb
            tar_views.append(m['tar_view'])

        vec_data['ref_u'], vec_data['ref_d'], vec_data['ref_bag'] = ref_u, ref_d, ref_bag
        vec_data['tar_u'], vec_data['tar_d'], vec_data['tar_bag'] = tar_u, tar_d, tar_bag 
        vec_data['gal_u'], vec_data['gal_d'], vec_data['gal_bag'] = tar_u, tar_d, tar_bag
        
        all_views = sorted(list(set(ref_views + tar_views)))
        view_to_id = {v: k for k, v in enumerate(all_views)}
        vec_data['ref_view'] = torch.tensor([view_to_id[v] for v in ref_views], device=device)
        vec_data['tar_view'] = torch.tensor([view_to_id[v] for v in tar_views], device=device)
        
        vec_data['gt_cloth_changed'] = (ref_u != tar_u) | (ref_d != tar_d)
        vec_data['gt_view_changed'] = (vec_data['ref_view'] != vec_data['tar_view'])
        vec_data['gt_bag_changed'] = (ref_bag != tar_bag)

    return vec_data

# =========================================================
# 3. 匹配逻辑 (Core Logic)
# =========================================================

def compute_match_matrices(q_vec, g_vec, dataset_name):
    """
    计算匹配矩阵 (Gallery vs Target)
    """
    mat_id = (q_vec['sid'][:, None] == g_vec['sid'][None, :])
#CASIA-B 逻辑有误
    if 'CASIA-B' in dataset_name:
        # 1. ViewGroup 匹配 (Strict & Soft)
        # 你的要求：Strict要粗粒度，Soft也要对上Group
        ret_view_grp_match = (q_vec['tar_view_grp'][:, None] == g_vec['gal_view_grp'][None, :])
        
        # 2. Attribute 匹配 (Strict & Soft)
        # 你的要求：Strict要严格对应，Soft变了也要严格对应
        ret_cond_match = (q_vec['tar_cond'][:, None] == g_vec['gal_cond'][None, :])
        
        return mat_id, {
            'ret_view_grp_match': ret_view_grp_match, 
            'ret_cond_match': ret_cond_match
        }

    elif 'CCPG' in dataset_name:
        ret_bag_match = (q_vec['tar_bag'][:, None] == g_vec['gal_bag'][None, :])
        ret_u_changed = (g_vec['gal_u'][None, :] != q_vec['ref_u'][:, None]) 
        ret_d_changed = (g_vec['gal_d'][None, :] != q_vec['ref_d'][:, None])
        ret_cloth_changed = ret_u_changed | ret_d_changed
        ret_view_changed = (g_vec['tar_view'][None, :] != q_vec['ref_view'][:, None])
        
        return mat_id, {
            'bag_match': ret_bag_match,
            'ret_cloth_changed': ret_cloth_changed,
            'ret_view_changed': ret_view_changed
        }

    return mat_id, {}

# =========================================================
# 4. 主入口
# =========================================================

def compute_gaitcir_metrics(q_feats, g_feats, q_metas, g_metas, dataset_name, tasks, config={}):
    device = q_feats.device
    q_feats = F.normalize(q_feats, p=2, dim=1)
    g_feats = F.normalize(g_feats, p=2, dim=1)
    sim_mat = torch.mm(q_feats, g_feats.t())
    
    q_vec = vectorize_metadata(q_metas, dataset_name, config, device)
    g_vec = vectorize_metadata(g_metas, dataset_name, config, device)
    
    mat_id, comps = compute_match_matrices(q_vec, g_vec, dataset_name)
    
    # 🔥🔥🔥 逻辑定义部分 🔥🔥🔥
    
    if 'CASIA-B' in dataset_name:
        # GT Change Flags (Ref vs Tar)
        gt_cond_changed = q_vec['gt_cond_changed'][:, None]
        gt_view_grp_changed = q_vec['gt_view_grp_changed'][:, None] 
        
        # Gallery Matches (Gal vs Tar)
        ret_cond_match = comps['ret_cond_match']     # Attribute 严格对应
        ret_view_grp_match = comps['ret_view_grp_match'] # View Group 对上
        
        # === Strict (Coarse-View, Strict-Attr) ===
        # 逻辑：View Group 必须对上 + Attribute 必须对上
        mask_strict = mat_id & ret_view_grp_match & ret_cond_match
        
        # === Soft (Conditional Match) ===
        # 1. View: 只要 View Group 变了，Gallery Group 就要对上 (Match)；没变不 Care
        match_view_soft = (~gt_view_grp_changed) | ret_view_grp_match
        
        # 2. Attribute: 只要 Attribute 变了，Gallery Attribute 就要对上 (Match)；没变不 Care
        # 注意：这里改成了 ret_cond_match，而不是之前的 ret_changed
        match_cond_soft = (~gt_cond_changed) | ret_cond_match
        
        mask_soft = mat_id & match_cond_soft & match_view_soft

    elif 'CCPG' in dataset_name:
        # CCPG 逻辑保持不变 (基于 Changed 状态)
        ret_bag_match = comps['bag_match']
        ret_cloth_changed = comps['ret_cloth_changed']
        ret_view_changed = comps['ret_view_changed']
        
        gt_cloth_changed = q_vec['gt_cloth_changed'][:, None]
        gt_view_changed = q_vec['gt_view_changed'][:, None]
        gt_bag_changed = q_vec['gt_bag_changed'][:, None] 
        
        match_cloth_strict = (gt_cloth_changed == ret_cloth_changed)
        match_view_strict = (gt_view_changed == ret_view_changed)
        match_bag_strict = ret_bag_match 
        mask_strict = mat_id & match_bag_strict & match_cloth_strict & match_view_strict
        
        match_cloth_soft = (~gt_cloth_changed) | ret_cloth_changed
        match_view_soft = (~gt_view_changed) | ret_view_changed
        match_bag_soft = (~gt_bag_changed) | ret_bag_match
        
        mask_soft = mat_id & match_bag_soft & match_cloth_soft & match_view_soft
        
    else:
        mask_strict = mat_id
        mask_soft = mat_id
        
    mask_id = mat_id
    
    # 5. 分任务统计
    final_output = {}
    unique_tasks = sorted(list(set(tasks)))
    if "Overall" not in unique_tasks: unique_tasks.append("Overall")
    task_array = np.array(tasks)
    
    for task_name in unique_tasks:
        if task_name == "Overall":
            indices = torch.arange(len(tasks), device=device)
        else:
            idx_list = np.where(task_array == task_name)[0]
            if len(idx_list) == 0: continue
            indices = torch.tensor(idx_list, device=device)
            
        sub_sim = sim_mat[indices]
        metrics = {'Count': len(indices)}
        metrics['Strict'] = compute_rank_k_ap(sub_sim, mask_strict[indices])
        metrics['Soft'] = compute_rank_k_ap(sub_sim, mask_soft[indices])
        metrics['ID'] = compute_rank_k_ap(sub_sim, mask_id[indices])
        final_output[task_name] = metrics
        
    return final_output