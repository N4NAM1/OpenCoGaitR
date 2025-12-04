import torch
import numpy as np
import torch.nn.functional as F

from utils import is_tensor


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



# =========================================================
# 1. 向量化预处理工具
# =========================================================

def vectorize_metadata(meta_list, dataset_name, config, device):
    """
    将元数据列表转换为 GPU Tensor 字典。
    """
    N = len(meta_list)
    vec_data = {}
    
    # 1. 基础属性: SID
    raw_sids = [m['sid'] for m in meta_list]
    _, sid_indices = np.unique(raw_sids, return_inverse=True)
    vec_data['sid'] = torch.tensor(sid_indices, device=device)

    if 'CASIA-B' in dataset_name:
        # --- CASIA-B ---
        # 视角映射
        view_map = {}
        view_groups = config.get('view_groups', {}) 
        for grp_idx, (grp_name, views) in enumerate(view_groups.items()):
            for v in views: view_map[v] = grp_idx
        
        tar_views = [view_map.get(m['tar_view'], -1) for m in meta_list]
        vec_data['tar_view'] = torch.tensor(tar_views, device=device)
        
        # 状态映射
        raw_conds = [m['tar_cond'] for m in meta_list]
        _, cond_indices = np.unique(raw_conds, return_inverse=True)
        vec_data['tar_cond'] = torch.tensor(cond_indices, device=device)

    elif 'CCPG' in dataset_name:
        # --- CCPG ---
        def parse(cond_str):
            parts = cond_str.split('_')
            u, d, bag = 0, 0, 0
            for p in parts:
                if p.startswith('U'): u = int(p[1:])
                elif p.startswith('D'): d = int(p[1:])
                elif p == 'BG': bag = 1
            return u, d, bag

        ref_u, ref_d, ref_bag = np.zeros(N), np.zeros(N), np.zeros(N)
        tar_u, tar_d, tar_bag = np.zeros(N), np.zeros(N), np.zeros(N)
        ref_views, tar_views = [], []

        for i, m in enumerate(meta_list):
            ru, rd, rb = parse(m['ref_cond'])
            ref_u[i], ref_d[i], ref_bag[i] = ru, rd, rb
            ref_views.append(m['ref_view'])
            
            tu, td, tb = parse(m['tar_cond'])
            tar_u[i], tar_d[i], tar_bag[i] = tu, td, tb
            tar_views.append(m['tar_view'])

        # 转 Tensor
        vec_data['ref_u'] = torch.tensor(ref_u, device=device)
        vec_data['ref_d'] = torch.tensor(ref_d, device=device)
        vec_data['tar_bag'] = torch.tensor(tar_bag, device=device) # [N]
        
        # 视角处理
        _, rv_ids = np.unique(ref_views, return_inverse=True)
        _, tv_ids = np.unique(tar_views, return_inverse=True)
        vec_data['ref_view'] = torch.tensor(rv_ids, device=device)
        vec_data['tar_view'] = torch.tensor(tv_ids, device=device)
        
        # 预计算 Query 自身的属性变化 (GT)
        # 换装：GT相对于Ref是否换了
        vec_data['gt_cloth_changed'] = (vec_data['ref_u'] != torch.tensor(tar_u, device=device)) | \
                                       (vec_data['ref_d'] != torch.tensor(tar_d, device=device))
        # 换视角：GT相对于Ref是否换了
        vec_data['gt_view_changed'] = (vec_data['ref_view'] != vec_data['tar_view'])

        # Gallery 属性 (Gallery 是 Target Image)
        vec_data['gal_u'] = torch.tensor(tar_u, device=device)
        vec_data['gal_d'] = torch.tensor(tar_d, device=device)

    return vec_data

# =========================================================
# 2. 核心匹配矩阵计算 (Components)
# =========================================================

def compute_match_matrices(q_vec, g_vec, dataset_name):
    """
    返回基础匹配组件字典 (components)
    """
    # ID 匹配 [N, M]
    mat_id = (q_vec['sid'][:, None] == g_vec['sid'][None, :])

    if 'CASIA-B' in dataset_name:
        mat_view = (q_vec['tar_view'][:, None] == g_vec['tar_view'][None, :])
        mat_cond = (q_vec['tar_cond'][:, None] == g_vec['tar_cond'][None, :])
        return mat_id, {'view': mat_view, 'cond': mat_cond}

    elif 'CCPG' in dataset_name:
        # 1. 背包 (Strict Match)
        mat_bag = (q_vec['tar_bag'][:, None] == g_vec['tar_bag'][None, :])
        
        # 2. 换装 (Relative Change Match)
        # Gallery(Retrieved) 相对于 Query(Ref) 的变化
        ret_u_changed = (g_vec['gal_u'][None, :] != q_vec['ref_u'][:, None]) 
        ret_d_changed = (g_vec['gal_d'][None, :] != q_vec['ref_d'][:, None])
        ret_cloth_changed = ret_u_changed | ret_d_changed
        # 逻辑：GT 变了 == Ret 变了
        mat_cloth = (q_vec['gt_cloth_changed'][:, None] == ret_cloth_changed)
        
        # 3. 视角 (Relative Change Match)
        ret_view_changed = (g_vec['tar_view'][None, :] != q_vec['ref_view'][:, None])
        gt_view_req_change = q_vec['gt_view_changed'][:, None]
        # 逻辑：GT 变了 == Ret 变了
        mat_view = (gt_view_req_change == ret_view_changed)
        
        return mat_id, {'bag': mat_bag, 'cloth': mat_cloth, 'view': mat_view}

    return mat_id, {}

# =========================================================
# 3. 指标计算与主流程 (Logic Composition)
# =========================================================

def compute_rank_k_ap(sim_mat, gt_mask, k_list):
    """ 计算 R@K 和 mAP """
    # 1. 排序
    scores, indices = torch.sort(sim_mat, dim=1, descending=True)
    # 2. 重排 GT
    gts = torch.gather(gt_mask.float(), 1, indices)
    
    results = {}
    
    # R@K
    for k in k_list:
        if k > gts.shape[1]:
            results[f'R{k}'] = 100.0
        else:
            # 只要前K个里有一个对，就算对 (Recall / Hit Rate)
            hit_k = (gts[:, :k].sum(dim=1) > 0).float().mean().item() * 100
            results[f'R{k}'] = hit_k
    
    # mAP
    cumsum = torch.cumsum(gts, dim=1)
    ranks = torch.arange(1, gts.size(1) + 1, device=gts.device).unsqueeze(0)
    precision = cumsum / ranks
    
    ap_sum = (precision * gts).sum(dim=1)
    num_rel = gts.sum(dim=1)
    mask = num_rel > 0
    ap = torch.zeros_like(ap_sum)
    ap[mask] = ap_sum[mask] / num_rel[mask]
    results['mAP'] = ap.mean().item() * 100
    
    return results

def compute_gaitcir_metrics(q_feats, g_feats, q_metas, g_metas, dataset_name, tasks, config):
    """
    主计算函数
    """
    device = q_feats.device
    k_list = config.get('k_list', [1, 5, 10])
    
    # 1. 向量化
    q_vec = vectorize_metadata(q_metas, dataset_name, config, device)
    g_vec = vectorize_metadata(g_metas, dataset_name, config, device)
    
    # 2. 相似度
    q_feats = F.normalize(q_feats, p=2, dim=1)
    g_feats = F.normalize(g_feats, p=2, dim=1)
    sim_mat = torch.mm(q_feats, g_feats.t())
    
    # 3. 获取组件矩阵
    mat_id, comps = compute_match_matrices(q_vec, g_vec, dataset_name)
    
    # 4. 🔥 构造全局 Mask (Vectorized Logic Composition)
    # 这一步是实现 "Overall Soft = 混合加权" 的关键
    
    if 'CASIA-B' in dataset_name:
        # Strict: ID + View + Cond
        mask_strict = mat_id & comps['view'] & comps['cond']
        # Soft: ID + Cond (忽略 View)
        mask_soft = mat_id & comps['cond']
        
    elif 'CCPG' in dataset_name:
        # 组件引用
        m_bag, m_cloth, m_view = comps['bag'], comps['cloth'], comps['view']
        
        # Strict: ID + Bag + Cloth + View (全对)
        mask_strict = mat_id & m_bag & m_cloth & m_view
        
        # Soft: 根据每个样本的 Task 类型动态决定
        # 将 tasks 列表转为 GPU Tensor 索引以便广播
        task_array = np.array(tasks)
        
        # 创建 Task 对应的布尔索引 [N, 1]
        is_attr = torch.tensor(task_array == 'attribute_change', device=device).unsqueeze(1)
        is_view = torch.tensor(task_array == 'viewpoint_change', device=device).unsqueeze(1)
        is_comp = torch.tensor(task_array == 'composite_change', device=device).unsqueeze(1)
        
        # 定义每种 Task 的 Soft 标准
        # 1. Attribute Task: 忽略 View (ID + Bag + Cloth)
        soft_attr = mat_id & m_bag & m_cloth
        
        # 2. Viewpoint Task: 忽略 Cloth/Bag (ID + View) -> 通常 Viewpoint 任务确实不关注衣服
        # 但如果你的 Viewpoint 任务也隐含了“衣服不变”，则应该加上 m_cloth
        # 按照之前的逻辑: "忽略属性错误" -> ID + View
        soft_view = mat_id & m_view
        
        # 3. Composite Task: Soft = Strict (全对)
        soft_comp = mask_strict
        
        # 4. 混合生成全局 Soft Mask
        # 逻辑: (是Attr任务 & 用Attr标准) | (是View任务 & 用View标准) ...
        # 对于不属于这三类的 (如 Overall 里的其他)，默认用 Strict
        is_other = ~(is_attr | is_view | is_comp)
        
        mask_soft = (is_attr & soft_attr) | \
                    (is_view & soft_view) | \
                    (is_comp & soft_comp) | \
                    (is_other & mask_strict)
                    
    else:
        # 默认
        mask_strict = mat_id
        mask_soft = mat_id

    # 5. ID Mask
    mask_id_only = mat_id

    # 6. 分组统计
    final_output = {}
    unique_tasks = sorted(list(set(tasks)))
    if "Overall" not in unique_tasks: unique_tasks.append("Overall")
    
    task_array = np.array(tasks)
    
    for task_name in unique_tasks:
        # 获取索引
        if task_name == "Overall":
            indices = torch.arange(len(tasks), device=device)
        else:
            # np.where 返回 tuple
            indices = torch.tensor(np.where(task_array == task_name)[0], device=device)
            
        if len(indices) == 0: continue
        
        # 切片
        sub_sim = sim_mat[indices]
        
        # 只需要切行 (Query 维度)，列 (Gallery) 保持全量
        sub_strict = mask_strict[indices]
        sub_soft = mask_soft[indices]     # 这里已经是混合好的正确 Soft Mask
        sub_id = mask_id_only[indices]
        
        metrics = {'Count': len(indices)}
        
        # 计算三套指标
        metrics['Strict'] = compute_rank_k_ap(sub_sim, sub_strict, k_list)
        metrics['Soft'] = compute_rank_k_ap(sub_sim, sub_soft, k_list)
        metrics['ID'] = compute_rank_k_ap(sub_sim, sub_id, k_list)
        
        final_output[task_name] = metrics
        
    return final_output

