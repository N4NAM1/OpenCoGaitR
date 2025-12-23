import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel
from ..base_model import BaseModel

class TGDS(nn.Module):
    """
    🔥 创新模块：Text-Guided Dynamic Sampling (文本引导的时序动态采样聚合)
    功能：利用文本指令作为 Query，从视频帧序列中动态采样并聚合最相关的步态特征。
    """
    def __init__(self, clip_feature_dim: int, projection_dim: int, num_heads: int = 8):
        super(TGDS, self).__init__()

        # 使用 MultiheadAttention 实现文本引导的时序筛选
        self.attn = nn.MultiheadAttention(embed_dim=projection_dim, num_heads=num_heads, batch_first=True)

        self.norm = nn.LayerNorm(clip_feature_dim)

    def forward(self, image_seq, text_features):
        """
        Args:
            image_seq: [B, T, D] CLIP 视频帧特征序列
            text_features: [B, D] CLIP 文本指令特征
        Returns:
            aggregated_feat: [B, D] 聚合后的时序特征
        """
        # 1. 投影到注意力空间
        # Query 来自文本 (B, 1, Proj_D)，Key/Value 来自图像序列 (B, T, Proj_D)
        query = text_features.unsqueeze(1)  # [B, 1, D]
        key = image_seq                     # [B, T, D]
        value = image_seq                   # [B, T, D]

        # 2. 🔥 跨模态时序注意力计算
        # attn_output 捕捉了与文本指令最匹配的帧组合
        attn_output, _ = self.attn(query, key, value)
        
        # 3. Add&Norm（使用均值作为 Base）
        res = attn_output.squeeze(1)
        output = self.norm(res + image_seq.mean(dim=1))
        
        return output

class Combiner(nn.Module):
    """ 保持原有优秀的门控融合逻辑 """
    def __init__(self, clip_feature_dim: int, projection_dim: int, hidden_dim: int):
        super(Combiner, self).__init__()
        self.text_projection_layer = nn.Linear(clip_feature_dim, projection_dim)
        self.image_projection_layer = nn.Linear(clip_feature_dim, projection_dim)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.combiner_layer = nn.Linear(projection_dim * 2, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, clip_feature_dim) 
        self.dropout3 = nn.Dropout(0.5)
        self.dynamic_scalar = nn.Sequential(
            nn.Linear(projection_dim * 2, hidden_dim), 
            nn.ReLU(), nn.Dropout(0.5), nn.Linear(hidden_dim, 1), nn.Sigmoid()
        )

    def forward(self, image_features, text_features):
        text_projected = self.dropout1(F.relu(self.text_projection_layer(text_features)))
        image_projected = self.dropout2(F.relu(self.image_projection_layer(image_features)))
        raw_combined = torch.cat((text_projected, image_projected), -1)
        combined_features = self.dropout3(F.relu(self.combiner_layer(raw_combined)))
        mlp_out = self.output_layer(combined_features)
        sigma = self.dynamic_scalar(raw_combined)
        output = mlp_out + sigma * text_features + (1 - sigma) * image_features
        return F.normalize(output, p=2, dim=-1, eps=1e-6)

class GaitTCI(BaseModel):
    def __init__(self, cfgs, training=True):
        super().__init__(cfgs, training)
        
    def build_network(self, model_cfg):
        model_id = model_cfg.get('backbone', "openai/clip-vit-base-patch32")
        print(f"🏗️ Building Gait-TCI Model: {model_id}")
        
        self.clip = CLIPModel.from_pretrained(model_id)
        for param in self.clip.parameters():
            param.requires_grad = False
            
        self.feature_dim = self.clip.projection_dim 
        proj_dim = model_cfg.get('projection_dim', 512)
        hidden_dim = model_cfg.get('hidden_dim', 2048)
        
        # 🔥 实例化两个核心模块
        self.tgds = TGDS(self.feature_dim, proj_dim)
        self.combiner = Combiner(self.feature_dim, proj_dim, hidden_dim)
        
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)
        
        if self.training:
            loss_cfg = self.cfgs.get('loss_cfg', {})
            self.loss_alpha = loss_cfg.get('alpha', 0.5)
            self.loss_fn = nn.CrossEntropyLoss()

    def _encode_image(self, img_tensor):
        """ 🔥 修改点：不再在内部做 Max Pooling，保留序列信息 """
        B, T, C, H, W = img_tensor.shape
        img_flat = img_tensor.view(-1, C, H, W)
        with torch.no_grad():
            feat = self.clip.get_image_features(img_flat)
        # 返回 [B, T, D]
        feat = feat.view(B, T, -1)
        return F.normalize(feat, p=2, dim=-1, eps=1e-6)

    def _encode_text(self, input_ids, attention_mask):
        with torch.no_grad():
            feat = self.clip.get_text_features(input_ids, attention_mask)
        return F.normalize(feat, p=2, dim=-1, eps=1e-6)

    def forward(self, inputs):
        if self.training: return self.forward_train(inputs)
        else: return self.forward_test(inputs)
        
    def forward_train(self, inputs):
        ref, tar, txt_ids, txt_mask, inv_ids, inv_mask = inputs
        
        # 1. 提取序列特征
        ref_seq = self._encode_image(ref) # [B, T, D]
        tar_seq = self._encode_image(tar) # [B, T, D]
        txt_feat = self._encode_text(txt_ids, txt_mask) # [B, D]
        inv_feat = self._encode_text(inv_ids, inv_mask) # [B, D]
        
        # 2. 🔥 第一阶段：文本引导聚合 (TGDS)
        # 此时 ref_feat 变成了受 text 启发聚合后的单向量
        ref_agg = self.tgds(ref_seq, txt_feat)
        tar_agg = self.tgds(tar_seq, inv_feat)
        
        # 3. 🔥 第二阶段：Combiner 融合
        q_fwd = self.combiner(ref_agg, txt_feat)
        q_inv = self.combiner(tar_agg, inv_feat)
        
        # 4. 计算 Loss
        logit_scale = self.logit_scale.exp()
        labels = torch.arange(len(q_fwd), device=self.device)
        
        # 注意：tar_feat 在对比时使用其序列的均值或 Max 作为 Target 表达，
        # 也可以直接用 tar_agg。这里为了稳健，使用 tar_seq 的 mean。
        tar_target = tar_seq.mean(dim=1)
        logits_fwd = (q_fwd @ tar_target.T) * logit_scale
        loss_fwd = self.loss_fn(logits_fwd, labels)
        
        # 逆向 Loss (Cycle Consistency)
        # 目标是 q_inv 经过变换后能回到 ref 的原始特征
        ref_target = ref_seq.mean(dim=1)
        loss_inv = 1.0 - F.cosine_similarity(q_inv, ref_target).mean()
        
        total_loss = loss_fwd + self.loss_alpha * loss_inv
        
        return {'loss': total_loss, 'acc_loss': loss_fwd, 'inv_loss': loss_inv}

    def forward_test(self, inputs):
            ref_list, tar_list, txt_ids, txt_mask, tasks, metas = inputs
            # txt_feat 是整个 batch 的特征: [B, D] (例如 [4, 512])
            txt_feat = self._encode_text(txt_ids, txt_mask)
            
            ref_feats = []
            tar_feats = []
            
            # 🔥 修改点：使用 enumerate 获取索引 i，以便取出对应的文本特征
            for i, (r, t) in enumerate(zip(ref_list, tar_list)):
                # 1. 准备视频序列 [1, T, D]
                r = r.unsqueeze(0).to(self.device)
                t = t.unsqueeze(0).to(self.device)
                r_seq = self._encode_image(r) 
                t_seq = self._encode_image(t)
                
                # 2. 🔥 修正：取出当前样本对应的第 i 个文本特征，并保持维度 [1, D]
                curr_txt_feat = txt_feat[i].unsqueeze(0) 
                
                # 3. 现在 Query(Batch=1) 和 Key(Batch=1) 维度匹配了
                r_agg = self.tgds(r_seq, curr_txt_feat)
                
                ref_feats.append(r_agg)
                tar_feats.append(t_seq.mean(dim=1)) # Target 端继续用均值
            
            # 拼接回 Batch 维度 [B, D]
            ref_feats = torch.cat(ref_feats, dim=0)
            tar_feats = torch.cat(tar_feats, dim=0)
            
            # 最终 Combiner 融合 (这里 txt_feat 是完整的 [B, D]，ref_feats 也是 [B, D]，可以一起计算)
            q_feat = self.combiner(ref_feats, txt_feat)
            
            return {
                "query_feat": q_feat,
                "tar_feat": tar_feats,
                "tasks": tasks,
                "metas": metas
            }