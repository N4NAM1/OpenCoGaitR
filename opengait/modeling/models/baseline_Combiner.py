import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel

# 引入我们修改过的基类
from ..base_model import BaseModel

class Combiner(nn.Module):
    """
    Advanced Combiner: Projection + Dynamic Residual
    """
    def __init__(self, clip_feature_dim: int, projection_dim: int, hidden_dim: int):
        super(Combiner, self).__init__()
        # 1. 特征投影层
        self.text_projection_layer = nn.Linear(clip_feature_dim, projection_dim)
        self.image_projection_layer = nn.Linear(clip_feature_dim, projection_dim)

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        
        # 2. 融合层
        self.combiner_layer = nn.Linear(projection_dim * 2, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, clip_feature_dim) 

        self.dropout3 = nn.Dropout(0.5)
        
        # 3. 动态标量预测
        self.dynamic_scalar = nn.Sequential(
            nn.Linear(projection_dim * 2, hidden_dim), 
            nn.ReLU(), 
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, image_features, text_features):
        # 投影
        text_projected = self.dropout1(F.relu(self.text_projection_layer(text_features)))
        image_projected = self.dropout2(F.relu(self.image_projection_layer(image_features)))

        # 拼接
        raw_combined = torch.cat((text_projected, image_projected), -1)
        
        # 路径 1: MLP
        combined_features = self.dropout3(F.relu(self.combiner_layer(raw_combined)))
        mlp_out = self.output_layer(combined_features)
        
        # 路径 2: 动态残差
        sigma = self.dynamic_scalar(raw_combined)
        
        # 融合
        output = mlp_out + sigma * text_features + (1 - sigma) * image_features
        
        # 归一化
        return F.normalize(output, p=2, dim=-1)

class GaitCIRModel(BaseModel):
    """
    GaitCIR 主模型
    """
    def __init__(self, cfgs, training=True):
        super().__init__(cfgs, training)
        
    def build_network(self, model_cfg):
        # 1. Backbone
        model_id = model_cfg.get('backbone', "openai/clip-vit-base-patch32")
        print(f"🏗️ Building Backbone: {model_id}")
        
        self.clip = CLIPModel.from_pretrained(model_id)
        for param in self.clip.parameters():
            param.requires_grad = False
            
        self.feature_dim = self.clip.projection_dim 
        
        # 2. Combiner (参数可配置)
        proj_dim = model_cfg.get('projection_dim', 512)
        hidden_dim = model_cfg.get('hidden_dim', 2048)
        print(f"🏗️ Building Combiner (Proj={proj_dim}, Hidden={hidden_dim})")
        
        self.combiner = Combiner(self.feature_dim, proj_dim, hidden_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)
        
        # 3. Loss
        if self.training:
            self.loss_fn = nn.CrossEntropyLoss()
            self.alpha = model_cfg.get('loss_alpha', 0.5)

    def forward(self, inputs):
        if self.training:
            return self.forward_train(inputs)
        else:
            return self.forward_test(inputs)

    def _encode_image(self, img_tensor):
        B, T, C, H, W = img_tensor.shape
        img_flat = img_tensor.view(-1, C, H, W)
        with torch.no_grad():
            feat = self.clip.get_image_features(img_flat)
        feat = feat.view(B, T, -1)
        agg_feat = feat.max(dim=1)[0]
        return F.normalize(agg_feat, p=2, dim=-1)

    def _encode_text(self, input_ids, attention_mask):
        with torch.no_grad():
            feat = self.clip.get_text_features(input_ids, attention_mask)
        return F.normalize(feat, p=2, dim=-1)

    def forward_train(self, inputs):
        ref, tar, txt_ids, txt_mask, inv_ids, inv_mask = inputs
        
        ref_feat = self._encode_image(ref)
        tar_feat = self._encode_image(tar)
        txt_feat = self._encode_text(txt_ids, txt_mask)
        inv_feat = self._encode_text(inv_ids, inv_mask)
        
        q_fwd = self.combiner(ref_feat, txt_feat)
        q_inv = self.combiner(tar_feat, inv_feat)
        
        logit_scale = self.logit_scale.exp()
        labels = torch.arange(len(q_fwd), device=self.device)
        
        logits_fwd = (q_fwd @ tar_feat.T) * logit_scale
        loss_fwd = self.loss_fn(logits_fwd, labels)
        
        # Inverse Loss (正则项)
        loss_inv = 1.0 - F.cosine_similarity(q_inv, ref_feat).mean()
        
        total_loss = loss_fwd + self.alpha * loss_inv
        
        return {'loss': total_loss, 'acc_loss': loss_fwd, 'inv_loss': loss_inv}

    def forward_test(self, inputs):
        ref_list, tar_list, txt_ids, txt_mask, tasks, metas = inputs
        
        txt_feat = self._encode_text(txt_ids, txt_mask)
        
        ref_feats = []
        tar_feats = []
        for r, t in zip(ref_list, tar_list):
            r = r.unsqueeze(0).to(self.device)
            t = t.unsqueeze(0).to(self.device)
            ref_feats.append(self._encode_image(r))
            tar_feats.append(self._encode_image(t))
            
        ref_feats = torch.cat(ref_feats, dim=0)
        tar_feats = torch.cat(tar_feats, dim=0)
        
        q_feat = self.combiner(ref_feats, txt_feat)
        
        return {
            "query_feat": q_feat,
            "tar_feat": tar_feats,
            "tasks": tasks,
            "metas": metas
        }