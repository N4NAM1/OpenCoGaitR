import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

# === 定义 Dataset (修正版：解决 Processed 包含 oc 的 bug) ===
class OcclusionOnlyDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_list = []
        
        print(f"🔍 Scanning 'oc' (occlusion) images in {root_dir}...")
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    full_path = os.path.join(root, file)
                    
                    # === 核心修正逻辑 ===
                    # 1. 将路径拆分为各个部分的列表
                    # 例如: ['root', 'autodl-tmp', 'SUSTech1K_Processed', 'RGB', '0001', '00-nm', '000', '000.jpg']
                    path_parts = full_path.split(os.sep)
                    
                    is_real_oc = False
                    for part in path_parts:
                        part_lower = part.lower()
                        
                        # 2. 必须排除根目录名字里的 'processed' 干扰
                        if 'processed' in part_lower:
                            continue
                            
                        # 3. 在剩下的部分里找 'oc'
                        # 比如 '01-oc', 'occlusion', 'type-oc' 都会被匹配
                        if 'oc' in part_lower:
                            is_real_oc = True
                            break
                    
                    if is_real_oc:
                        self.image_list.append(full_path)
                        
        if len(self.image_list) == 0:
            print("⚠️ Warning: No images containing 'oc' were found! Please check your dataset structure.")
        else:
            print(f"✅ Found {len(self.image_list)} 'oc' images (Filtered out 'Processed' false positives).")

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        img_path = self.image_list[index]
        image = Image.open(img_path).convert("RGB")
        return image, img_path

def get_palette(num_cls):
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device}")

    model_name = "mattmdjaga/segformer_b2_clothes"
    print(f"📥 Loading model: {model_name} ...")
    
    processor = SegformerImageProcessor.from_pretrained(model_name)
    model = SegformerForSemanticSegmentation.from_pretrained(model_name)
    model.to(device)
    model.eval()

    dataset = OcclusionOnlyDataset(args.input_dir)
    
    if len(dataset) == 0:
        return

    palette = get_palette(18)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=lambda x: list(zip(*x)))

    with torch.no_grad():
        for batch_imgs, batch_paths in tqdm(loader, desc="Generating Parsing"):
            inputs = processor(images=list(batch_imgs), return_tensors="pt").to(device)
            outputs = model(**inputs)
            logits = outputs.logits
            
            for i in range(len(batch_imgs)):
                img_path = batch_paths[i]
                orig_w, orig_h = batch_imgs[i].size
                
                logit = logits[i].unsqueeze(0)
                upsampled_logit = F.interpolate(
                    logit, 
                    size=(orig_h, orig_w), 
                    mode="bilinear", 
                    align_corners=False
                )
                
                pred_seg = upsampled_logit.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
                
                rel_path = os.path.relpath(img_path, args.input_dir)
                save_path = os.path.join(args.output_dir, rel_path)
                save_path = os.path.splitext(save_path)[0] + '.png'
                
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                
                mask_img = Image.fromarray(pred_seg)
                mask_img.putpalette(palette)
                mask_img.save(save_path)

    print(f"✅ Parsing generation complete! Saved to {args.output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True, help='Root of RGB images')
    parser.add_argument('--output_dir', type=str, required=True, help='Root to save Parsing maps')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    args = parser.parse_args()
    
    main(args)