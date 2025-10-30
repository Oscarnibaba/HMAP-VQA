import warnings
from .vision.vit import VisionTransformer, interpolate_pos_embed
from .vision.mae_v2 import vit_mask_base_patch16
from .vision.path_embeds import PatchEmbeds
from .xbert import BertConfig, BertForMaskedLM
from transformers import BertTokenizer
import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
import torch.distributed as dist
from torchvision.ops import roi_align

warnings.filterwarnings("ignore")

class MUMCBASE(nn.Module):
    def __init__(self,
                 config=None,
                 use_contrastive_loss=False,
                 use_matching_loss=False,
                 use_mlm_loss=False,
                 ):
        super().__init__()
        self.visual_encoder, self.vision_width = build_visual_encoder(img_size=config['image_res'], url=config['vision_deit_path'])
        # self.visual_encoder, self.vision_width = build_visual_encoder(model='mask_vit', img_size=config['image_res'], url=config['vit_mae_pretrain_path'])
        self.tokenizer = BertTokenizer.from_pretrained(config['text_config'])
        self.text_encoder, self.text_width = build_text_encoder(text_config=config['text_config'], bert_config=config['bert_config'])

        self.embed_dim = config['embed_dim']
        self.pos_embed_dim = config.get('pos_embed_dim', self.vision_width)  # 768 or 512 etc
        self.x_pos_embed = nn.Sequential(
            nn.Linear(1, self.pos_embed_dim),
            nn.ReLU(),
            nn.Linear(self.pos_embed_dim, self.vision_width)
        )
        self.y_pos_embed = nn.Sequential(
            nn.Linear(1, self.pos_embed_dim),
            nn.ReLU(),
            nn.Linear(self.pos_embed_dim, self.vision_width)
        )
        self.region_proj = nn.Linear(self.vision_width, self.embed_dim)  # e.g. 768 → 256

        if use_contrastive_loss:
            self.embed_dim = config['embed_dim']
            self.vision_proj = nn.Linear(self.vision_width, self.embed_dim)
            self.text_proj = nn.Linear(self.text_width, self.embed_dim)
            self.temp = nn.Parameter(torch.ones([]) * config['temp'])
            # momentum related
            self.visual_encoder_m, _ = build_visual_encoder(model='mask_vit', img_size=config['image_res'], load_params=False)
            self.text_encoder_m, _ = build_text_encoder(text_config=config['text_config'], bert_config=config['bert_config'])
            self.vision_proj_m = nn.Linear(self.vision_width, self.embed_dim)
            self.text_proj_m = nn.Linear(self.text_width, self.embed_dim)
            self.queue_size = config['queue_size']
            self.momentum = config['momentum']

            self.model_pairs = [[self.visual_encoder, self.visual_encoder_m],
                                [self.vision_proj, self.vision_proj_m],
                                [self.text_encoder, self.text_encoder_m],
                                [self.text_proj, self.text_proj_m],
                                ]

            self.copy_params()

            # create the queue
            self.register_buffer("image_queue", torch.randn(self.embed_dim, self.queue_size))  # shape = [256, 65536]
            self.register_buffer("text_queue", torch.randn(self.embed_dim, self.queue_size))  # shape = [256, 65536]
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

            self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
            self.text_queue = nn.functional.normalize(self.text_queue, dim=0)

        if use_matching_loss:
            # self.itm_head = nn.Linear(self.text_width, 2)                           # one MLP layer
            self.itm_head = build_itm(input_dim=self.text_width, output_dim=2)   # two MLP layers

        if use_mlm_loss:
            self.mlm_probability = config['mlm_probability']


    def get_vision_embeds(self, image, mask_ratio=0.0):
        if mask_ratio == 0.0:
            image_embeds = self.visual_encoder(image)
        else:
            image_embeds = self.visual_encoder(image, mask_ratio=mask_ratio)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)
        return image_embeds, image_atts, image_feat

    def get_momentum_vision_embeds(self, image, mask_ratio=0.0):
        if mask_ratio == 0.0:
            image_embeds_m = self.visual_encoder_m(image)
        else:
            image_embeds_m = self.visual_encoder_m(image, mask_ratio=mask_ratio)
        image_feat_m = F.normalize(self.vision_proj_m(image_embeds_m[:, 0, :]), dim=-1)
        return image_embeds_m, image_feat_m

    def get_text_embeds(self, text):
        text_output = self.text_encoder.bert(text.input_ids,
                                             attention_mask=text.attention_mask,
                                             return_dict=True,
                                             mode='text')
        text_embeds = text_output.last_hidden_state
        text_feat = F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)
        return text_embeds, text_feat

    def get_momentum_text_embeds(self, text):
        text_output_m = self.text_encoder_m.bert(text.input_ids,
                                                 attention_mask=text.attention_mask,
                                                 return_dict=True,
                                                 mode='text')
        text_feat_m = F.normalize(self.text_proj_m(text_output_m.last_hidden_state[:, 0, :]), dim=-1)
        return text_feat_m

    def get_features(self, image_embeds=None, text_embeds=None):
        if image_embeds is None:
            return F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)
        elif text_embeds is None:
            return F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)
        else:
            return F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1),\
                   F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)

    def get_contrastive_loss(self, image_feat, text_feat, image_feat_m, text_feat_m, alpha=0):
        # get momentum features
        with torch.no_grad():
            self._momentum_update()
            image_feat_all = torch.cat([image_feat_m.t(), self.image_queue.clone().detach()], dim=1)
            text_feat_all = torch.cat([text_feat_m.t(), self.text_queue.clone().detach()], dim=1)
            sim_i2t_m = image_feat_m @ text_feat_all / self.temp
            sim_t2i_m = text_feat_m @ image_feat_all / self.temp

            sim_targets = torch.zeros(sim_i2t_m.size()).to(image_feat.device)
            sim_targets.fill_diagonal_(1)

            sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
            sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets

        sim_i2t = image_feat @ text_feat_all / self.temp
        sim_t2i = text_feat @ image_feat_all / self.temp

        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1).mean()

        loss_ita = (loss_i2t + loss_t2i) / 2

        self._dequeue_and_enqueue(image_feat_m, text_feat_m)
        return loss_ita

    def get_matching_loss(self, image_embeds, image_atts, image_feat, text_embeds, text_atts, text_feat):
        bs = image_embeds.size(0)
        output_pos = self.text_encoder.bert(encoder_embeds=text_embeds,
                                            attention_mask=text_atts,
                                            encoder_hidden_states=image_embeds,
                                            encoder_attention_mask=image_atts,
                                            return_dict=True,
                                            mode='fusion',
                                            )
        with torch.no_grad():
            sim_i2t = image_feat @ text_feat.t() / self.temp
            sim_t2i = text_feat @ image_feat.t() / self.temp

            weights_i2t = F.softmax(sim_i2t[:, :bs], dim=1) + 1e-4
            weights_t2i = F.softmax(sim_t2i[:, :bs], dim=1) + 1e-4

            weights_i2t.fill_diagonal_(0)
            weights_t2i.fill_diagonal_(0)

        # select a negative image for each text
        image_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_embeds_neg.append(image_embeds[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

        # select a negative text for each image
        text_embeds_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_embeds_neg.append(text_embeds[neg_idx])
            text_atts_neg.append(text_atts[neg_idx])
        text_embeds_neg = torch.stack(text_embeds_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        text_embeds_all = torch.cat([text_embeds, text_embeds_neg], dim=0)
        text_atts_all = torch.cat([text_atts, text_atts_neg], dim=0)

        image_embeds_all = torch.cat([image_embeds_neg, image_embeds], dim=0)
        image_atts_all = torch.cat([image_atts, image_atts], dim=0)

        output_neg = self.text_encoder.bert(encoder_embeds=text_embeds_all,
                                            attention_mask=text_atts_all,
                                            encoder_hidden_states=image_embeds_all,
                                            encoder_attention_mask=image_atts_all,
                                            return_dict=True,
                                            mode='fusion',
                                            )

        vl_embeddings = torch.cat([output_pos.last_hidden_state[:, 0, :], output_neg.last_hidden_state[:, 0, :]], dim=0)
        vl_output = self.itm_head(vl_embeddings)  # self.itm_head = nn.Linear(text_width, 2)
        itm_labels = torch.cat([torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)], dim=0).to(image_embeds.device)
        loss_itm = F.cross_entropy(vl_output, itm_labels)
        return loss_itm

    def get_mlm_loss(self, text, image_embeds, image_atts, image_embeds_m, alpha=0):
        input_ids = text.input_ids.clone()
        labels = input_ids.clone()
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        input_ids, labels = self.mask(input_ids, self.text_encoder.config.vocab_size, image_embeds.device, targets=labels,
                                      probability_matrix=probability_matrix)

        with torch.no_grad():
            logits_m = self.text_encoder_m(input_ids,
                                           attention_mask=text.attention_mask,
                                           encoder_hidden_states=image_embeds_m,
                                           encoder_attention_mask=image_atts,
                                           return_dict=True,
                                           return_logits=True,
                                           )
        mlm_output = self.text_encoder(input_ids,
                                       attention_mask=text.attention_mask,
                                       encoder_hidden_states=image_embeds,
                                       encoder_attention_mask=image_atts,
                                       return_dict=True,
                                       labels=labels,
                                       soft_labels=F.softmax(logits_m, dim=-1),
                                       alpha=alpha
                                       )
        return mlm_output.loss

    def get_region_itc_loss(self, region_visual_feats, region_text_feats, device):
        region_img_all = torch.cat(region_visual_feats, dim=0)
        region_txt_all = torch.cat(region_text_feats, dim=0)

        if region_img_all.shape[0] == 0:
            return torch.tensor(0.0, device=device)

        sim_r2t = region_img_all @ region_txt_all.T / self.temp
        sim_t2r = region_txt_all @ region_img_all.T / self.temp

        labels = torch.arange(sim_r2t.size(0), device=device)
        loss_r2t = F.cross_entropy(sim_r2t, labels)
        loss_t2r = F.cross_entropy(sim_t2r, labels)
        return (loss_r2t + loss_t2r) / 2

    def get_region_text_feats(self, region_texts, device):
        region_text_feats = []
        for desc_list in region_texts:
            if len(desc_list) == 0:
                region_text_feats.append(torch.empty(0, self.embed_dim, device=device))
                continue

            tokenized = self.tokenizer(desc_list, padding='longest', truncation=True, max_length=30,
                                       return_tensors="pt").to(device)
            text_out = self.text_encoder.bert(
                input_ids=tokenized.input_ids,
                attention_mask=tokenized.attention_mask,
                return_dict=True,
                mode='text'
            )
            feat = F.normalize(self.text_proj(text_out.last_hidden_state[:, 0, :]), dim=-1)
            region_text_feats.append(feat)
        return region_text_feats

    def get_region_feats(self, image, region_boxes, feature_map=None):
        """
        Extract region-level visual features (with spatial encoding) for a batch of images.

        Args:
            image (Tensor): Input images of shape [B, 3, H, W]
            region_boxes (List[Tensor]): List of [N_i, 4] tensors of bounding boxes for each image
            feature_map (Tensor, optional): Optional feature map from the visual encoder [B, C, H', W']

        Returns:
            List[Tensor]: List of region features per image, each of shape [N_i, embed_dim]
        """
        B, _, H, W = image.shape
        device = image.device

        # Extract feature map if not provided
        if feature_map is None:
            if hasattr(self.visual_encoder, 'get_intermediate_layers'):
                # 获取ViT的token序列 [B, N_tokens, D]
                tokens = self.visual_encoder.get_intermediate_layers(image)[-1]

                # 去掉CLS token，只保留patch tokens
                tokens_wo_cls = tokens[:, 1:, :]  # [B, N_patches, D]

                # 计算特征图尺寸 (假设是正方形)
                num_patches = tokens_wo_cls.shape[1]
                H_feat = W_feat = int(num_patches ** 0.5)

                # 重排为特征图格式 [B, D, H', W']
                feature_map = tokens_wo_cls.reshape(B, H_feat, W_feat, -1).permute(0, 3, 1, 2)
            else:
                raise AttributeError(
                    "Visual encoder must have 'get_intermediate_layers' method to extract ROI features.")

        # 获取特征图尺寸
        _, C_feat, H_feat, W_feat = feature_map.shape

        # Accumulate regions
        all_boxes = []
        batch_indices = []
        for b, boxes in enumerate(region_boxes):
            if boxes.numel() == 0:
                continue
            if boxes.shape[-1] != 4:
                raise ValueError(f"Boxes must be of shape [N_i, 4], got {boxes.shape}")

            # 将归一化坐标转换为特征图坐标
            # 输入boxes是[0,1]归一化坐标，需要转换为特征图坐标
            boxes_feat = boxes.clone()
            boxes_feat[:, [0, 2]] *= W_feat  # x坐标
            boxes_feat[:, [1, 3]] *= H_feat  # y坐标

            # 确保坐标在有效范围内
            boxes_feat[:, [0, 2]] = torch.clamp(boxes_feat[:, [0, 2]], 0, W_feat)
            boxes_feat[:, [1, 3]] = torch.clamp(boxes_feat[:, [1, 3]], 0, H_feat)

            all_boxes.append(boxes_feat)
            batch_indices.append(torch.full((boxes.size(0),), b, dtype=torch.long, device=device))

        if not all_boxes:
            return [torch.empty(0, self.embed_dim, device=device) for _ in range(B)]

        boxes = torch.cat(all_boxes, dim=0)  # [R, 4]
        batch_indices = torch.cat(batch_indices, dim=0)  # [R]

        # 构建rois: [batch_idx, x1, y1, x2, y2]
        rois = torch.cat([batch_indices.unsqueeze(1).float(), boxes], dim=1)  # [R, 5]

        # ROI Align (spatial_scale=1.0因为我们已经转换到特征图坐标)
        roi_feats = roi_align(feature_map, rois, output_size=(1, 1), spatial_scale=1.0)  # [R, C, 1, 1]
        roi_feats = roi_feats.squeeze(-1).squeeze(-1)  # [R, C]

        # === Positional encoding ===
        # 使用原始归一化坐标计算位置编码
        # 重新获取原始归一化坐标用于位置编码
        boxes_norm_list = []
        for b, boxes in enumerate(region_boxes):
            if boxes.numel() > 0:
                boxes_norm_list.append(boxes)
        if boxes_norm_list:
            boxes_norm = torch.cat(boxes_norm_list, dim=0)  # 原始归一化坐标
            x_center = (boxes_norm[:, 0] + boxes_norm[:, 2]) / 2  # [R] 归一化中心x
            y_center = (boxes_norm[:, 1] + boxes_norm[:, 3]) / 2  # [R] 归一化中心y
        else:
            x_center = torch.empty(0, device=device)
            y_center = torch.empty(0, device=device)

        x_embed = self.x_pos_embed(x_center.unsqueeze(1))  # [R, D]
        y_embed = self.y_pos_embed(y_center.unsqueeze(1))  # [R, D]

        # Combine visual and positional features
        fused_feats = roi_feats + x_embed + y_embed  # [R, D]

        # Project to embedding dim and normalize
        fused_feats = self.region_proj(fused_feats)  # [R, embed_dim]
        fused_feats = F.normalize(fused_feats, dim=-1)  # [R, embed_dim]

        # Split back to per-image tensors
        per_img_counts = [boxes.shape[0] for boxes in region_boxes]
        split_feats = []
        offset = 0
        for count in per_img_counts:
            if count == 0:
                split_feats.append(torch.empty(0, self.embed_dim, device=device))
            else:
                split_feats.append(fused_feats[offset:offset + count])
                offset += count
        return split_feats  # List[Tensor[N_i, embed_dim]]

    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat):
        # gather keys before updating queue
        if dist.is_available() and dist.is_initialized():
            image_feats = concat_all_gather(image_feat)
            text_feats = concat_all_gather(text_feat)
        else:
            image_feats = image_feat
            text_feats = text_feat
        # image_feats = concat_all_gather(image_feat)
        # text_feats = concat_all_gather(text_feat)

        batch_size = image_feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr

    def mask(self, input_ids, vocab_size, device, targets=None, masked_indices=None, probability_matrix=None):
        if masked_indices is None:
            # 使用语义感知的掩码策略
            masked_indices = self.semantic_aware_mask_indices(input_ids, probability_matrix)

        masked_indices[input_ids == self.tokenizer.pad_token_id] = False
        masked_indices[input_ids == self.tokenizer.cls_token_id] = False

        if targets is not None:
            targets[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long).to(device)
        input_ids[indices_random] = random_words[indices_random]
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged

        if targets is not None:
            return input_ids, targets
        else:
            return input_ids

    def semantic_aware_mask_indices(self, input_ids, probability_matrix):
        """基于词汇重要性的语义感知掩码"""
        if not hasattr(self, 'vocab_importance'):
            # 如果没有加载词汇表，使用原始随机掩码
            return torch.bernoulli(probability_matrix).bool()

        batch_size, seq_len = input_ids.shape
        masked_indices = torch.zeros_like(input_ids, dtype=torch.bool)

        # 计算基础掩码概率（保持与原始策略一致）
        base_mask_prob = probability_matrix[0, 0].item() if probability_matrix.numel() > 0 else 0.15

        for b in range(batch_size):
            for i in range(seq_len):
                token_id = input_ids[b, i].item()

                # 跳过特殊token
                if token_id in [self.tokenizer.pad_token_id, self.tokenizer.cls_token_id, self.tokenizer.sep_token_id]:
                    continue

                # 将token ID转换为文本
                try:
                    word = self.tokenizer.decode([token_id]).strip()
                    if not word or word.startswith('##'):
                        continue
                except:
                    continue

                # 获取词汇重要性
                word_info = self.get_word_importance(word)
                category = word_info['category']

                # 基于基础概率调整掩码概率
                if category == 'high_importance':
                    mask_prob = min(0.9, base_mask_prob * 2.0)  # 高重要性词汇，最多90%
                elif category == 'medium_importance':
                    mask_prob = min(0.7, base_mask_prob * 1.5)  # 中等重要性词汇，最多70%
                else:
                    mask_prob = max(0.05, base_mask_prob * 0.5)  # 低重要性词汇，最少5%

                # 随机决定是否掩码
                if torch.rand(1).item() < mask_prob:
                    masked_indices[b, i] = True

        # 确保掩码数量在合理范围内（基于原始概率矩阵）
        total_tokens = (input_ids != self.tokenizer.pad_token_id).sum()
        expected_masks = int(total_tokens * base_mask_prob)
        actual_masks = masked_indices.sum()

        # 如果掩码太多，随机减少一些
        if actual_masks > expected_masks * 1.5:
            excess_masks = actual_masks - expected_masks
            mask_positions = torch.where(masked_indices)
            if len(mask_positions[0]) > 0:
                indices_to_remove = torch.randperm(len(mask_positions[0]))[:excess_masks]
                for idx in indices_to_remove:
                    b, i = mask_positions[0][idx], mask_positions[1][idx]
                    masked_indices[b, i] = False

        return masked_indices

    def get_word_importance(self, word):
        """获取词汇的重要性信息"""
        if not hasattr(self, 'vocab_importance'):
            return {'category': 'low_importance'}

        word_lower = word.lower()
        return self.vocab_importance.get(word_lower, {'category': 'low_importance'})

    def load_vocabulary(self, vocab_path):
        """加载词汇表"""
        import json
        try:
            with open(vocab_path, 'r', encoding='utf-8') as f:
                vocabulary = json.load(f)

            self.vocab_importance = {}

            # 高重要性词汇
            for word, stats in vocabulary['high_importance'].items():
                self.vocab_importance[word.lower()] = {
                    'importance': stats['importance'],
                    'category': 'high_importance',
                    'count': stats['count']
                }

            # 中等重要性词汇
            for word, stats in vocabulary['medium_importance'].items():
                self.vocab_importance[word.lower()] = {
                    'importance': stats['importance'],
                    'category': 'medium_importance',
                    'count': stats['count']
                }

            # 低重要性词汇
            for word, stats in vocabulary['low_importance'].items():
                self.vocab_importance[word.lower()] = {
                    'importance': stats['importance'],
                    'category': 'low_importance',
                    'count': stats['count']
                }

            print(f"词汇表加载完成: {len(self.vocab_importance)} 个词汇")
            return True
        except Exception as e:
            print(f"词汇表加载失败: {e}")
            return False


def build_visual_encoder(model='vit', img_size=256, load_params=True, url=None):

    vision_width = 768
    if model == 'vit':
        visual_encoder = VisionTransformer(img_size=img_size, patch_size=16, embed_dim=vision_width, depth=12,
                                           num_heads=12, mlp_ratio=4,
                                           qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
    elif model == 'mask_vit':
        visual_encoder = vit_mask_base_patch16(img_size=img_size)
    elif model == 'path_embeds':
        visual_encoder = PatchEmbeds(img_size=img_size)
        load_params = False
        url = None


    if load_params:
        if url is not None:
            # download from https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth
            state_dict = torch.load(url, map_location='cpu')["model"]
        else:
            state_dict = torch.hub.load_state_dict_from_url(
                                            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                                            map_location="cpu", check_hash=True)["model"]
        pos_embed_reshaped = interpolate_pos_embed(state_dict['pos_embed'], visual_encoder)
        state_dict['pos_embed'] = pos_embed_reshaped
        msg = visual_encoder.load_state_dict(state_dict, strict=False)
        print('missing_keys = {}, unexpected_keys = {}'.format(msg.missing_keys, msg.unexpected_keys))

    return visual_encoder, vision_width

def build_text_encoder(text_config='bert-base-uncased', bert_config='configs/config_bert.json'):
    bert_conf = BertConfig.from_json_file(bert_config)
    text_encoder = BertForMaskedLM.from_pretrained(text_config, config=bert_conf)
    text_width = text_encoder.config.hidden_size
    return text_encoder, text_width

def build_itm(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, input_dim * 2),
        nn.LayerNorm(input_dim * 2),
        nn.GELU(),
        nn.Linear(input_dim * 2, output_dim)
    )


def load_checkpoint(model, url):
    checkpoint = torch.load(url, map_location='cpu')
    state_dict = checkpoint['model']

    state_dict['visual_encoder.pos_embed'] = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'], model.visual_encoder)
    if 'visual_encoder_m.pos_embed' in model.state_dict().keys():
        state_dict['visual_encoder_m.pos_embed'] = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'], model.visual_encoder_m)

    for key in model.state_dict().keys():
        if key in state_dict.keys():
            if state_dict[key].shape != model.state_dict()[key].shape:
                del state_dict[key]

    msg = model.load_state_dict(state_dict, strict=False)
    print('load checkpoint from %s' % url)
    return model, msg

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output