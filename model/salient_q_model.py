
import math
import types
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Optional, Tuple, List, Dict, Any
from scipy.optimize import linear_sum_assignment
from transformers import AutoConfig
from peft import LoraConfig, get_peft_model, TaskType
import gc


def box_iou(boxes1, boxes2):
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    inter_x1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
    inter_y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    inter_x2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    inter_y2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])

    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    union_area = area1[:, None] + area2[None, :] - inter_area

    return inter_area / (union_area + 1e-7)


def generalized_box_iou(boxes1, boxes2):
    boxes1 = torch.clamp(boxes1, 0, 1)
    boxes2 = torch.clamp(boxes2, 0, 1)

    inter_x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
    inter_y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
    inter_x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
    inter_y2 = torch.min(boxes1[:, 3], boxes2[:, 3])

    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union_area = area1 + area2 - inter_area + 1e-7

    iou = inter_area / union_area

    enclose_x1 = torch.min(boxes1[:, 0], boxes2[:, 0])
    enclose_y1 = torch.min(boxes1[:, 1], boxes2[:, 1])
    enclose_x2 = torch.max(boxes1[:, 2], boxes2[:, 2])
    enclose_y2 = torch.max(boxes1[:, 3], boxes2[:, 3])

    enclose_area = (enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1) + 1e-7

    return iou - (enclose_area - union_area) / enclose_area


class SaliencyModule(nn.Module):

    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim
        self.saliency_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim // 4, feature_dim // 16),
            nn.GELU(),
            nn.Linear(feature_dim // 16, 1),
            nn.Sigmoid()
        )
        self.gating_alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, features):
        saliency_scores = self.saliency_head(features)
        gated_features = features + self.gating_alpha * features * saliency_scores
        return gated_features, saliency_scores


class PositionEmbeddingSine2D(nn.Module):
    def __init__(self, feature_dim, temperature):
        super().__init__()
        self.feature_dim = feature_dim
        self.temperature = temperature
        self.dim_per_axis = feature_dim // 2

    def forward(self, grid_h, grid_w, device, dtype):
        y_coords = torch.linspace(0, 1, grid_h, device=device)
        x_coords = torch.linspace(0, 1, grid_w, device=device)

        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        norm_coords = torch.stack([xx.flatten(), yy.flatten()], dim=-1)

        dim_t = torch.arange(0, self.dim_per_axis, 2, device=device, dtype=torch.float32)
        div_term = self.temperature ** (dim_t / self.dim_per_axis)

        pos_x = xx.unsqueeze(-1) / div_term
        pos_x = torch.stack([torch.sin(pos_x * math.pi), torch.cos(pos_x * math.pi)], dim=-1)
        pos_x = pos_x.flatten(-2)

        pos_y = yy.unsqueeze(-1) / div_term
        pos_y = torch.stack([torch.sin(pos_y * math.pi), torch.cos(pos_y * math.pi)], dim=-1)
        pos_y = pos_y.flatten(-2)

        pos_embed = torch.cat([pos_x, pos_y], dim=-1)

        if dtype is not None:
            pos_embed = pos_embed.to(dtype=dtype)
            norm_coords = norm_coords.to(dtype=dtype)

        return pos_embed.flatten(0, 1), norm_coords


class PositionAwareScout(nn.Module):

    def __init__(
        self,feature_dim,num_scouts,num_heads,saliency_temperature,num_classes,use_learned_pos):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_scouts = num_scouts
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.saliency_temp = nn.Parameter(torch.tensor(saliency_temperature))
        self.num_classes = num_classes
        self.use_learned_pos = use_learned_pos
        self.scout_queries = nn.Embedding(num_scouts, feature_dim)
        nn.init.normal_(self.scout_queries.weight, std=0.02)

        self.query_pos = nn.Embedding(num_scouts, feature_dim)
        nn.init.normal_(self.query_pos.weight, std=0.02)

        if use_learned_pos:
            self.pos_encoder = PositionEncoderMLP(feature_dim)
        else:
            self.pos_encoder = PositionEmbeddingSine2D(feature_dim)

        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        self.out_proj = nn.Linear(feature_dim, feature_dim)

        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 4, feature_dim),
            nn.Dropout(0.1)
        )

        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)

        self.bbox_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, feature_dim // 4),
            nn.ReLU(),
            nn.Linear(feature_dim // 4, 4),
            nn.Sigmoid()
        )

        self.class_head = nn.Linear(feature_dim, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
        nn.init.normal_(self.bbox_head[-2].weight, std=0.001)
        nn.init.zeros_(self.bbox_head[-2].bias)

    def forward(
        self,visual_features, saliency_scores,grid_h, grid_w):
        B, T, D = visual_features.shape
        device = visual_features.device

        if saliency_scores.dim() == 3:
            saliency_scores = saliency_scores.squeeze(-1)

        pos_embed, norm_coords = self.pos_encoder(grid_h, grid_w, device)
        pos_embed = pos_embed.to(dtype=visual_features.dtype)
        norm_coords = norm_coords.to(dtype=visual_features.dtype)

        expected_tokens = grid_h * grid_w
        if T != expected_tokens:
            if T < expected_tokens:
                pos_embed = pos_embed[:T]
                norm_coords = norm_coords[:T]
            else:
                pad_size = T - expected_tokens
                pos_embed = torch.cat([pos_embed, pos_embed[-1:].expand(pad_size, -1)], dim=0)
                norm_coords = torch.cat([norm_coords, norm_coords[-1:].expand(pad_size, -1)], dim=0)

        queries = self.scout_queries.weight.unsqueeze(0).expand(B, -1, -1)
        query_pos = self.query_pos.weight.unsqueeze(0).expand(B, -1, -1)

        Q = self.q_proj(queries + query_pos)
        K = self.k_proj(visual_features + pos_embed.unsqueeze(0))
        V = self.v_proj(visual_features)

        N = self.num_scouts
        Q = Q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        saliency_bias = saliency_scores.view(B, 1, 1, T) * self.saliency_temp
        attn_scores = attn_scores + saliency_bias

        attn_weights = F.softmax(attn_scores, dim=-1)

        attended = torch.matmul(attn_weights, V)
        attended = attended.transpose(1, 2).contiguous().view(B, N, D)
        attended = self.out_proj(attended)

        scout_tokens = self.norm1(queries + attended)
        scout_tokens = self.norm2(scout_tokens + self.ffn(scout_tokens))

        pred_boxes = self.bbox_head(scout_tokens)
        pred_logits = self.class_head(scout_tokens)

        avg_attn = attn_weights.mean(dim=1)
        norm_coords_expanded = norm_coords.unsqueeze(0).expand(B, -1, -1)
        attended_coords = torch.einsum('bnt,btc->bnc', avg_attn, norm_coords_expanded)

        return pred_boxes, pred_logits, attended_coords, attn_weights


class SalientQModelV2(Qwen3VLForConditionalGeneration):

    def __init__(
        self,config,num_scouts,saliency_weight,detection_weight,coord_weight,saliency_temperature,use_learned_pos):
        super().__init__(config)
        self.feature_dim = config.text_config.hidden_size
        self.num_scouts = num_scouts
        self.use_learned_pos = use_learned_pos

        self.saliency_weight = saliency_weight
        self.detection_weight = detection_weight
        self.coord_weight = coord_weight

        self.saliency_module = SaliencyModule(feature_dim=self.feature_dim)
        self.scout_module = PositionAwareScout(
            feature_dim=self.feature_dim,
            num_scouts=num_scouts,
            num_heads=8,
            saliency_temperature=saliency_temperature,
            num_classes=9,
            use_learned_pos=use_learned_pos,
        )

        self._aux_saliency_scores = None
        self._aux_pred_boxes = None
        self._aux_pred_logits = None
        self._aux_attended_coords = None
        self._aux_attn_weights = None
        self._aux_grid_dims = None

        self._patch_inner_model()

    def _patch_inner_model(self):
        outer_model = self
        original_get_image_features = self.model.get_image_features

        def patched_get_image_features(inner_self,pixel_values,image_grid_thw):
            pixel_values = pixel_values.type(inner_self.visual.dtype)
            image_embeds, deepstack_image_embeds = inner_self.visual(pixel_values, grid_thw=image_grid_thw)

            spatial_merge = inner_self.visual.spatial_merge_size
            split_sizes = (image_grid_thw.prod(-1) // (spatial_merge ** 2)).tolist()

            grid_dims = []
            for i in range(image_grid_thw.shape[0]):
                t, h, w = image_grid_thw[i].tolist()
                grid_dims.append((h // spatial_merge, w // spatial_merge))
            outer_model._aux_grid_dims = grid_dims

            gated_embeds, saliency_scores = outer_model.saliency_module(image_embeds)
            outer_model._aux_saliency_scores = saliency_scores

            gated_per_image = torch.split(gated_embeds, split_sizes)
            saliency_per_image = torch.split(saliency_scores.squeeze(-1), split_sizes)

            all_pred_boxes = []
            all_pred_logits = []
            all_attended_coords = []
            all_attn_weights = []

            for idx, (img_features, img_saliency) in enumerate(
                zip(gated_per_image, saliency_per_image)
            ):
                grid_h, grid_w = grid_dims[idx]
                pred_boxes, pred_logits, attended_coords, attn_weights = (
                    outer_model.scout_module(
                        visual_features=img_features.unsqueeze(0),
                        saliency_scores=img_saliency.unsqueeze(0),
                        grid_h=grid_h,
                        grid_w=grid_w,
                    )
                )
                all_pred_boxes.append(pred_boxes.squeeze(0))
                all_pred_logits.append(pred_logits.squeeze(0))
                all_attended_coords.append(attended_coords.squeeze(0))
                all_attn_weights.append(attn_weights.squeeze(0))

            outer_model._aux_pred_boxes = torch.stack(all_pred_boxes) if all_pred_boxes else None
            outer_model._aux_pred_logits = torch.stack(all_pred_logits) if all_pred_logits else None
            outer_model._aux_attended_coords = torch.stack(all_attended_coords) if all_attended_coords else None
            outer_model._aux_attn_weights = all_attn_weights

            gated_per_image_list = list(torch.split(gated_embeds, split_sizes))
            return gated_per_image_list, deepstack_image_embeds

        self.model.get_image_features = types.MethodType(patched_get_image_features, self.model)

    def compute_saliency_loss(self,saliency_masks,image_grid_thw):
        if self._aux_saliency_scores is None:
            return None

        pred_saliency = self._aux_saliency_scores.squeeze(-1)
        target_saliency = saliency_masks.view(-1)

        min_len = min(len(pred_saliency), len(target_saliency))
        pred_saliency = pred_saliency[:min_len]
        target_saliency = target_saliency[:min_len]

        pred_f32 = pred_saliency.float()
        target_f32 = target_saliency.float()

        intersection = (pred_f32 * target_f32).sum()
        union = pred_f32.sum() + target_f32.sum()
        dice_loss = 1.0 - (2.0 * intersection + 1e-6) / (union + 1e-6)

        return dice_loss

    def compute_detection_loss(self, detection_boxes,detection_labels):
        if self._aux_pred_boxes is None or self._aux_pred_logits is None:
            return None

        pred_boxes = self._aux_pred_boxes
        pred_logits = self._aux_pred_logits
        batch_size = detection_boxes.shape[0]
        device = pred_boxes.device

        total_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
        num_valid_batches = 0

        for i in range(batch_size):
            valid_mask = detection_labels[i] > 0
            if not valid_mask.any():
                continue

            target_boxes_i = detection_boxes[i][valid_mask]
            target_labels_i = detection_labels[i][valid_mask]
            pred_boxes_i = pred_boxes[i]
            pred_logits_i = pred_logits[i]

            pred_boxes_f32 = pred_boxes_i.float()
            target_boxes_f32 = target_boxes_i.float()

            l1_cost = torch.cdist(pred_boxes_f32, target_boxes_f32, p=1)
            pred_probs = pred_logits_i.float().softmax(dim=-1)
            class_cost = -pred_probs[:, target_labels_i.long()]
            cost_matrix = l1_cost + class_cost

            try:
                cost_np = cost_matrix.detach().cpu().numpy()
                row_ind, col_ind = linear_sum_assignment(cost_np)
                row_ind = torch.tensor(row_ind, device=device)
                col_ind = torch.tensor(col_ind, device=device)
            except Exception:
                n = min(pred_boxes_i.shape[0], target_boxes_i.shape[0])
                row_ind = torch.arange(n, device=device)
                col_ind = torch.arange(n, device=device)

            matched_pred_boxes = pred_boxes_i[row_ind]
            matched_target_boxes = target_boxes_i[col_ind]
            matched_pred_logits = pred_logits_i[row_ind]
            matched_target_labels = target_labels_i[col_ind]

            l1_loss = F.l1_loss(matched_pred_boxes.float(), matched_target_boxes.float(), reduction='mean')
            giou_loss = self._compute_giou_loss(matched_pred_boxes.float(), matched_target_boxes.float())
            class_loss = F.cross_entropy(matched_pred_logits.float(), matched_target_labels.long())

            total_loss = total_loss + l1_loss + giou_loss + class_loss
            num_valid_batches += 1

        if num_valid_batches > 0:
            return total_loss / num_valid_batches
        return torch.tensor(0.0, device=device, requires_grad=True)

    def compute_coordinate_loss(self,detection_boxes,detection_labels):
        if self._aux_attended_coords is None or self._aux_pred_boxes is None:
            return None

        attended_coords = self._aux_attended_coords
        pred_boxes = self._aux_pred_boxes
        batch_size = detection_boxes.shape[0]
        device = pred_boxes.device

        total_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
        num_valid_batches = 0

        for i in range(batch_size):
            valid_mask = detection_labels[i] > 0
            if not valid_mask.any():
                continue

            target_boxes_i = detection_boxes[i][valid_mask]
            target_centers = torch.stack([
                (target_boxes_i[:, 0] + target_boxes_i[:, 2]) / 2,
                (target_boxes_i[:, 1] + target_boxes_i[:, 3]) / 2,
            ], dim=-1)

            pred_boxes_f32 = pred_boxes[i].float()
            target_boxes_f32 = target_boxes_i.float()

            l1_cost = torch.cdist(pred_boxes_f32, target_boxes_f32, p=1)

            try:
                cost_np = l1_cost.detach().cpu().numpy()
                row_ind, col_ind = linear_sum_assignment(cost_np)
                row_ind = torch.tensor(row_ind, device=device)
                col_ind = torch.tensor(col_ind, device=device)
            except Exception:
                n = min(len(pred_boxes[i]), len(target_boxes_i))
                row_ind = torch.arange(n, device=device)
                col_ind = torch.arange(n, device=device)

            matched_attended = attended_coords[i].float()[row_ind]
            matched_centers = target_centers.float()[col_ind]
            coord_loss = F.mse_loss(matched_attended, matched_centers)

            total_loss = total_loss + coord_loss
            num_valid_batches += 1

        if num_valid_batches > 0:
            return total_loss / num_valid_batches
        return torch.tensor(0.0, device=device, requires_grad=True)

    def _compute_giou_loss(self,pred_boxes,target_boxes) ->:
        pred_boxes = torch.clamp(pred_boxes.float(), 0, 1)
        target_boxes = torch.clamp(target_boxes.float(), 0, 1)

        inter_x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
        inter_y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
        inter_x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
        inter_y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])

        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

        pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
        target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])
        union_area = pred_area + target_area - inter_area + 1e-7

        iou = inter_area / union_area

        enclose_x1 = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
        enclose_y1 = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
        enclose_x2 = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
        enclose_y2 = torch.max(pred_boxes[:, 3], target_boxes[:, 3])
        enclose_area = (enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1) + 1e-7

        giou = iou - (enclose_area - union_area) / enclose_area
        return (1.0 - giou).mean()

    def forward(
        self,
        input_ids,
        attention_mask,
        position_ids,
        past_key_values,
        inputs_embeds,
        labels,
        pixel_values,
        pixel_values_videos,
        image_grid_thw,
        video_grid_thw,
        cache_position,
        saliency_masks,
        detection_boxes,
        detection_labels,
        target_boxes:,
        target_labels,
        **kwargs,
    ):
        if detection_boxes is None and target_boxes is not None:
            detection_boxes = target_boxes
        if detection_labels is None and target_labels is not None:
            detection_labels = target_labels

        self._aux_saliency_scores = None
        self._aux_pred_boxes = None
        self._aux_pred_logits = None
        self._aux_attended_coords = None
        self._aux_attn_weights = None

        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            cache_position=cache_position,
            **kwargs,
        )

        if self.training and labels is not None:
            total_loss = outputs.loss if outputs.loss is not None else torch.tensor(0.0, device=self.device)

            saliency_loss = torch.tensor(0.0, device=self.device)
            if saliency_masks is not None and self._aux_saliency_scores is not None:
                saliency_loss = self.compute_saliency_loss(saliency_masks, image_grid_thw)
                total_loss = total_loss + self.saliency_weight * saliency_loss

            detection_loss = torch.tensor(0.0, device=self.device)
            coord_loss = torch.tensor(0.0, device=self.device)

            if detection_boxes is not None and detection_labels is not None:
                has_valid_targets = (detection_labels > 0).any()
                if has_valid_targets and self._aux_pred_boxes is not None:
                    detection_loss = self.compute_detection_loss(detection_boxes, detection_labels)
                    total_loss = total_loss + self.detection_weight * detection_loss

                    coord_loss = self.compute_coordinate_loss(detection_boxes, detection_labels)
                    total_loss = total_loss + self.coord_weight * coord_loss

            outputs.loss = total_loss
            outputs.saliency_loss = saliency_loss
            outputs.detection_loss = detection_loss
            outputs.coord_loss = coord_loss

        return outputs


def create_salient_q_v2_model(base_model_path: str = "Qwen/Qwen3-VL-8B-Instruct",num_scouts, saliency_weight,detection_weight,coord_weight,saliency_temperature,
    use_learned_pos, use_lora,lora_r,lora_alpha):
    config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
    model = SalientQModelV2(
        config,
        num_scouts=num_scouts,
        saliency_weight=saliency_weight,
        detection_weight=detection_weight,
        coord_weight=coord_weight,
        saliency_temperature=saliency_temperature,
        use_learned_pos=use_learned_pos,
    )

    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    missing, unexpected = model.load_state_dict(base_model.state_dict(), strict=False)

    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    if use_lora:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.1,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        )
        model = get_peft_model(model, lora_config)

    for name, param in model.named_parameters():
        if any(x in name for x in ['saliency_module', 'scout_module']):
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    return model
