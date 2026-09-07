import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import clip
from template import *
from torchvision import transforms as T
from clip.model import *

REAL_RELATION_PROMPT = "This is a real face. The face is the same as the anchor."
SPOOF_RELATION_PROMPT = "This is a spoof face. The face is different from the anchor."

def attribute_balanced_ce_loss(sm_matrix_annotation, sm_matrix_pseudo, annotation_label, pseudo_label):
    sm_matrix_annotation = torch.softmax(sm_matrix_annotation, dim=1)
    sm_matrix_pseudo = torch.softmax(sm_matrix_pseudo, dim=1)
    xs_pos = sm_matrix_annotation[:, 1].reshape(-1)
    xs_neg = sm_matrix_annotation[:, 0].reshape(-1)
    annotation_label = annotation_label.reshape(-1)
    los_pos = annotation_label * torch.log(xs_pos)
    los_neg = (1 - annotation_label) * torch.log(xs_neg)
    annotation_loss = -(los_pos + los_neg).sum() / los_pos.shape[0]
    pseudo_loss = sm_matrix_pseudo.new_tensor(0.0)
    return annotation_loss, pseudo_loss

class MaskAwareVisionEncoder(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        self.conv1_alpha = nn.Conv2d(in_channels=1, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)
        self.bn1 = nn.BatchNorm2d(width)
        self.relu1 = nn.ReLU(inplace=True)

        self.transformer = Transformer(width, layers, heads)
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))


    def forward(self, x: torch.Tensor,mask: torch.Tensor):

        x = self.conv1(x)
        mask = self.conv1_alpha(mask)
        x = x + mask
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)

        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        mask = mask.reshape(mask.shape[0], mask.shape[1], -1)
        mask = mask.permute(0, 2, 1)
        mask = torch.cat([self.class_embedding.to(mask.dtype) + torch.zeros(mask.shape[0], 1, mask.shape[-1], dtype=mask.dtype, device=mask.device), mask], dim=1)


        return x,mask

class CLIPBackbone(nn.Module):
    def __init__(self,

                 input_resolution: int,
                 patch_size: int,
                 width: int,
                 layers: int,
                 heads: int,
                 output_dim: int,


                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int):

        super().__init__()

        self.context_length = context_length
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.text_projection = nn.Parameter(torch.empty(transformer_width, output_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.ln_final = LayerNorm(transformer_width)

        self.visual = MaskAwareVisionEncoder(
                input_resolution=input_resolution,
                patch_size=patch_size,
                width=width,
                layers=layers,
                heads=heads,
                output_dim=output_dim )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

    def forward(self, x, mask):
        return self.visual(x,mask)

    def text_encoder(self,text):

        x = self.token_embedding(text).to(torch.float32)
        x = x + self.positional_embedding.to(torch.float32)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).to(torch.float32)
        return x

    def build_attention_mask(self):


        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

class AnchorRelativeCrossAttention(nn.Module):
    def __init__(self,  d_model, nhead, num_layers, dim_feedforward=2048, dropout=0.1):
        super(AnchorRelativeCrossAttention,self).__init__()

        self.d_model = d_model
        self.head_dim = d_model // nhead
        self.num_heads = nhead

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout,batch_first = True)


        self.qkv_project_reference = nn.Linear(d_model, d_model * 3)
        self.qkv_project_inference = nn.Linear(d_model, d_model * 2)

        self.q_linear_reference = nn.ModuleList([nn.Linear(self.head_dim, self.head_dim) for _ in range(nhead)])
        self.k_linear_reference = nn.ModuleList([nn.Linear(self.head_dim, self.head_dim) for _ in range(nhead)])
        self.v_linear_reference = nn.ModuleList([nn.Linear(self.head_dim, self.head_dim) for _ in range(nhead)])

        self.q_linear_inference = nn.ModuleList([nn.Linear(self.head_dim, self.head_dim) for _ in range(nhead)])
        self.k_linear_inference = nn.ModuleList([nn.Linear(self.head_dim, self.head_dim) for _ in range(nhead)])
        self.out_linear  = nn.Linear(d_model, d_model)

        self.transformer_encoder_1 = nn.TransformerEncoder(encoder_layer, num_layers=num_layers-1)
        self.transformer_encoder_2 = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.layer_norm = nn.LayerNorm(d_model)

        tau_in_init = 0.1
        self.tau = nn.Parameter(torch.tensor(tau_in_init))

    def forward(self,reference,inference,reference_mask,inference_mask):

        query_r, key_r, value_r = self.qkv_project_reference(reference).chunk(3, dim=-1)
        query_i, key_i = self.qkv_project_inference(inference).chunk(2, dim=-1)

        batch_size = query_r.size(0)

        def reshape_heads(x):
            return x.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        query_r = reshape_heads(query_r)
        key_r = reshape_heads(key_r)
        value_r = reshape_heads(value_r)
        query_i = reshape_heads(query_i)
        key_i = reshape_heads(key_i)


        M = reference_mask @ inference_mask.permute(0,2,1)
        M = torch.where(M != 0, M.new_tensor(0.10), M.new_tensor(0.08))

        heads = []
        for i in range(self.num_heads):
            q = self.q_linear_reference[i](query_r[:, i, :, :]) - self.q_linear_inference[i](query_i[:, i, :, :])
            k = self.k_linear_reference[i](key_r[:, i, :, :]) - self.k_linear_inference[i](key_i[:, i, :, :])
            v = self.v_linear_reference[i](value_r[:, i, :, :])


            scores = torch.matmul(q, k.transpose(-2, -1)) *  M
            attn_weights = torch.softmax(scores, dim=-1)
            head = torch.matmul(attn_weights, v)
            heads.append(head)

        heads = torch.cat(heads, dim=-1)
        attn_output = self.out_linear(heads)
        attn_output = self.layer_norm(attn_output)
        attn_output_last = self.transformer_encoder_1(attn_output)
        attn_output = self.transformer_encoder_2(attn_output_last)
        return attn_output


class AnchorizedSpoofArtifactExtractor(nn.Module):
    def __init__(self,  d_model, nhead, num_layers, dim_feedforward=2048, dropout=0.1):
        super(AnchorizedSpoofArtifactExtractor,self).__init__()
        self.d_model = d_model
        self.head_dim = d_model // nhead
        self.num_heads = nhead

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout,batch_first = True)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(d_model)
        tau_in_init = 0.1
        self.tau = nn.Parameter(torch.tensor(tau_in_init))

        self.kv_project_reference = nn.Linear(d_model, d_model * 2)
        self.q_project_inference = nn.Linear(d_model, d_model)

        self.k_linear_reference = nn.ModuleList([nn.Linear(self.head_dim, self.head_dim) for _ in range(nhead)])
        self.v_linear_reference = nn.ModuleList([nn.Linear(self.head_dim, self.head_dim) for _ in range(nhead)])

        self.q_linear_inference = nn.ModuleList([nn.Linear(self.head_dim, self.head_dim) for _ in range(nhead)])
        self.out_linear  = nn.Linear(d_model, d_model)

    def forward(self,reference,inference,reference_mask,inference_mask):

        key_r, value_r = self.kv_project_reference(reference).chunk(2, dim=-1)
        query_i = self.q_project_inference(inference)

        batch_size = query_i.size(0)

        def reshape_heads(x):
            return x.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        key_r = reshape_heads(key_r)
        value_r = reshape_heads(value_r)
        query_i = reshape_heads(query_i)
        M = reference_mask @ inference_mask.permute(0,2,1)
        M = torch.where(M != 0, M.new_tensor(0.10), M.new_tensor(0.08))

        heads = []
        for i in range(self.num_heads):
            q = self.q_linear_inference[i](query_i[:, i, :, :])
            k = self.k_linear_reference[i](key_r[:, i, :, :])
            v = self.v_linear_reference[i](value_r[:, i, :, :])


            scores = torch.matmul(q, k.transpose(-2, -1)) *  M


            attn_weights = torch.softmax(scores, dim=-1)
            head = torch.matmul(attn_weights, v)
            heads.append(head)

        heads = torch.cat(heads, dim=-1)
        attn_output = self.out_linear(heads)
        attn_output = self.layer_norm(attn_output)
        attn_output = self.transformer_encoder(attn_output)
        return attn_output


class TRAM(nn.Module):
    def __init__(self, state_dict,cross_layer,low_level_layer=6):
        super().__init__()

        vit = "visual.proj" in state_dict
        if vit:

            vision_width = state_dict["visual.conv1.weight"].shape[0]
            vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
            vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
            grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
            image_resolution = vision_patch_size * grid_size

        embed_dim = state_dict["text_projection"].shape[1]
        context_length = state_dict["positional_embedding"].shape[0]
        vocab_size = state_dict["token_embedding.weight"].shape[0]
        transformer_width = state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))
        vision_heads = vision_width  // 64


        self.clip_backbone = CLIPBackbone(
                    input_resolution=image_resolution,
                    patch_size=vision_patch_size,
                    width=vision_width,
                    layers=vision_layers,
                    heads=vision_heads,
                    output_dim=embed_dim,
                    context_length = context_length,
                    vocab_size = vocab_size,
                    transformer_width = transformer_width,
                    transformer_heads = transformer_heads,
                    transformer_layers = transformer_layers
                        )

        for key in ["input_resolution", "context_length", "vocab_size"]:
            if key in state_dict:
                del state_dict[key]

        self.clip_backbone.load_state_dict(state_dict, strict=False)

        self.rca = AnchorRelativeCrossAttention(vision_width,transformer_heads,cross_layer-1)
        self.frpl = AnchorizedSpoofArtifactExtractor(vision_width,transformer_heads,low_level_layer-1)
        self.relation_projection =  nn.Sequential(
                        nn.Linear(in_features=768, out_features=2048),

                        nn.ReLU(),
                        nn.Linear(in_features=2048, out_features=1024),

                        nn.ReLU(),
                        nn.Linear(in_features=1024, out_features=768)
                                    )

        self.triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2, eps=1e-7)
        self.cls_loss = nn.CrossEntropyLoss()


        self.stage_1_num = 200
        self.stage_2_num = 0

        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    def _update_prototypes(self, cams: torch.Tensor, labels: torch.Tensor):
        with torch.no_grad():
            for cls in range(self.num_labels):
                mask = (labels == cls)
                if not mask.any():
                    continue
                cls_cams = cams[mask]
                batch_mean = cls_cams.mean(dim=0)
                if self.prototype_counts[cls] == 0:
                    self.gradcam_prototypes[cls] = batch_mean
                else:
                    self.gradcam_prototypes[cls] = (
                        self.proto_momentum * self.gradcam_prototypes[cls]
                        + (1.0 - self.proto_momentum) * batch_mean
                    )
                self.prototype_counts[cls] += mask.sum()


    def _gradcam_proto_loss(self, cams: torch.Tensor, labels: torch.Tensor, eps: float = 1e-6):
        if (self.prototype_counts == 0).all():
            return cams.new_tensor(0.0)

        proto = self.gradcam_prototypes[labels]
        B, C, H, W = cams.shape


        cams = F.relu(cams)
        proto = F.relu(proto)


        cams_sum = cams.sum(dim=(2, 3), keepdim=True)
        proto_sum = proto.sum(dim=(2, 3), keepdim=True)


        uniform = cams.new_full((1, 1, H, W), 1.0 / (H * W))

        cams = torch.where(
            cams_sum > 0,
            cams / (cams_sum + eps),
            uniform
        )
        proto = torch.where(
            proto_sum > 0,
            proto / (proto_sum + eps),
            uniform
        )


        cams = cams.clamp_min(eps)
        proto = proto.clamp_min(eps)


        cams = cams / cams.sum(dim=(2, 3), keepdim=True)
        proto = proto / proto.sum(dim=(2, 3), keepdim=True)


        m = 0.5 * (cams + proto)
        m = m.clamp_min(eps)
        m = m / m.sum(dim=(2, 3), keepdim=True)


        kl1 = F.kl_div(cams.log(), m, reduction='batchmean')
        kl2 = F.kl_div(proto.log(), m, reduction='batchmean')
        js = 0.5 * (kl1 + kl2)

        return js

    def _gradcam_from_tokens(self,
                            tokens: torch.Tensor,
                            logits: torch.Tensor,
                            proto_idx: torch.Tensor):
        B, T, C = tokens.shape
        H = W = self.grid_size

        assert logits.shape[0] == B, "logits batch size must match tokens"
        assert logits.shape[1] == 2, "expect binary logits [real, spoof]"
        assert proto_idx.shape[0] == B, "proto_idx length must match batch"


        coarse_idx = (proto_idx != 0).long()


        selected = logits.gather(1, coarse_idx.view(-1, 1))
        target = selected.sum()

        grads = torch.autograd.grad(
            outputs=target,
            inputs=tokens,
            retain_graph=True,
            create_graph=False,
            only_inputs=True,
            allow_unused=True,
        )[0]

        if grads is None:
            return None


        weights = grads.mean(dim=1, keepdim=True)
        cam_all = (tokens * weights).sum(dim=2)
        cam_all = F.relu(cam_all)


        if T == H * W + 1:
            cam_vec = cam_all[:, 1:]
        elif T == H * W:
            cam_vec = cam_all
        else:
            return None

        cams = cam_vec.view(B, 1, H, W)


        cams = cams - cams.amin(dim=(2, 3), keepdim=True)
        cams = cams / (cams.amax(dim=(2, 3), keepdim=True) + 1e-6)

        return cams
    def encode_text(self, text):

        text = clip.tokenize(text).cuda()


        text_token = self.clip_backbone.text_encoder(text)


        text_token = text_token[torch.arange(text_token.shape[0]), text.argmax(dim=-1)] @ self.clip_backbone.text_projection


        return text_token

    def feature(self,reference,inference,ref_mask,inf_mask):
        reference,ref_mask = self.clip_backbone(reference,ref_mask)
        inference,inf_mask = self.clip_backbone(inference,inf_mask)
        cls_feat =  self.rca(reference,inference,ref_mask,inf_mask)
        cls_feat = self.relation_projection(cls_feat[:, 0, :])
        cls_feat = self.clip_backbone.visual.ln_post(cls_feat)
        cls_feat = cls_feat @ self.clip_backbone.visual.proj
        cls_feat = cls_feat / cls_feat.norm(dim=-1, keepdim=True)

        image_features = self.frpl(reference,inference,ref_mask,inf_mask)
        image_features = self.clip_backbone.visual.ln_post(image_features[:, 0, :])
        image_features = image_features @ self.clip_backbone.visual.proj
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        fine_grained_template_diff = [SPOOF_RELATION_PROMPT, generate_indicator_template(0),generate_indicator_template(1),generate_indicator_template(2),generate_indicator_template(3)]
        fine_grained_template_same = [REAL_RELATION_PROMPT, generate_indicator_template(0,False),generate_indicator_template(1,False),generate_indicator_template(2,False),generate_indicator_template(3,False)]

        logit_scale = self.clip_backbone.logit_scale.exp()

        text_grained_features_diff = self.encode_text(fine_grained_template_diff)
        text_grained_features_diff = text_grained_features_diff / text_grained_features_diff.norm(dim=-1, keepdim=True)
        fg_sm_diff = logit_scale * image_features @ text_grained_features_diff.t()

        text_grained_features_same = self.encode_text(fine_grained_template_same)
        text_grained_features_same = text_grained_features_same / text_grained_features_same.norm(dim=-1, keepdim=True)
        fg_sm_same = logit_scale * image_features @ text_grained_features_same.t()

        hint = fg_sm_diff[:,1:]>fg_sm_same[:,1:]

        annotation = []
        for ind in range(hint.shape[0]):
            template_diff = SPOOF_RELATION_PROMPT
            template_same = REAL_RELATION_PROMPT
            hint_array = hint[ind].nonzero().flatten().detach().cpu().numpy()
            if len(hint_array) != 0:
                for hint_ind in hint_array:
                    template_diff += ' '+generate_indicator_template(hint_ind)


        return cls_feat, image_features,template_diff


    def triplet(self,input_tuple,input_mask,input_caption,input_indicators,iter_num):
            cls_loss = 0
            triplet_loss = 0
            txt_loss = 0
            annotation_loss = 0
            pseudo_loss = 0
            fg_hist = []
            multi_label_hist = []

            for i in range(len(input_tuple)):
                img_real_ref, img_fake_inf, img_real_inf = input_tuple[i]
                img_real_ref_mask, img_fake_inf_mask, img_real_inf_mask = input_mask[i]
                fake_inf_real_caption,fake_inf_fake_caption,real_inf_real_caption,real_inf_fake_caption = input_caption[i]
                indicators_positive,indicators_negative = input_indicators[i]


                real_inf_text =  torch.stack([self.encode_text(real_inf_real_caption), self.encode_text(real_inf_fake_caption)])
                fake_inf_text =  torch.stack([self.encode_text(fake_inf_real_caption), self.encode_text(fake_inf_fake_caption)])


                multi_label_real = []
                multi_label_fake = []
                for ind in range(img_real_ref.shape[0]):
                    temp = torch.stack([t[ind] for t in indicators_negative])
                    multi_label_fake.append(temp)
                    temp = torch.stack([t[ind] for t in indicators_positive])
                    multi_label_real.append(temp)

                multi_label_real = torch.stack(multi_label_real)
                multi_label_fake = torch.stack(multi_label_fake)
                multi_label = torch.cat([multi_label_real,multi_label_fake],dim=0)

                img_real_ref,mask_real_ref = self.clip_backbone(img_real_ref,img_real_ref_mask)
                img_fake_inf,mask_fake_inf = self.clip_backbone(img_fake_inf,img_fake_inf_mask)
                img_real_inf,mask_real_inf = self.clip_backbone(img_real_inf,img_real_inf_mask)


                img_real_diff = self.rca(img_real_ref,img_real_inf,mask_real_ref,mask_real_inf)
                img_fake_diff = self.rca(img_real_ref,img_fake_inf,mask_real_ref,mask_fake_inf)

                low_level_real_diff = self.frpl(img_real_ref,img_real_inf,mask_real_ref,mask_real_inf)
                low_level_fake_diff = self.frpl(img_real_ref,img_fake_inf,mask_real_ref,mask_real_inf)


                real_diff_cls_token = self.relation_projection(img_real_diff[:, 0, :])
                fake_diff_cls_token = self.relation_projection(img_fake_diff[:, 0, :])


                logit_scale = self.clip_backbone.logit_scale.exp()


                img_real_diff = self.clip_backbone.visual.ln_post(low_level_real_diff[:, 0, :])
                img_real_diff = img_real_diff @ self.clip_backbone.visual.proj
                img_fake_diff = self.clip_backbone.visual.ln_post(low_level_fake_diff[:, 0, :])
                img_fake_diff = img_fake_diff @ self.clip_backbone.visual.proj

                img_real_diff = img_real_diff / img_real_diff.norm(dim=-1, keepdim=True)
                img_fake_diff = img_fake_diff / img_fake_diff.norm(dim=-1, keepdim=True)
                img_diff = torch.concat([img_real_diff,img_fake_diff],dim=0)


                cls_real_diff = self.clip_backbone.visual.ln_post(real_diff_cls_token)
                cls_real_diff = cls_real_diff @ self.clip_backbone.visual.proj
                cls_fake_diff = self.clip_backbone.visual.ln_post(fake_diff_cls_token)
                cls_fake_diff = cls_fake_diff @ self.clip_backbone.visual.proj

                cls_real_diff = cls_real_diff / cls_real_diff.norm(dim=-1, keepdim=True)
                cls_fake_diff = cls_fake_diff / cls_fake_diff.norm(dim=-1, keepdim=True)
                cls_diff = torch.concat([cls_real_diff,cls_fake_diff],dim=0)


                fine_grained_template_diff = [SPOOF_RELATION_PROMPT, generate_indicator_template(0),generate_indicator_template(1),generate_indicator_template(2),generate_indicator_template(3)]
                fine_grained_template_same = [REAL_RELATION_PROMPT, generate_indicator_template(0,False),generate_indicator_template(1,False),generate_indicator_template(2,False),generate_indicator_template(3,False)]

                text_grained_features_diff = self.encode_text(fine_grained_template_diff)
                text_grained_features_diff = text_grained_features_diff / text_grained_features_diff.norm(dim=-1, keepdim=True)
                fg_sm_diff = logit_scale * img_diff @ text_grained_features_diff.t()

                text_grained_features_same = self.encode_text(fine_grained_template_same)
                text_grained_features_same = text_grained_features_same / text_grained_features_same.norm(dim=-1, keepdim=True)
                fg_sm_same = logit_scale * img_diff @ text_grained_features_same.t()

                multi_label = multi_label.long().cuda()


                hint = fg_sm_diff[:,1:]>fg_sm_same[:,1:]
                fg_sm = torch.stack([fg_sm_same,fg_sm_diff], dim=2)


                annotation = []
                for ind in range(hint.shape[0]):
                    template_diff = SPOOF_RELATION_PROMPT
                    template_same = REAL_RELATION_PROMPT
                    hint_array = hint[ind].nonzero().flatten().detach().cpu().numpy()
                    if len(hint_array) != 0:
                        for hint_ind in hint_array:
                            template_diff += ' '+generate_indicator_template(hint_ind)
                    template_diff = self.encode_text(template_diff)
                    template_same = self.encode_text(template_same)
                    template_diff = template_diff / template_diff.norm(dim=-1, keepdim=True)
                    template_same = template_same / template_same.norm(dim=-1, keepdim=True)
                    fg_sm_cls_diff = logit_scale * cls_diff[ind] @ template_diff.t()
                    fg_sm_cls_same = logit_scale * cls_diff[ind] @ template_same.t()
                    annotation.append(torch.stack([fg_sm_cls_same,fg_sm_cls_diff], dim=1))


                fg_sm_annotated = torch.stack(annotation,dim=1)
                fg_sm_annotated = fg_sm_annotated.squeeze(0)


                fg_sm_pseudo = fg_sm[:,1:]
                fg_sm_pseudo = fg_sm_pseudo.reshape(fg_sm_pseudo.shape[0]*fg_sm_pseudo.shape[1],fg_sm_pseudo.shape[2])
                annotated_label = torch.cat((torch.zeros(img_real_diff.shape[0]),torch.ones(img_fake_diff.shape[0])),dim=0).type(torch.LongTensor).cuda()
                pseudo_label = multi_label.reshape(multi_label.shape[0]*multi_label.shape[1])


                temp_annotation_loss,temp_pseudo_loss = attribute_balanced_ce_loss(fg_sm_annotated,fg_sm_pseudo,annotated_label,pseudo_label)


                annotation_loss += temp_annotation_loss
                if not torch.isnan(temp_pseudo_loss):
                    pseudo_loss += temp_pseudo_loss


                fg_sm = fg_sm.reshape(fg_sm.shape[0]*fg_sm.shape[1],fg_sm.shape[2])
                fg_hist.extend(fg_sm.detach().cpu().numpy())
                multi_label = torch.cat([annotated_label,pseudo_label], dim=0)
                multi_label_hist.extend(multi_label.detach().cpu().numpy())

                if i == 0:
                    similarity = fg_sm_annotated
                    label = annotated_label
                else:
                    similarity = torch.cat((similarity,fg_sm_annotated),dim=0)
                    label = torch.cat((label,annotated_label),dim=0)


            cls_loss = annotation_loss/len(input_tuple)
            pseudo_loss = pseudo_loss/len(input_tuple)

            view_loss = 0
            return similarity,label,cls_loss,pseudo_loss,view_loss,fg_hist,multi_label_hist


    def forward(self, reference,inference,ref_mask,inf_mask,text,aug=False):
        reference,ref_mask = self.clip_backbone(reference,ref_mask)
        reference_cls = reference[:, 0, :]
        reference_token = reference[:, 1:, :]

        inference,inf_mask = self.clip_backbone(inference,inf_mask)
        inference_cls = inference[:, 0, :]
        inference_token = inference[:, 1:, :]


        cls_feat =  self.rca(reference,inference,ref_mask,inf_mask)
        image_features =  self.frpl(reference,inference,ref_mask,inf_mask)


        cls_feat = self.relation_projection(cls_feat[:, 0, :])
        cls_feat = self.clip_backbone.visual.ln_post(cls_feat)
        cls_feat = cls_feat @ self.clip_backbone.visual.proj
        cls_feat = cls_feat / cls_feat.norm(dim=-1, keepdim=True)

        image_features = self.clip_backbone.visual.ln_post(image_features[:, 0, :])
        image_features = image_features @ self.clip_backbone.visual.proj
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        fine_grained_template_diff = [SPOOF_RELATION_PROMPT, generate_indicator_template(0),generate_indicator_template(1),generate_indicator_template(2),generate_indicator_template(3)]
        fine_grained_template_same = [REAL_RELATION_PROMPT, generate_indicator_template(0,False),generate_indicator_template(1,False),generate_indicator_template(2,False),generate_indicator_template(3,False)]

        logit_scale = self.clip_backbone.logit_scale.exp()

        text_grained_features_diff = self.encode_text(fine_grained_template_diff)
        text_grained_features_diff = text_grained_features_diff / text_grained_features_diff.norm(dim=-1, keepdim=True)
        fg_sm_diff = logit_scale * image_features @ text_grained_features_diff.t()

        text_grained_features_same = self.encode_text(fine_grained_template_same)
        text_grained_features_same = text_grained_features_same / text_grained_features_same.norm(dim=-1, keepdim=True)
        fg_sm_same = logit_scale * image_features @ text_grained_features_same.t()

        hint = fg_sm_diff[:,1:]>fg_sm_same[:,1:]

        annotation = []
        for ind in range(hint.shape[0]):
            template_diff = SPOOF_RELATION_PROMPT
            template_same = REAL_RELATION_PROMPT
            hint_array = hint[ind].nonzero().flatten().detach().cpu().numpy()
            if len(hint_array) != 0:
                for hint_ind in hint_array:
                    template_diff += ' '+generate_indicator_template(hint_ind)
            template_diff = self.encode_text(template_diff)
            template_same = self.encode_text(template_same)
            template_diff = template_diff / template_diff.norm(dim=-1, keepdim=True)
            template_same = template_same / template_same.norm(dim=-1, keepdim=True)
            fg_sm_cls_diff = logit_scale * cls_feat[ind] @ template_diff.t()
            fg_sm_cls_same = logit_scale * cls_feat[ind] @ template_same.t()
            annotation.append(torch.stack([fg_sm_cls_same,fg_sm_cls_diff], dim=1))

        similarity = torch.stack(annotation, dim=1)[0]
        return similarity

    def forward_5_shots(self, reference, inference, ref_mask, inf_mask, text, aug=False):


        if not isinstance(reference, (list, tuple)):
            reference_list = [reference]
        else:
            reference_list = list(reference)

        if not isinstance(ref_mask, (list, tuple)):
            ref_mask_list = [ref_mask] * len(reference_list)
        else:
            ref_mask_list = list(ref_mask)

        assert len(reference_list) == len(ref_mask_list), "reference_list and ref_mask_list must have same length"


        inference_feat, inf_mask_feat = self.clip_backbone(inference, inf_mask)
        inference_cls = inference_feat[:, 0, :]
        inference_token = inference_feat[:, 1:, :]

        per_ref_similarity = []

        for ref_img, ref_m in zip(reference_list, ref_mask_list):

            ref_img, ref_m = ref_img.cuda(),ref_m.cuda()
            reference_feat, ref_mask_feat = self.clip_backbone(ref_img, ref_m)
            reference_cls = reference_feat[:, 0, :]
            reference_token = reference_feat[:, 1:, :]


            cls_feat = self.rca(reference_feat, inference_feat, ref_mask_feat, inf_mask_feat)
            image_features = self.frpl(reference_feat, inference_feat, ref_mask_feat, inf_mask_feat)


            cls_feat = self.relation_projection(cls_feat[:, 0, :])
            cls_feat = self.clip_backbone.visual.ln_post(cls_feat)
            cls_feat = cls_feat @ self.clip_backbone.visual.proj
            cls_feat = cls_feat / cls_feat.norm(dim=-1, keepdim=True)

            image_features = self.clip_backbone.visual.ln_post(image_features[:, 0, :])
            image_features = image_features @ self.clip_backbone.visual.proj
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)


            fine_grained_template_diff = [
                SPOOF_RELATION_PROMPT,
                generate_indicator_template(0),
                generate_indicator_template(1),
                generate_indicator_template(2),
                generate_indicator_template(3),
            ]
            fine_grained_template_same = [
                REAL_RELATION_PROMPT,
                generate_indicator_template(0, False),
                generate_indicator_template(1, False),
                generate_indicator_template(2, False),
                generate_indicator_template(3, False),
            ]

            logit_scale = self.clip_backbone.logit_scale.exp()

            text_grained_features_diff = self.encode_text(fine_grained_template_diff)
            text_grained_features_diff = text_grained_features_diff / text_grained_features_diff.norm(dim=-1, keepdim=True)
            fg_sm_diff = logit_scale * image_features @ text_grained_features_diff.t()

            text_grained_features_same = self.encode_text(fine_grained_template_same)
            text_grained_features_same = text_grained_features_same / text_grained_features_same.norm(dim=-1, keepdim=True)
            fg_sm_same = logit_scale * image_features @ text_grained_features_same.t()

            hint = fg_sm_diff[:, 1:] > fg_sm_same[:, 1:]

            annotation = []
            for ind in range(hint.shape[0]):
                template_diff = SPOOF_RELATION_PROMPT
                template_same = REAL_RELATION_PROMPT
                hint_array = hint[ind].nonzero().flatten().detach().cpu().numpy()
                if len(hint_array) != 0:
                    for hint_ind in hint_array:
                        template_diff += ' ' + generate_indicator_template(hint_ind)

                template_diff_feat = self.encode_text(template_diff)
                template_same_feat = self.encode_text(template_same)
                template_diff_feat = template_diff_feat / template_diff_feat.norm(dim=-1, keepdim=True)
                template_same_feat = template_same_feat / template_same_feat.norm(dim=-1, keepdim=True)

                fg_sm_cls_diff = logit_scale * cls_feat[ind] @ template_diff_feat.t()
                fg_sm_cls_same = logit_scale * cls_feat[ind] @ template_same_feat.t()


                annotation.append(torch.stack([fg_sm_cls_same, fg_sm_cls_diff], dim=0))

            similarity_i = torch.stack(annotation, dim=0)
            per_ref_similarity.append(similarity_i)


        return per_ref_similarity
