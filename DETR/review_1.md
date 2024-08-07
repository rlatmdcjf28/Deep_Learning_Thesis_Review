# 📄 DETR - One Stage Detector(Anchor Free)
<br>
<br>

## 🔍 Research Background & Idea
### object detection의 goal은 each interest object에 대해 a set of bounding boxes와 category labels을 예측하는 것이다.
### Modern detectors는 large set of proposals, anchors, window centers를 통해 prediction 작업을 간접적으로 해결한다.
### 위 detector들의 성능은 duplicate prediction을 제거하기 위한 postprocessing(NMS), anchor set의 설계 및 target box를 anchor에 할당하는 heuristic에 큰 영향을 받는다.
### 이를 단순화 하기 위해 저자들은 surrogate tasks을 우회하는 direct set prediction 접근법을 제안.
### 또한, DETR은 Anchor boxes나 NMS와 같은 hand design된 components를 제거하여 detection pipline을 단순화한다.
### DETR은 이전의 Detectors에서 사용되던 Anchor boxes를 설계하지 않고, a set of prediction으로 detection을 진행한다.
### NLP에서 popular architecture가 된 Transformer 기반의 Encoder-Decoder Architecture가 set prediction task에 적합하다고 판단하여 Object Detection에도 적용.
### 또한 predict된 object와 ground truth(GT) object 간의 bipartite matching을 수행하는 a set loss function을 사용한다.

<br>

![simple model architecture - ref : official paper](https://github.com/user-attachments/assets/41ce9ed5-8070-4049-8272-e5bfd671e699) [간단한 모델 구조 그림]

<br>
<br>


## 🔍 Model Code (ref : official code)
### ⓵ ⓶ ⓷ ⓸ ⓹ ⓺ ⓻ ⓼ ⓽ ⓪
### 1️⃣ 2️⃣ 3️⃣ 4️⃣ 5️⃣ 6️⃣ 7️⃣ 8️⃣ 9️⃣ 0️⃣

### 1️⃣ DETR.forward (samples : NestedTensor)
```python
"""
forward로 NestedTensor가 입력되어야 함. NestedTensor의 구성요소는 다음과 같음
⓵ samples.tensor -> batch image = (bs, 3, H, W)
⓶ samples.mask   -> binary mask = (bs, H, W)

다음 element를 return.
⓵ pred_logits -> classification logits(include no-object) for all queries = (bs, n_q, n_cls+1)
⓶ pred_boxes  -> normalized된 box coordinates for all queries = (cx, cy, h, w)
⓷ aux_outputs -> optional. only return when auxilary losses are activated
"""
features, pos = self.backbone(samples)

src, mask = features[-1].decompose()

hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]
outputs_class = self.class_embed(hs)
outputs_coord = self.bbox_embed(hs).sigmoid()
out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
if self.aux_loss:
    out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
return out
```

### 1️⃣ Backbone
```python
position_embedding = build_position_encoding(args) # PE_sine or PE_learned
backbone = Backbone(args.backbone)
model = Joiner(backbone, position_embedding) # Joiner -> Backbone과 PE를 결합
model.num_channels = backbone.num_channels
return model 
```
### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; initial image &nbsp;:&nbsp; $\boldsymbol{x_{img} \in \mathbb{R}^{3 \times H_{0} \times W_{0}}}$
### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; After Backbone &nbsp;:&nbsp; $\boldsymbol{f \in \mathbb{R}^{C \times H \times W}}$, &nbsp; 여기서 일반적으로 $\boldsymbol{C = 2048}$ 이고 $\boldsymbol{H, W = {H_0 \over 32}, {W_0 \over 32}}$
### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 1X1 Convolution으로 $\boldsymbol{f}$를 새로운 feature map $z_0$를 생성한다 : $\boldsymbol{z_0 \in \mathbb{R}^{d \times H \times W}}$

<br>

### 2️⃣ Transformer.forward (src, mask, query_embed, pos_embed)
### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; bs, c, H, W = src.shape
### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; src $\boldsymbol{\in}$ (bs, C, H, W) &nbsp; $\boldsymbol{\Rightarrow}$ &nbsp; src.flatten(2).permute(2, 0, 1) $\boldsymbol{\in}$ (H$\boldsymbol{\times}$W, bs, C)

### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; mask $\boldsymbol{\in}$ (bs, H, W) &nbsp; $\boldsymbol{\Rightarrow}$ &nbsp; mask.flatten(1) $\boldsymbol{\in}$ (bs, H$\boldsymbol{\times}$w)

### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; query_embed $\boldsymbol{\in}$ (n_q, d) &nbsp; $\boldsymbol{\Rightarrow}$ &nbsp; query_embed.unsqueeze(1).repeat(1, bs, 1) $\boldsymbol{\in}$ (n_q, bs, d)

### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; tgt = torch.zeros_like(query_embed) $\boldsymbol{\in}$ (n_q, bs, d)

### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; memory = TransformerEncoder(src, key_padding_mask = mask, pos = pos_embed) $\boldsymbol{\in}$ (H$\boldsymbol{\times}$W, bs, d)

### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; hs = TransformerDecoder(tgt, memory, memory_key_padding_mask = mask, pos = pos_embed, query_pos = query_embed) $\boldsymbol{\in}$ (n_layer, n_q, bs, d)

### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; return hs.transpose(1, 2) $\boldsymbol{\in}$ (n_layer, bs, n_q, d), &nbsp; memory.permute(1, 2, 0).view(bs, c, H, W) $\boldsymbol{\in}$ (bs, d, H, W)

### 3️⃣ TransformerEncoder.forward (src, src_key_padding_mask = mask, pos = pos_embed)
### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; output = src
### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; for layer in TransformerEncoderLayer:
### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; output = layer(output, src_mask = mask, src_key_padding_mask = src_key_padding_mask, pos = pos)






### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;

<br>
<br>


## 🔍 Loss Function


<br>
<br>
