# 📄 CornerNet Review - One Stage Detector(Anchor Free)
<br>
<br>

## 🔍 Research Background
### Object Detection는 두 방식으로 나뉨 : one stage detector와 two stage detector
### 기존의 one stage detector는 Anchor box들을 이미지에 densly하게  생성하고,
### prediction에서 Anchor boxes들의 점수를 매겨 final box를 만들어 회귀시키는 방법을 사용했었다.
### 이로 인해 두가지의 drawbacks을 가지는데,
### 첫 번째로 very large set of anchor boxes가 필요하다.
### (DSSD(Fu et al., 2017) 에서는 40k개 이상, RetinaNet(Lin et al., 2017) 에서는 100k 이상의 Anchor boxes가 필요)
### 이는 각 Anchor boxes가 Ground Truth boxes와 겹치게 Train 되기 때문이며,
### 대부분의 Ground Truth boxes(GT boxes)와 충분히 겹치기 위해서는(IoU) 많은 Anchor boxes가 필요하다.
### 결과적으로, GT boxes와 overlap 되는 Anchor boxes의 비율은 매우 작고,
### 이는 positive Anchor box와 negative Anchor box 사이의 inbalance를 발생시키고, Train 속도를 낮추게 된다.
### 두 번째로, Anchor boxes의 사용은 많은 HyperParameters와 design choice를 발생시킨다.
### (how many boxes, what sizes, what aspect ratios)
### 이 parameters은 대부분 heuristic(경험적, 직관적)으로 이루어진다.
### 이에, 저자들은 Anchor boxes를 제안하지 않는 Anchor Free 방법을 선택한 CornerNet을 제안한다.



<br>
<br>

## 🔍 Idea of CornerNet
### CornerNet의 Idea는 Object를 box로 detection하지 않고, 
### __a pair of keypoints(the top-left corner & bottom-right corner)로 detection__ 하는 것.
### single convolution network를 이용하여, all categories의 top-left corner와 
### bottom-right corner를 나타내는 heatmap을 예측하고,
### 각 detection된 corner에 대한 embedding vector를 예측한다.
### embedding vector는 동일한 object category에 속하는 a pair of corners를 group하는데 사용된다.

![Heatmap & Embedding - official paper](https://github.com/user-attachments/assets/dfb952c5-821f-4b7c-8a3e-53d9d853b41f)
<br>
<br>

### 또 다른 Idea는 __Corner Pooling__ 이다.
### Corner Pooling 이란, Convolution Network가 Bounding boxes의 corner를 더 잘 localize 할 수 있도록 돕는 
### Custom Pooling Layer 이다. (공식 깃허브를 보면 torch.utils.cpp_extension으로 구현되어있음.)
### Bounding box의 corner는 종종 object의 외부에 있기에 local info만으로는 corner를 정확히 찾기 어려울 수 있다.
### pixel 위치에 top-left corner가 존재하는지 여부를 결정하기 위해서는
### object의 topmost boundary를 찾기 위해 해당 pixel 위치에서 수직으로 아래쪽을,
### object의 leftmost boundary를 찾기 위해 해당 pixel 위치에서 수평으로 오른쪽을 살펴봐야 한다.
<br>

![example - offical paper](https://github.com/user-attachments/assets/ac0c2dfc-73ff-4bca-b2df-2cff5e58f8d5)

<br>
<br>

## 🔍 Corner Pooling - Bottom Pooling example (ref : official github)
### 공식 github를 보면, models/py_utils/_cpools/src/ 밑에 cpp로 작성된 파일들이 존재.
### (bottom_pool.cpp, left_pool.cpp, right_pool.cpp, top_pool.cpp)
### 구조상 동작 방식은 비슷하니, bottom_pool.cpp 을 보자.
### 1. pool_forward
### input.shape = (bs, c, h, w)
### output = torch.zeros_like(input) 을 생성.
### input Tensor의 height값을 가져온다 : height = input.size(2)
### input Tensor와 output Tensor의 마지막 column를 복사 
### tensor.select(dim, idx) &nbsp;&nbsp; ⇒ &nbsp;&nbsp; (dim+1)차원에서 (idx)번째 인덱스를 선택.
#### dim은 batch &nbsp;&nbsp; ⇒ &nbsp;&nbsp; channel &nbsp;&nbsp; ⇒ &nbsp;&nbsp; height &nbsp;&nbsp; ⇒ &nbsp;&nbsp; width 순서임. 
### output Tensor의 마지막 row에 input_temp을 복사 : output_temp.copy_(input_temp)
### Vertical 방향으로 maxpooling 수행 

```cpp
at::Tensor max_temp;
    for (int64_t ind = 0; ind < height - 1; ++ind) {
        input_temp  = input.select(2, ind + 1); # 3번째 차원(height)에서 (ind+1)번째 index를 선택
        output_temp = output.select(2, ind); # output의 ind번째 row들을 output_temp에 입력
        max_temp    = output.select(2, ind + 1); # output의 (ind+1)번째 row들을 max_temp에 입력 -> pool 연산

        at::max_out(max_temp, input_temp, output_temp); # input_temp와 output_temp의 각 element를 비교하여 
                                                          더 큰 값을 max_temp에 저장 -> pool 연산
    }
```
### ④⑤⑥⑦⑧⑨⑩

### 이해하기 어려울 수도 있으니 예시를 살펴보자.
### ① input = $\begin{bmatrix} 7 & 1 & 3 \\ 2 & 4 & 5 \\ 3 & 9 & 4 \end{bmatrix}$, &nbsp; shape = (1, 1, 3, 3)
### ② output = torch.zeros_like(input) = $\begin{bmatrix} 0 & 0 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & 0 \end{bmatrix}$
### ③ height &nbsp; = &nbsp; input.size(2) &nbsp; = 3
### ④ input_temp &nbsp; = &nbsp; input.select(2, 0) &nbsp; = &nbsp; $\boldsymbol{[7, 1, 3]}$
### ⑤ output_temp &nbsp; = &nbsp; output.select(2, 0) &nbsp; = &nbsp; $\boldsymbol{[0, 0, 0]}$
### ⑥ output_temp.copy_(input_temp) &nbsp; = &nbsp; $\boldsymbol{[7, 1, 3]}$ &nbsp; -> &nbsp; output Tensor의 첫 번째 row가 결정됨.
### ⑦ < ind = 0 >
### &nbsp;&nbsp;&nbsp;&nbsp; input_temp &nbsp; = &nbsp; input.select(2, 0+1) &nbsp; -> &nbsp; $\boldsymbol{[2, 4, 5]}$
### &nbsp;&nbsp;&nbsp;&nbsp; output_temp &nbsp; = &nbsp; output.select(2, 0) &nbsp; -> &nbsp; $\boldsymbol{[7, 1, 3]}$
### &nbsp;&nbsp;&nbsp;&nbsp; max_temp = output.select(2, 0+1) &nbsp; -> &nbsp; $\boldsymbol{[0, 0, 0]}$
### &nbsp;&nbsp;&nbsp;&nbsp; at::max_out(max_temp, input_temp, output_temp) &nbsp; -> &nbsp; max_temp가 $\boldsymbol{[7, 4, 5]}$ 로 바뀜. &nbsp; max_temp가 변경되면 output 텐서도 변경됨.
### &nbsp;&nbsp;&nbsp;&nbsp; 위 작업을 ind = height - 1까지 반복
### ⑧ output = $\begin{bmatrix} 7 & 1 & 3 \\ 7 & 4 & 5 \\ 7 & 9 & 5 \end{bmatrix}$ 을 return.
### ⑨⑩




### 2. pool_backward (input ,grad_output)
### auto : 변수의 type을 compiler가 자동으로 추론하도록 함.



<br>
<br>

## 🔍 Model Architecture - official code를 참고하여 설명
![Model Architecture - official paper](https://github.com/user-attachments/assets/5fac8d6f-5a3a-404d-aaf4-ef06a660ef98)
##### 전체적인 구조 그림은 위 그림과 같다.


### backbone으로 두 개의 Hourglass Network를 사용.
### 

<br>
<br>

## 🔍 Loss Function

### total Training Loss $\boldsymbol{L}$ 은 detection loss($\boldsymbol{L_{det}}$), pull loss($\boldsymbol{L_{pull}}$), push loss($\boldsymbol{L_{push}}$), offset loss($\boldsymbol{L_{offset}}$) 의 총합이다 :


###  $\boldsymbol{L = L_{det} + \alpha L_{pull} + \beta L_{push} + \gamma L_{off}}$


### 여기서

- ### $\boldsymbol{L_{det}}$ &nbsp;:&nbsp; detection loss
- ### $\boldsymbol{L_{pull}}$ &nbsp;:&nbsp; pull loss
- ### $\boldsymbol{L_{push}}$ &nbsp;:&nbsp; push loss
- ### $\boldsymbol{L_{off}}$ &nbsp;:&nbsp; offset loss
- ### $\boldsymbol{\alpha, \beta, \gamma}$ &nbsp;:&nbsp; 각각 pull, push, offset 손실의 가중치

### ① $\boldsymbol{L_{det}}$ &nbsp;:&nbsp; variant focal loss 


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ![CodeCogsEqn](https://github.com/user-attachments/assets/00850702-d875-43f8-a55f-683d303e15d8)
##### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ~~[github 올리면 latex 오류가 나서;; 그냥 그림으로 붙였습니다]~~





### &nbsp;&nbsp;&nbsp;&nbsp; $N$ = number of objects in Image
### &nbsp;&nbsp;&nbsp;&nbsp; $\alpha, \; \beta$ = 각 points의 contribution을 control하는 Hyperparameters ($\alpha = 2, \; \beta=4$)
### &nbsp;&nbsp;&nbsp;&nbsp; Gaussian bump가 $y_{cij}$ 에 encode 되어있기 때문에 $(1-y_{cij})$ 는 GT location 주변에 패널티를 줄인다.

### ② $L_{push} \And L_{pull}$ &nbsp;:&nbsp; Embedding loss
### &nbsp;&nbsp;&nbsp;&nbsp; Network는 each detected corner에 대한 Embedding vector를 예측하여 
### &nbsp;&nbsp;&nbsp;&nbsp; tl-corner와 br-corner가 같은 bounding box에 속하는 경우, embedding 거리가 작도록 함.
### &nbsp;&nbsp;&nbsp;&nbsp; 그런 다음 tl-corner와 br-corner의 embedding 간의 거리를 바탕으로 corner를 grouping 한다.
### &nbsp;&nbsp;&nbsp;&nbsp; Embedding의 실제 값은 중요하지 않고, 거리만이 corner를 grouping 하는데 사용된다.
### &nbsp;&nbsp;&nbsp;&nbsp; $L_{pull} = {1 \over N} \displaystyle\sum_{k=1}^{N} [(e_{tk}-e_{k})^2 \; + \; (e_{bk}-e_{k})^2]$
### &nbsp;&nbsp;&nbsp;&nbsp; $L_{push} = {1 \over N(N+1)} \; \displaystyle\sum_{k=1}^{N} \; \displaystyle\sum_{j=1, \; j \not ={k}}^{N}max(o, \Delta - |e_{k}-e_{j}|)$
### &nbsp;&nbsp;&nbsp;&nbsp; 여기서 $e_{k}$는 $e_{tk}$ 와 $e_{bk}$ 의 평균이고, $\Delta$= 1 로 설정.
### &nbsp;&nbsp;&nbsp;&nbsp; GT corner location 에서만 loss를 적용
### ③ $L_{off}$ &nbsp;:&nbsp; offset loss
### &nbsp;&nbsp;&nbsp;&nbsp; Image가 Convolution 될 때, output의 크기는 input보다 작음.
### &nbsp;&nbsp;&nbsp;&nbsp; 이미지의 위치는 $(x,\, y)$ 에서 heatmap의 위치 $(\lfloor {x \over n}\rfloor), \; (\lfloor {y \over n}\rfloor)$ 으로 mapping된다.
### &nbsp;&nbsp;&nbsp;&nbsp; heatmap에서 Input Image로 위치를 다시 mapping 할 때, precision이 손실된 수 있으며, 
### &nbsp;&nbsp;&nbsp;&nbsp;이는 small bounding box 의 IoU에 영향을 끼칠 수 있다.
### &nbsp;&nbsp;&nbsp;&nbsp; 이에, corner 위치를 Input Resolution으로 remapping 하기 전에,
### &nbsp;&nbsp;&nbsp;&nbsp; corner 위치를 약간 조정하는 position offset을 예측.


### &nbsp;&nbsp;&nbsp;&nbsp; $O_{k} = ({x \over n} -\lfloor {x \over n}\rfloor, {y \over n} -\lfloor {y \over n}\rfloor)$
### &nbsp;&nbsp;&nbsp;&nbsp; 여기서 $O_{k}$ 는 offset 이고, $x_{k}, \; y_{k}$ 는 각각 corner $k$ 의 $x$ 좌표, $y$ 좌표이다.

### &nbsp;&nbsp;&nbsp;&nbsp; 특히, 모든 categories 의 tl-corner가 공유하는 하나의 offset set과 br-corner가 공유하는 또 다른 set을 예측.
### &nbsp;&nbsp;&nbsp;&nbsp; Training을 위해 GT corner와 offset set의 smoothL1Loss를 구한다.
### &nbsp;&nbsp;&nbsp;&nbsp; $L_{off} = {1 \over N} \displaystyle\sum_{k=1}^{N} smoothL1Loss(O_{k},\; \hat{O}_{k})$
### &nbsp;&nbsp;&nbsp;&nbsp; $O_{k}$ 는 pred, &nbsp; $\hat{O}_{k}$ 는 GT



<br>
<br>
