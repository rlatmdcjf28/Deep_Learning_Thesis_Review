# 📄 CenterNet Review - One Stage Detector(Anchor Free)

<br>

### ①②③④⑤⑥⑦⑧⑨⑩

## 🔍 Research Background
현재 object detection task에서 popular flowchart는 Anchor-based object detection 이다.<br>
Anchor-based 에서는 사전에 정의된 크기의 rectangle들을 배치하고, 이를 Ground Truth(GT) objects 에 맞추기 위해 regress한다.<br>
이러한 접근은 GT objects와 충분히 높은 IoU를 보장하기 위해 많은 수의 Anchor가 필요하며, 각 Anchor box의 size와 aspect ratio를 수동으로 design해야 한다.<br>
또한, Anchor box는 보통 GT box와 일치하지 않아, bounding box classification에 적합하지 않다.<br>
이를 해결하기 위해, CornerNet이라는 a keypoint-based object detection pipline이 제안되었다.<br>
CornerNet은 each object를 a pair of corner keypoints로 표현하여 Anchor box의 필요성을 없애고, one stage object detection accuracy에서 SOTA를 달성하였다.<br>
하지만, CornerNet의 performance는 object의 global information을 refering하는 능력이 부족하다.<br>
이 문제를 해결하기 위해, 저자들은 the central part of a proposal, 즉 geometric center에 가까운 영역을 explore하는 low-cost의 효과적인 solution인 CenterNet을 제안.


### 1.
1. 
2.

<br>

## 🔍 Idea of CenterNet - Center Pooling & Cascade Corner Pooling
① Center Pooling<br>
&nbsp;&nbsp;&nbsp;&nbsp; Center keypoints가 objects의 visual patterns을 more recognizable하여 proposal된 central part를 더 쉽게 인식하도록 함.<br>
&nbsp;&nbsp;&nbsp;&nbsp; 이는 Center key-points를 predict하는 featuremap에서 horizontal & vertical의 sum 중 maximum을 얻음으로 할 수 있다.

<br>

② Cascade Corner Pooling<br>
&nbsp;&nbsp;&nbsp;&nbsp; 기존의 CornerNet의 Corner Pooling에 internal information을 인식하는 ability를 부여한다.<br>
&nbsp;&nbsp;&nbsp;&nbsp; 이는 object의 feature map에서 boundary와 internal direction의 max summed response를 얻음으로 달성된다.<br>
&nbsp;&nbsp;&nbsp;&nbsp; 이러한 two-directional pooling method가 feature-level noises에 더 robust됨을 확인하였고,<br>
&nbsp;&nbsp;&nbsp;&nbsp; precision과 recall의 improvement에 기여함을 확인하였다.

<br>

## 🔍 CornerNet vs CenterNet
CornerNet은 corners를 detect하기 위해 두 개의 heatmaps(heatmap of top-left corners & heatmap of bottom-right corners)을 생성한다.<br>
The heatmaps은 different categories의 keypoints의 location을 나타내고, 각 keypoint에 대한 confidence score를 할당한다.<br>
이외에도, each corner에 대해 Embedding과 a group of offsets을 예측한다.<br>
Embedding은 tow corners가 같은 object인지 식별하는데 사용.<br>
Offset은 heatmaps의 corner를 Input Image로 remap하는 방법을 학습한다.<br>
object bounding box를 생성하기 위해, tl corners 와 br corners의 top-k개를 각각 heatmap에서 scores에 따라 선택한다.<br>
그런 다음, a pair of corners의 Embedding vectors의 거리를 계산하여 이 pair가 동일한 객체에 속하는지를 결정한다.<br>
object bounding box는 distance가 threshold보다 작으면 생성되고, bounding box에는 a pair of corners의 average scores가 confidence score로 할당된다.<br>

CenterNet은 CornerNet을 baseline으로 사용한다.<br>
CenterNet은 CornerNet과 달리, object를 detect하기 위해 3개의 key-points를 사용한다.<br>
이렇게 함으로써 ROI Pooling을 일부 상속할 수 있다.<br>
그리고 Center Pooling과 Cascade Corner Pooling을 사용하여 object 내부의 visual patterns을 key-point detection process에 도입한다.


<br>

## 🔍 Model Architecture
![image](https://github.com/user-attachments/assets/1b31e6de-b72a-437b-98b6-ce4b3abd2a0d)
[모델의 전체적인 구조 그림]

저자들은 each objects를 Center Keypoint와 a pair of corners로 표현한다.<br>
CornerNet을 기반으로, center keypoints를 위한 heatmap을 embedding 하고, center keypoints의 offset을 예측한다.<br>
그 다음, CornerNet에서 제안한 방법을 사용하여 top-k의 bounding box를 생성한다.<br>
하지만 incorrect bounding boxes를 효과적으로 filter out 하기 위해 detect된 center keypoints를 활용하여 다음 과정을 따른다.<br>
1. scores에 따라 top-k center keypoints를 select.
2. corresponding offsets을 활용하여 위에서 select한 center keypoints를 Input Image로 remap 한다.
3. each bounding boxes에 대해 central region을 정의하고, 이 central region에 center keypoints가 있는지 check한다.<br>이 때 center keypoints의 class label은 bounding boxes의 class label과 동일해야한다.
4. central region에 center keypoint가 detect되면 bounding box를 preserve한다.<br>bounding box의 score는 top-left corner, bottom-right corner, center keypoint의 average score로 replace된다.<br>if central region에 center keypoints가 없으면 bounding box는 remove된다.
   
central region in the bounding box의 size는 detection results에 영향을 미친다.<br>
예를 들어, smaller central regions는 small bounding boxes에 대한 recall rate를 낮추고, <br>larger central regions는 large bounding boxes에 대한 precision을 낮춘다.
- __recall__ ? &nbsp; GT object를 얼마나 잘 detect하는지에 대한 지표
- __precision__ ? &nbsp; detect한 objects 중 GT object가 얼마나 많은지에 대한 지표
-  why ? &nbsp; __smaller central regions__ 일 경우, GT objects의 center를 정확하게 맞추기 어렵다. <br>특히 object의 size가 작을수록 center가 작아지기 때문에 smaller central region은 이 object의 center point를 놓치기 쉬움.<br>결과적으로, small object의 center point를 놓치게 되어 Recall이 낮아지게 된다.<br> __larger central regions__ 일 경우, bounding box가 더 큰 area를 포함하게 되고, object의 center point와 상관없는 부분이 포함될 가능성이 있음.<br>결과적으로, large object의 경우 center point와 관련없는 다른 region이 detect될 수 있어 Precision이 낮아지게 된다.

이를 해결하기 위해, Bounding box의 size에 따라 adaptively으로 center region을 fit하는 __scale-aware central region__ 을 제안한다.<br>
<br>
Bounding box $i$가 preserve되어야할지 결정해야한다고 가정해보자.<br>
$\Large tl_{x}$ 와 $\Large tl_{y}$ 는 Top-Left corner의 coordinate를 나타내고, $\Large br_{x}$ 와 $\Large br_{y}$ 는 Bottom-Right corner의 coordinate를 나타낸다.<br>
centeral region $\Large j$를 정의하자.<br>
$\Large ctl_x$ 와 $\Large ctl_y$ 는 $\Large j$ 의 tl coordinate를 나타내고, $\Large cbr_x$ 와 $\Large cbr_y$ 는 $\Large j$ 의 br coordinate를 나타낸다.<br>
그러면 $\Large tl_x, tl_y, br_x, br_y, ctl_x, ctl_y, cbr_x, cbr_y$ 는 다음 관계를 만족해야 한다.


$
\Large {\begin{cases}
     ctl_x = {{(n + 1) tl_x + (n - 1) br_x} \over {2n}}\\
     ctl_y = {{(n + 1) tl_y + (n - 1) br_y} \over {2n}}\\
     cbr_x = {{(n - 1) tl_x + (n + 1) br_x} \over {2n}}\\
     cbr_y = {{(n - 1) tl_y + (n + 1) br_y} \over {2n}}
  \end{cases}}
$

여기서 $\Large n$ 은 central region $\Large j$ 의 scale을 결정하는 odd number이다.<br>
이 논문에서는 bounding box의 scale이 150보다 작으면 $\Large n = 3$ 으로,<br>
150보다 크면 $\Large n = 5$ 로 설정한다. 아래 그림은 $\Large n$에 따라 두 central region의 차이를 보여준다.<br>
위 식에 따라, __scale-aware central region__ 을 결정한 후, central region이 keypoints를 contain 하는지를 확인한다.

![Screenshot from 2024-07-26 16-59-21](https://github.com/user-attachments/assets/6d943769-38a9-46d0-90bd-043656bdbb38)


<br>

## 🔍 Loss Function

<br>







