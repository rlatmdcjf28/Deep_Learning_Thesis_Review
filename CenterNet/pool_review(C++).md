# 📄 CenterNet - Pool(C++)

<br>

input = torch.tensor([[[1, 2, 3], [3, 2, 1], [2, 2, 2]]])

$$\Large{\Rightarrow \text input =\begin{bmatrix}\begin{bmatrix}\begin{bmatrix}
       \ 1 & 2 & 3 \\
       \ 3 & 2 & 1 \\
       \ 2 & 2 & 2
     \ \end{bmatrix}\end{bmatrix}\end{bmatrix}}$$

output = torch.zeros_like(input)
$$\Large{\Rightarrow \text output = \begin{bmatrix}\begin{bmatrix}\begin{bmatrix}
       \ 0 & 0 & 0 \\
       \ 0 & 0 & 0 \\
       \ 0 & 0 & 0
     \ \end{bmatrix}\end{bmatrix}\end{bmatrix}}$$



## 🔍 TopPool

① Input의 마지막 row를 output의 마지막 row에 복사

$$\Large{\Rightarrow \text output = \begin{bmatrix}\begin{bmatrix}\begin{bmatrix}
       \ 0 & 0 & 0 \\
       \ 0 & 0 & 0 \\
       \ 2 & 2 & 2
     \ \end{bmatrix}\end{bmatrix}\end{bmatrix}}$$


② ind = 1 ( 아래에서 두 번째 row )<br>
&nbsp;&nbsp;&nbsp;&nbsp; input [[[3, 2, 1]]] 과 현재 output [[[2, 2, 2]]] 을 비교하여 최댓값을 선택.

$$\Large{\Rightarrow \text output = \begin{bmatrix}\begin{bmatrix}\begin{bmatrix}
       \ 0 & 0 & 0 \\
       \ 3 & 2 & 2 \\
       \ 2 & 2 & 2
     \ \end{bmatrix}\end{bmatrix}\end{bmatrix}}$$


③ ind = 2 ( 아래에서 세 번째 row )<br>
&nbsp;&nbsp;&nbsp;&nbsp; input [[[1, 2, 3]]] 과 현재 output [[[3, 2, 2]]] 을 비교하여 최댓값을 선택.

$$\Large{\Rightarrow \text output = \begin{bmatrix}\begin{bmatrix}\begin{bmatrix}
       \ 3 & 2 & 3 \\
       \ 3 & 2 & 2 \\
       \ 2 & 2 & 2
     \ \end{bmatrix}\end{bmatrix}\end{bmatrix}}$$

④ 최종 output은 아래와 같음

$$\Large{\Rightarrow \text output = \begin{bmatrix}\begin{bmatrix}\begin{bmatrix}
       \ 3 & 2 & 3 \\
       \ 3 & 2 & 2 \\
       \ 2 & 2 & 2
     \ \end{bmatrix}\end{bmatrix}\end{bmatrix}}$$

<br>
<br>

## 🔍 BottomPool
① Input의 첫 번째 row를 output의 첫 번째 row에 복사

$$\Large{\Rightarrow \text output = \begin{bmatrix}\begin{bmatrix}\begin{bmatrix}
       \ 1 & 2 & 3 \\
       \ 0 & 0 & 0 \\
       \ 0 & 0 & 0
     \ \end{bmatrix}\end{bmatrix}\end{bmatrix}}$$


② ind = 1 ( 위에서 두 번째 row )<br>
&nbsp;&nbsp;&nbsp;&nbsp; input [[[3, 2, 1]]] 과 현재 output [[[1, 2, 3]]] 을 비교하여 최댓값을 선택.

$$\Large{\Rightarrow \text output = \begin{bmatrix}\begin{bmatrix}\begin{bmatrix}
       \ 1 & 2 & 3 \\
       \ 3 & 2 & 3 \\
       \ 0 & 0 & 0
     \ \end{bmatrix}\end{bmatrix}\end{bmatrix}}$$


③ ind = 2 ( 위에서 세 번째 row )<br>
&nbsp;&nbsp;&nbsp;&nbsp; input [[[3, 2, 1]]] 과 현재 output [[[3, 2, 3]]] 을 비교하여 최댓값을 선택.

$$\Large{\Rightarrow \text output = \begin{bmatrix}\begin{bmatrix}\begin{bmatrix}
       \ 1 & 2 & 3 \\
       \ 3 & 2 & 3 \\
       \ 3 & 2 & 3
     \ \end{bmatrix}\end{bmatrix}\end{bmatrix}}$$

④ 최종 output은 다음과 같음

$$\Large{\Rightarrow \text output = \begin{bmatrix}\begin{bmatrix}\begin{bmatrix}
       \ 1 & 2 & 3 \\
       \ 3 & 2 & 3 \\
       \ 3 & 2 & 3
     \ \end{bmatrix}\end{bmatrix}\end{bmatrix}}$$

## 🔍 RightPool
① Input의 첫 번째 column를 output의 첫 번째 column에 복사

$$\Large{\Rightarrow \text output = \begin{bmatrix}\begin{bmatrix}\begin{bmatrix}
       \ 1 & 0 & 0 \\
       \ 3 & 0 & 0 \\
       \ 2 & 0 & 0
     \ \end{bmatrix}\end{bmatrix}\end{bmatrix}}$$


② ind = 1 ( 왼쪽에서 두 번째 column )<br>
&nbsp;&nbsp;&nbsp;&nbsp; input [[[2], [2], [2]]] 과 현재 output [[[1], [3], [2]]] 을 비교하여 최댓값을 선택.

$$\Large{\Rightarrow \text output = \begin{bmatrix}\begin{bmatrix}\begin{bmatrix}
       \ 1 & 2 & 0 \\
       \ 3 & 3 & 0 \\
       \ 2 & 2 & 0
     \ \end{bmatrix}\end{bmatrix}\end{bmatrix}}$$


③ ind = 2 ( 왼쪽에서 세 번재 column )<br>
&nbsp;&nbsp;&nbsp;&nbsp; input [[[3], [1], [2]]] 과 현재 output [[[2], [3], [2]]] 을 비교하여 최댓값을 선택.

$$\Large{\Rightarrow \text output = \begin{bmatrix}\begin{bmatrix}\begin{bmatrix}
       \ 1 & 2 & 3 \\
       \ 3 & 3 & 3 \\
       \ 2 & 2 & 2
     \ \end{bmatrix}\end{bmatrix}\end{bmatrix}}$$

④ 최종 output은 다음과 같음

$$\Large{\Rightarrow \text output = \begin{bmatrix}\begin{bmatrix}\begin{bmatrix}
       \ 1 & 2 & 3 \\
       \ 3 & 3 & 3 \\
       \ 2 & 2 & 2
     \ \end{bmatrix}\end{bmatrix}\end{bmatrix}}$$

## 🔍 LeftPool
① Input의 마지막 column를 output의 마지막 column에 복사

$$\Large{\Rightarrow \text output = \begin{bmatrix}\begin{bmatrix}\begin{bmatrix}
       \ 0 & 0 & 3 \\
       \ 0 & 0 & 1 \\
       \ 0 & 0 & 2
     \ \end{bmatrix}\end{bmatrix}\end{bmatrix}}$$


② ind = 1 ( 오른쪽에서 두 번째 column )<br>
&nbsp;&nbsp;&nbsp;&nbsp; input [[[2], [2], [2]]] 과 현재 output [[[3], [1], [2]]] 을 비교하여 최댓값을 선택.

$$\Large{\Rightarrow \text output = \begin{bmatrix}\begin{bmatrix}\begin{bmatrix}
       \ 0 & 3 & 3 \\
       \ 0 & 2 & 1 \\
       \ 0 & 2 & 2
     \ \end{bmatrix}\end{bmatrix}\end{bmatrix}}$$


③ ind = 2 ( 오른쪽에서 세 번째 column )<br>
&nbsp;&nbsp;&nbsp;&nbsp; input [[[1], [3], [2]]] 과 현재 output [[[3], [2], [2]]] 을 비교하여 최댓값을 선택.

$$\Large{\Rightarrow \text output = \begin{bmatrix}\begin{bmatrix}\begin{bmatrix}
       \ 3 & 3 & 3 \\
       \ 3 & 2 & 1 \\
       \ 2 & 2 & 2
     \ \end{bmatrix}\end{bmatrix}\end{bmatrix}}$$

④ 최종 output은 다음과 같음

$$\Large{\Rightarrow \text output = \begin{bmatrix}\begin{bmatrix}\begin{bmatrix}
       \ 3 & 3 & 3 \\
       \ 3 & 2 & 1 \\
       \ 2 & 2 & 2
     \ \end{bmatrix}\end{bmatrix}\end{bmatrix}}$$
