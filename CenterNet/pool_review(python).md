# 📄 CenterNet - Pool(python)

### C++로 작성한 pooling module을 import해서 사용
<br>

### ①②③④⑤⑥⑦⑧⑨⑩

## 🔍 Top-Left Pool &nbsp;&&nbsp; Bottom-Right Pool &nbsp;-&nbsp; pool Class
```python
class tl_pool(pool):
    def __init__(self, dim):
        super(tl_pool, self).__init__(dim, TopPool, LeftPool)


class TopPool(nn.Module):
    def forward(self, x):
        return TopPoolFunction.apply(x)

class LeftPool(nn.Module):
    def forward(self, x):
        return LeftPoolFunction.apply(x)




class br_pool(pool):
    def __init__(self, dim):
        super(br_pool, self).__init__(dim, BottomPool, RightPool)


class BottomPool(nn.Module):
    def forward(self, x):
        return BottomPoolFunction.apply(x)

class RightPool(nn.Module):
    def forward(self, x):
        return RightPoolFunction.apply(x)




class pool(nn.Module):
    def __init__(self, dim, pool1, pool2)
        super().__init__()
        self.pool1 = pool1()
        self.pool2 = pool2()
        
        self.p1_conv1 = convolution(3, dim, 128) 
        self.p2_conv1 = convolution(3, dim, 128)

        self.p_conv1 = nn.Conv2d(128, dim, (3, 3), padding=(1, 1), bias=False)
        self.p_bn1   = nn.BatchNorm2d(dim)

        self.conv1 = nn.Conv2d(dim, dim, (1, 1), bias=False)
        self.bn1   = nn.BatchNorm2d(dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = convolution(3, dim, dim)

        self.look_conv1 = convolution(3, dim, 128)
        self.look_conv2 = convolution(3, dim, 128)
        self.P1_look_conv = nn.Conv2d(128, 128, (3, 3), padding=(1, 1), bias=False)
        self.P2_look_conv = nn.Conv2d(128, 128, (3, 3), padding=(1, 1), bias=False)
        ...

    def forward(self, x):
        # x.shape = (dim, h, w)
        # pool 1
        look_conv1   = self.look_conv1(x) # shape = (128, h, w)
        p1_conv1     = self.p1_conv1(x)   # shape = (128, h, w)
        look_right   = self.pool2(look_conv1) # shape = (128, h, w)
        P1_look_conv = self.P1_look_conv(p1_conv1+look_right) # shape = (128, h, w)
        pool1        = self.pool1(P1_look_conv) # shape = (128, h, w)

        # pool 2
        look_conv2   = self.look_conv2(x)
        p2_conv1     = self.p2_conv1(x)
        look_down    = self.pool1(look_conv2)
        P2_look_conv = self.P2_look_conv(p2_conv1+look_down)
        pool2        = self.pool2(P2_look_conv)

        # pool 1 + pool 2
        p_conv1 = self.p_conv1(pool1 + pool2) # shape = (128, h, w)
        p_bn1   = self.p_bn1(p_conv1)

        conv1 = self.conv1(x) # shape = ()
        bn1   = self.bn1(conv1)
        relu1 = self.relu1(p_bn1 + bn1)

        conv2 = self.conv2(relu1)
        return conv2



```



<br>

## 🔍 CenterPool &nbsp;-&nbsp; pool_cross Class



<br>