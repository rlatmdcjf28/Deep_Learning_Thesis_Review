# 📄 Stacked Hourglass Networks for Human Pose Estimation Review

<br>

~~CenterNet 논문을 보는데, 사전에 필요한 지식들(Hourglass Network, CornerNet)을 좀 보고 논문을 봐야겠다고 생각이 들어서 review하게 됨~~

### ①②③④⑤⑥⑦⑧⑨⑩

## 🔍 Research Background

<br>


## 🔍 Model Architecture - In PyTorch
```python
class PoseNet(nn.Module):
    def __init__(self, nstack, inp_dim, oup_dim, bn=False, increase=0, **kwargs):
        super(PoseNet, self).__init__()
        self.nstack = nstack
        self.pre = nn.Sequential(
            Conv(3, 64, 7, 2, bn=True, relu=True), # shape = (b, 64, h', w'), (h', w' = h//2, w//2)
            Residual(64, 128), # need_skip = False, shape = (b, 128, h', w')
            Pool(2, 2), # shape = (b, 128, h", w")
            Residual(128, 128), # need_skip = True, shape = (b, 128, h", w")
            Residual(128, inp_dim) # need_skip = False, shape = (b, inp_dim, h", w")
        )
        
        self.hgs = nn.ModuleList( [
        nn.Sequential(
            Hourglass(4, inp_dim, bn, increase),
        ) for i in range(nstack)] )
        
        self.features = nn.ModuleList( [
        nn.Sequential(
            Residual(inp_dim, inp_dim),
            Conv(inp_dim, inp_dim, 1, bn=True, relu=True)
        ) for i in range(nstack)] )
        
        self.outs = nn.ModuleList( [Conv(inp_dim, oup_dim, 1, relu=False, bn=False) for i in range(nstack)] )
        self.merge_features = nn.ModuleList( [Merge(inp_dim, inp_dim) for i in range(nstack-1)] )
        self.merge_preds = nn.ModuleList( [Merge(oup_dim, inp_dim) for i in range(nstack-1)] )
        self.nstack = nstack
        self.heatmapLoss = HeatmapLoss()
    def forward(self, x):
          ## our posenet
          x = imgs.permute(0, 3, 1, 2) #x of size 1,3,inpdim,inpdim
          x = self.pre(x)
          combined_hm_preds = []
          for i in range(self.nstack):
              hg = self.hgs[i](x)
              feature = self.features[i](hg)
              preds = self.outs[i](feature)
              combined_hm_preds.append(preds)
              if i < self.nstack - 1:
                  x = x + self.merge_preds[i](preds) + self.merge_features[i](feature)
          return torch.stack(combined_hm_preds, 1)
```
- `x = imgs.permute(0, 3, 1, 2)` $\Large \Rightarrow$ convert __(b, h, w, c)__ image to __(b, c, h, w)__

- `x = self.pre(x)` <br>
$\Large \Rightarrow$ nn.Sequential(<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; __Conv(3, 64, 7, 2, bn=True, relu=True)__, <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; __Residual(64, 128)__, <br> 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; __Pool(2, 2)__, <br> 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; __Residual(128, 128)__, <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; __Residual(128, inp_dim)__ <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; )<br>

  - __Conv(inp_dim=3, out_dim=64, kernel_size=7, stride=2, bn=True, relu=True)__
    - `def __init__(...):` 
      - `if relu:`
        - `self.relu = nn.ReLU()`
      - `if bn:`
        - `self.bn = nn.BatchNorm2d(out_dim)`
    - `def forward(self, x):`
      - __x.shape = (b, 3, h, w)__
      - `x = self.conv(x) = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, bias=True)(x)`
      - `x = self.bn(x)`
      - `x = self.relu(x)`
      - `return x` <br>
      $\Large \Rightarrow$ __x.shape = (b, 64, $\Large\lfloor{{{h-1}\over{2}} + 1}\rfloor$, $\Large\lfloor{{{w - 1}\over{2}} + 1}\rfloor$) = (b, out_dim, h', w')__

  - __Redisual(inp_dim, out_dim)__
    - `def __init__(...):` 
      - `if inp_dim == out_dim:`
        - `self.need_skip = False`
      - `else:`
        - `self.need_skip = True`
    - `def forward(self, x):`
      - __x.shape = (b, 64, h', w')__
      - `if self.need_skip:`
        - `redisual = self.skip_layer(x)` = Conv(inp_dim, out_dim, 1, relu=False) $\Large \Rightarrow$ __shape = (b, 128, h', w')__
      - `else:`
        - `residual = x` $\Large \Rightarrow$ __shape = (b, 64, h', w')__
      - `out = self.bn1(x)` = nn.BatchNorm2d(inp_dim)
      - `out = self.relu(out)` = nn.ReLU()
      - `out = self.conv1(out)` = Conv(inp_dim, int(out_dim/2), 1, relu=False) $\Large \Rightarrow$ __shape = (b, 32, h', w')__
      - `out = self.bn2(out)` = nn.BatchNorm2d(int(out_dim/2))
      - `out = self.relu(out)` = nn.ReLU()
      - `out = self.conv2(out)` = Conv(int(out_dim/2), int(out_dim/2), 3, relu=False) $\Large \Rightarrow$ __shape = (b, 32, h', w')__
      - `out = self.bn3(out)` = nn.BatchNorm2d(int(out_dim/2))
      - `out = self.relu(out)` = nn.ReLU()
      - `out = self.conv3(out)` = Conv(int(out_dim/2), out_dim, 1, relu=False) $\Large \Rightarrow$ __shape = (b, 64, h', w')__
      - `out += residual` $\Large \Rightarrow$ __shape = (b, 128, h', w')__ or __shape = (b, 64, h', w')__
      - `return out`<br>
        $\Large \Rightarrow$ __inp_feature와 out_feature의 num of channels 은 다를 수 있지만, h와 w는 동일__

<br>


- `for i in range(self.nstack):` $\Large \Rightarrow$ nstack 만큼 반복  <br> 
&nbsp;&nbsp;&nbsp;&nbsp; `hg = self.hgs[i](x)` $\Large \Rightarrow$ self.hgs = __ModuleList[nn.Sequential(Hourglass(...), &nbsp;...)]__<br>
&nbsp;&nbsp;&nbsp;&nbsp; `feature = self.features[i](hg)` $\Large \Rightarrow$ self.features = __ModuList[nn.Sequential(Residual(...), Conv(...)), &nbsp;...]__ <br>
&nbsp;&nbsp;&nbsp;&nbsp; `preds = self.outs[i](feature)` $\Large \Rightarrow$ self.outs = __ModuleList[Conv(...), &nbsp;...]__<br>
&nbsp;&nbsp;&nbsp;&nbsp; `combined_hm_preds.append(preds)` <br>
&nbsp;&nbsp;&nbsp;&nbsp; `if i < self.nstack - 1:` <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; `x = x + self.merge_preds[i](preds) + self.merge_features[i](feature)` <br> 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $\Large \Rightarrow$ self.merge_preds = __ModuleList[Merge(inp_dim, inp_dim), &nbsp;...]__ , <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; self.merge_features = __ModuleList[Merge(out_dim, out_dim), &nbsp;...]__<br>

  - `hgs = self.hgs(x) = ModuleList([nn.Sequential(Hourglass(4, inp_dim, bn, increase)) for i in range(nstack)])`<br>
    ```python
    class Hourglass(nn.Module):<br>
        def __init__(self, n=4, f=inp_dim, bn=None, increase=0):
            super(Hourglass, self).__init__()
            nf = f + increase 
            self.up1 = Residual(f, f) 
            # Lower branch 
            self.pool1 = Pool(2, 2) 
            self.low1 = Residual(f, nf) 
            self.n = n
            # Recursive hourglass 
            if self.n > 1: 
                self.low2 = Hourglass(n-1, nf, bn=bn) 
            else: 
                self.low2 = Residual(nf, nf)
            self.low3 = Residual(nf, f)
            self.up2 = nn.Upsample(scale_factor=2, mode='nearest')

        def forward(self, x):       # x      = (b, inp_dim, h, w)
            up1  = self.up1(x)      # up1    = (b, inp_dim, h, w)
            pool1 = self.pool1(x)   # pool1  = (b, inp_dim, h', w')
            low1 = self.low1(pool1) # low1   = (b, inp_dim, h', w')
            low2 = self.low2(low1)  # low2   = (b, inp_dim, h', w')
            low3 = self.low3(low2)  # low3   = (b, inp_dim, h', w')
            up2  = self.up2(low3)   # up1    = (b, inp_dim, h, w)
            return up1 + up2        # return = (b, inp_dim, h, w)
    ```


   - `features = self.features(hgs) = ModuleList([nn.Sequential(Residual(), Conv()) for i in range(nstack)])`
   
     - ```python
       nn.Sequential(
           Residual(inp_dim, inp_dim),
           Conv(inp_dim, inp_dim, 1, bn=True, relu=True)
         )
       ```

      - Residual( inp_dim, &nbsp;inp_dim ).forward(x) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; : &nbsp; __inp = (b, inp_dim, h, w) &nbsp; $\Large \Rightarrow$ &nbsp; out = (b, inp_dim, h, w)__
      - Conv( inp_dim, &nbsp;inp_dim, 1, bn=True, relu=True ) &nbsp; : &nbsp; __inp = (b, inp_dim, h, w) &nbsp; $\Large \Rightarrow$ &nbsp; out = (b, inp_dim, h, w)__
  
  - `preds = self.outs(features) = ModuleList([Conv(inp_dim, out_dim, 1, relu=False, bn=False) for i in range(nstack)])`
    - 1X1 Convolution
    - __inp = (b, inp_dim, h, w) &nbsp; $\Large \Rightarrow$ &nbsp; out = (b, out_dim, h, w)__

  -  `combined_hm_preds.append(preds)` = List[ preds=( b, out_dim, h, w ) ]

  - `if i < self.nstack - 1: x = x + self.merge_preds[i](preds) + self.merge_features[i](feature)`
    - Merge( x_dim, y_dim) = __Conv( x_dim, y_dim, 1, bn=False, relu=False )__
  
    - self.merge_preds(preds) = ModuleList([Merge(out_dim, inp_dim) for _ in range(nstack-1)]) <br>
    __inp = (b, out_dim, h, w) $\Large \Rightarrow$ out = (b, inp_dim, h, w)__ <br>

    - self.merge_features(preds) = ModuleList([Merge(inp_dim inp_dim) for _ in range(nstack-1)]) <br>
    __inp = (b, inp_dim, h, w) $\Large \Rightarrow$ out = (b, inp_dim, h, w)__ <br>
    - x = x + self.merge_predbs[ i ][ preds ] + self.merge_features[ i ][ preds ] <br>
      $\Large \Rightarrow$ __x = ( b, inp_dim, h, w )__

<br>

- `return torch.stack(combined_hm_preds)` <br>
  $\Large \Rightarrow$ combined_hm_preds = __[ ( b, inp_dim, h, w ), ... ( nstack ) ]__ <br>
  $\Large \Rightarrow$ torch.stack(combined_hm_preds) = __(b, nstack, inp_dim, h, w)__

<br><br>

## ✭ &nbsp; Model Shape Summary
### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; inp = <span style="color:yellow">( b, h, w, c )</span>
### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $\Large \Rightarrow$ &nbsp; perm = x.permute( inp ) = <span style="color:yellow">( b, c, h, w )</span>
### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $\Large \Rightarrow$ &nbsp; pre = x.pre( perm ) = <span style="color:yellow">( b, inp_dim, h'', w'' ) , &nbsp; (h'', w'' = ($\Large\lfloor{{{h-1}\over{2}} + 1}\rfloor \over 2$, $\Large\lfloor{{{w - 1}\over{2}} + 1}\rfloor \over 2$))</span>
### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $\Large \Rightarrow$ &nbsp; for &nbsp;i &nbsp;in &nbsp;range( nstack ) :  
### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $\large \rightarrow$ &nbsp; hg &nbsp;=&nbsp; hg[ i ] ( pre ) : ModuleList &nbsp;=&nbsp; <span style="color:yellow">( b, inp_dim, h'', w'' )</span>
### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $\large \rightarrow$ &nbsp; feature &nbsp;=&nbsp; features[ i ] ( hg ) : ModuleList &nbsp;=&nbsp; <span style="color:yellow">( b, inp_dim, h'', w'' )</span>
### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $\large \rightarrow$ &nbsp; preds &nbsp;=&nbsp; outs[ i ] ( feature ) : ModuleList &nbsp;=&nbsp; <span style="color:yellow">( b, out_dim, h'', w'' )</span>
### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $\large \rightarrow$ &nbsp; combined_hm_preds.append ( preds )

### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $\large \rightarrow$ &nbsp; if &nbsp; i &nbsp; < &nbsp; nstack - 1:
### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $\large \rightarrow$ &nbsp; x &nbsp;=&nbsp; x &nbsp;+&nbsp; merge_predbs[ i ] ( preds ) &nbsp;+&nbsp; merge_features[ i ] (feature)  
### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; merge_predbs[ i ] ( preds ) &nbsp;=&nbsp; <span style="color:yellow">( b, inp_dim, h'', w'' )</span>
### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; merge_features[ i ] ( feature ) &nbsp;=&nbsp; <span style="color:yellow">( b, inp_dim, h'', w'' )</span>
### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $\large \rightarrow$ &nbsp; ...
### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $\large \rightarrow$ &nbsp; return &nbsp;torch.stack(combined_hm_preds, 1)
### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $\Large \Rightarrow$ &nbsp; return &nbsp;torch.stack(combined_hm_preds, 1) &nbsp;=&nbsp; <span style="color:yellow">(b, n_stack, inp_dim, h'', w'')</span>

<br>

## 🔍 Loss
```python
class HeatmapLoss(torch.nn.Module):
    """
    loss for detection heatmap
    """
    def __init__(self):
        super(HeatmapLoss, self).__init__()

    def forward(self, pred, gt):
        l = ((pred - gt)**2)
        l = l.mean(dim=3).mean(dim=2).mean(dim=1)
        return l ## l of dim bsize
```
Loss로는 MSE(Mean Squared Error)를 사용함.<br>
pred heatmap과 gt heatmap 간의 차이를 제곱하고, 이를 평균하는 방식으로 Loss를 계산하고 있음을 볼 수 있음.

