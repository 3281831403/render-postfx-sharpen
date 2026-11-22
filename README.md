加了锐化之后，噪点跟着一起被放大，亮边还出现白边：local_min_max 只扫上下左右四个邻居，斜向边缘的极值整块漏掉；contrast_adaptive_weight 用线性映射代替平滑曲线，低对比区域被过度锐化；unsharp 的高斯核固定成 3×3，radius 参数形同虚设；unsharp 用原图减模糊图做高提升，符号反了，边缘越锐越糊；clamp_overshoot 只钳上限不钳下限，暗边一路溢出成黑边；sharpen 拿有符号细节去和 threshold 比较，负向边缘被整段跳过；noise_gain 用 amount² 估噪声增益，报告值系统性偏低；sharpen 直接在 sRGB 域里运算，亮度与锐度非线性地耦合在一起；validate_params 不检查 amount 上限，过度锐化参数被悄悄放行；sharpen_batch 复用上一帧的模糊缓冲，第二帧实际接着第一帧的结果在算。sharpen/cas.py 是核心，tests/test_cas.py 15个红。要修的是：补齐 3×3 八邻域极值、平滑的对比度自适应权重、随 radius 缩放的高斯核、正确的高提升符号、双端过冲钳制、基于局部对比度的阈值判断、正确的噪声增益公式、线性光域处理、amount 上限校验和逐帧独立的模糊缓冲。

## 运行方式

在仓库根目录执行：

```text
python -m pytest tests -q
```

初始状态必须恰好是 `15 failed`，不能出现 collection error、import error 或 syntax error。修好 `sharpen/cas.py` 后应得到 `15 passed`。环境为 Python 3 + pytest，只允许使用标准库 `math`，离线运行、不使用随机数，浮点比较容差为 `1e-9`。

## 字段表

| 字段 | 取值 |
| --- | --- |
| 所属领域 | 游戏、图形与多媒体 |
| 子方向 | 图形渲染 |
| 任务类型 | Bugfix |
| 语言/框架 | Python 3, pytest |
| 环境可复现等级 | 无外部依赖 |

## 约束

`tests/test_cas.py` 不可修改。它是验收契约：15 个测试分别覆盖上述 10 个缺陷，以及 amount=0 恒等、强边缘梯度增大、输出不越出局部邻域、噪声区增量小于边缘区、批处理与逐帧单独处理一致这 5 条回归要求。`sharpen/cas.py` 中 8 个函数的 docstring 写明了完整契约（sRGB 0..1、线性光 0..1、锐化权重 0..1、容差 1e-9），实现时不得改动函数签名。
