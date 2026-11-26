# Causal Forcing 长视频实验

`main` 对应复现使用的 [Causal Forcing 上游提交](https://github.com/thu-ml/Causal-Forcing/tree/52c39015bcb307da785b476822aefce4488ea342)。`ybr` 的实验使用同一份五秒生成器权重，比较普通滑窗、Deep Forcing 及加入帧多样性和单帧限额后的 30 秒生成结果。

| 实验 | 记录 |
|---|---|
| 原论文复现 | [运行条件与逐提示词结果](复现记录.md) |
| 同权重滑窗与官方长视频权重 | [五种方法配对比较](official-long/对照结果.md) |
| 帧级多样性 | [分段诊断](segment-diagnostic/分段诊断.md) |
| 单帧限额与 15%、50% 系数对照 | [方法、结果、复现入口](quota-study/对照实验.md) |
| 乘法调制 | [分数与失败案例](multiplicative-diagnostic/结果.md) |
| 30 层缓存分析 | [逐层体检](attention-audit/体检结果.md) |

各实验目录保存逐视频分数、运行记录和样例。四个大型归档可从 [实验数据 Release](https://github.com/yan-borui/Causal-Forcing/releases/tag/experiment-data) 下载。
