# V2 实验：M2-wave0-residual 最小闭环与消融结果

## 1. 实验目标

本轮实验根据 `docs/v2.md` 的优化方向，先实现并跑通最小闭环版 `M2-wave0-residual`，然后补充两组消融实验：

| 模型 | 输入 | 输出方式 | 目的 |
| --- | --- | --- | --- |
| M2-direct | 过去 24 小时风场 + ERA5 未来目标时效风场 | 直接预测未来波浪场 | 检验未来风场输入的贡献 |
| M2-wave0-direct | 过去 24 小时风场 + ERA5 未来目标时效风场 + t0 波浪场 | 直接预测未来波浪场 | 检验当前海况输入的贡献 |
| M2-wave0-residual | 过去 24 小时风场 + ERA5 未来目标时效风场 + t0 波浪场 | 预测相对 t0 波浪场的增量 | 检验残差预测是否更稳 |

本轮实验仍是快速验证版本，不是业务预报版本。未来风场使用的是 ERA5 真实再分析风场，因此它表示“理想未来风场驱动”条件下的上限实验；后续若要接近真实预报，需要把未来风场换成数值天气预报风场。

## 2. 数据与任务设置

数据目录：

```text
E:\PyCharm_WorkSpace\pangu\Wind_Wave\data\2025
```

输入区域为西北太平洋风场：

```text
5-45N, 95-150E
```

输出区域为中国近海波浪场：

```text
15-40N, 105-135E
```

输入变量：

```text
past_wind:   过去 24 小时逐小时 u10, v10
future_wind: +6h, +12h, +24h, +48h, +72h 的 u10, v10
wave0:       t0 时刻 SWH, MWP, cos(MWD), sin(MWD)
```

输出变量：

```text
SWH, MWP, cos(MWD), sin(MWD)
```

预测时效：

```text
+6h, +12h, +24h, +48h, +72h
```

## 3. 本轮实现内容

代码新增了 V2 数据流和模型分支：

1. `WindWaveSeq2SeqDataset` 新增返回 `future_wind` 和 `wave0`。
2. 新增 `WindWaveV2Model`，包含历史风场 ConvLSTM 编码、未来风场 CNN 编码、当前波浪场 CNN 编码和多时效预测头。
3. `train.py` 新增 `--model-variant` 和 `--run-name`：
   - `m1`
   - `m2-direct`
   - `m2-wave0-direct`
   - `m2-wave0-residual`
4. `evaluate.py` 可以从 checkpoint 读取模型类型，并把测试指标写入对应 run 目录。
5. 修复了两个真实数据烟测暴露出的鲁棒性问题：
   - 某个 lead 没有有限波向格点时，指标记为 `NaN` 而不是中断训练。
   - `wave0` 中 ERA5 陆地或缺测 NaN 在进入模型和残差相加前替换为归一化空间的 0，避免 NaN 污染损失。

## 4. 运行命令

烟测命令：

```powershell
& 'C:\Users\SLDUO\anaconda3\envs\pangu\python.exe' Wind_Wave/train.py `
  --epochs 1 `
  --batch-size 4 `
  --max-samples 4 `
  --spatial-stride 16 `
  --hidden-channels 8 `
  --num-workers 0 `
  --preload-spatial `
  --model-variant m2-wave0-residual `
  --run-name smoke_m2_wave0_residual_v4
```

正式快速实验统一使用：

```powershell
--epochs 5
--batch-size 8
--spatial-stride 16
--hidden-channels 16
--num-workers 0
--preload-spatial
```

三个 run 名称为：

```text
v2_m2_wave0_residual
v2_m2_direct
v2_m2_wave0_direct
```

每组训练后均使用 best checkpoint 运行 `evaluate.py`，测试指标保存到：

```text
Wind_Wave\outputs\<run-name>\logs\test_metrics_by_lead.csv
```

## 5. 烟测结果

`M2-wave0-residual` 最小烟测完整跑通，用时约 147 秒。结果如下：

| 项目 | 数值 |
| --- | ---: |
| train_loss | 0.4343 |
| val_loss | 0.4544 |

烟测已写出：

```text
checkpoints/seq2seq_convlstm_best.pt
logs/train_log.csv
logs/metrics_by_lead.csv
logs/baseline_metrics_by_lead.csv
logs/training_curve.png
samples/predictions_preview.npz
```

## 6. 正式快速实验结果

### 6.1 验证集损失

| 模型 | best epoch | best val loss | final val loss |
| --- | ---: | ---: | ---: |
| M2-wave0-residual | 2 | 0.359835 | 0.378129 |
| M2-direct | 5 | 0.496198 | 0.496198 |
| M2-wave0-direct | 3 | 0.349803 | 0.362643 |

从验证损失看，`M2-wave0-direct` 最好，`M2-wave0-residual` 次之，`M2-direct` 明显较弱。说明仅加入未来风场还不够，当前波浪场 `wave0` 对模型很关键。

### 6.2 测试集平均指标

下表为 5 个 lead time 的平均测试指标。SWH 和 MWP 为 RMSE，MWD 为角度 MAE，越低越好。

| 模型 | SWH 平均 RMSE | MWP 平均 RMSE | MWD 平均 MAE |
| --- | ---: | ---: | ---: |
| persistence | 0.758 | 0.949 | 36.88 |
| M2-direct | 0.501 | 0.813 | 21.57 |
| M2-wave0-residual | 0.459 | 0.680 | 21.47 |
| M2-wave0-direct | 0.436 | 0.655 | 20.82 |

相对 persistence 的平均提升：

| 模型 | SWH 提升 | MWP 提升 | MWD 提升 |
| --- | ---: | ---: | ---: |
| M2-direct | 33.9% | 14.3% | 41.5% |
| M2-wave0-residual | 39.5% | 28.4% | 41.8% |
| M2-wave0-direct | 42.5% | 30.9% | 43.5% |

### 6.3 分时效测试结果

SWH RMSE：

| lead | persistence | M2-direct | M2-wave0-residual | M2-wave0-direct |
| ---: | ---: | ---: | ---: | ---: |
| +6h | 0.336 | 0.506 | 0.283 | 0.296 |
| +12h | 0.558 | 0.500 | 0.396 | 0.383 |
| +24h | 0.807 | 0.466 | 0.490 | 0.448 |
| +48h | 1.012 | 0.506 | 0.549 | 0.511 |
| +72h | 1.077 | 0.528 | 0.577 | 0.542 |

MWP RMSE：

| lead | persistence | M2-direct | M2-wave0-residual | M2-wave0-direct |
| ---: | ---: | ---: | ---: | ---: |
| +6h | 0.447 | 0.815 | 0.395 | 0.438 |
| +12h | 0.717 | 0.785 | 0.594 | 0.560 |
| +24h | 1.020 | 0.762 | 0.725 | 0.687 |
| +48h | 1.240 | 0.828 | 0.815 | 0.787 |
| +72h | 1.320 | 0.874 | 0.868 | 0.804 |

MWD MAE：

| lead | persistence | M2-direct | M2-wave0-residual | M2-wave0-direct |
| ---: | ---: | ---: | ---: | ---: |
| +6h | 13.80 | 20.33 | 12.55 | 14.82 |
| +12h | 25.33 | 21.00 | 18.87 | 19.78 |
| +24h | 40.65 | 21.49 | 22.55 | 21.06 |
| +48h | 53.51 | 22.38 | 26.80 | 23.60 |
| +72h | 51.09 | 22.64 | 26.56 | 24.87 |

## 7. 结果解读

第一，`wave0` 是短时效预测的关键输入。`M2-direct` 没有当前波浪场，在 +6h 的 SWH、MWP、MWD 全部差于 persistence；加入 `wave0` 后，`M2-wave0-residual` 和 `M2-wave0-direct` 在 +6h 的 SWH、MWP 明显改善。

第二，残差版确实改善了最短时效。`M2-wave0-residual` 在 +6h 的 SWH、MWP 和 MWD 都是三组 M2 模型里最好的，说明“在当前海况基础上预测变化量”对短时效有帮助。

第三，5 个时效平均来看，`M2-wave0-direct` 当前最好。它的平均 SWH、MWP 和 MWD 指标均优于 residual 版本。这说明当前残差设计还没有完全发挥预期优势，尤其在 +24h 到 +72h 上，直接预测反而更稳。

第四，`M2-direct` 在中长期仍然有价值。虽然它短时效较差，但在 +24h 到 +72h 的 SWH 和 MWD 上明显优于 persistence，说明 ERA5 未来目标时效风场确实提供了有效驱动信息。

第五，本轮 V2 相比前期 M1 有明显提升。前期 M1 测试集 5 个时效平均约为 SWH 0.655、MWP 0.901、MWD 33.86。本轮最佳的 `M2-wave0-direct` 达到 SWH 0.436、MWP 0.655、MWD 20.82。不过需要注意，M1 是前期 20 epoch 历史风场模型，V2 是本轮 5 epoch 快速实验，二者训练设置不完全相同，只适合作趋势参考。

## 8. 需要继续优化的地方

1. 当前 residual 对 `cos(MWD)` 和 `sin(MWD)` 直接做残差相加，可能破坏单位圆约束。后续可以对波向单独设计角度损失，或者输出角度变化再还原。
2. 当前损失仍是四个变量等权的归一化 MSE。短时效如果是重点，可以尝试给 +6h、+12h 更高权重。
3. 当前只训练 5 个 epoch，且验证损失在第 2 到第 3 个 epoch 后开始波动，后续应加入 early stopping、weight decay 和学习率调度。
4. 当前未来风场只取 5 个目标时效点。后续可以把未来风场扩展为 +6h 到 +72h 的连续 6 小时间隔序列，让模型看到更完整的风场强迫过程。
5. 当前空间步长为 16，只适合快速实验。确认结构后应逐步试 `spatial-stride=8`，再考虑更高分辨率。
6. 当前只用 2025 年数据，样本季节和天气型有限。后续应扩展多年数据，特别是台风、冬季风和季节转换期样本。

## 9. 下一步建议

建议下一步优先走 `M2-wave0-direct` 作为当前主线，同时保留 `M2-wave0-residual` 做短时效专项优化：

1. 将 `M2-wave0-direct` 训练到 10-20 epoch，加入 early stopping，确认它是否稳定优于 residual。
2. 对 `M2-wave0-residual` 改进波向处理和 lead 加权损失，重点优化 +6h 和 +12h。
3. 用相同设置复跑 M1、M2-direct、M2-wave0-direct、M2-wave0-residual，形成严格可比的正式消融表。
4. 在低分辨率确认后，把 `spatial-stride` 从 16 降到 8，检查空间细节是否明显改善。

本轮最重要的结论是：加入当前波浪场 `wave0` 后，短时效问题明显缓解；加入未来 ERA5 风场后，中长期波浪预测也明显优于 persistence。残差方案在 +6h 最有优势，但当前整体最稳的是 `M2-wave0-direct`。
