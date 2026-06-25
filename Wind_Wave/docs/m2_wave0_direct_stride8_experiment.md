# M2-wave0-direct spatial_stride=8 实验总结

记录时间：2026-06-20  
输出目录：`Wind_Wave/outputs/M2wave0__in_past24hWind+futureLeadWind+waveT0__direct_nores__s8_b1024_h64_bf16__ep20__y2016-2025`

## 1. 实验目的

本实验验证 `m2-wave0-direct` 模型在 `spatial_stride=8` 时，是否可以稳定学习“西北太平洋风场 -> 中国近海波浪场”的多时效预报任务。

模型输入过去 24 小时逐小时风场，预测未来 5 个时效的波浪场：

- 输入变量：`u10`, `v10`
- 输入区域：`5-45N, 95-150E`
- 输出变量：`swh`, `mwp`, `cos_mwd`, `sin_mwd`
- 输出区域：`15-40N, 105-135E`
- 预测时效：`+6h`, `+12h`, `+24h`, `+48h`, `+72h`

`m2-wave0-direct` 会额外使用当前时刻波浪场 `wave0` 作为初始海况信息，然后直接预测未来波浪场，不做 residual 差值预测。

## 2. 数据设置

本次训练使用已经转换好的 Zarr 缓存数据：

- 风场：`Wind_Wave/data/zarr/era5_wind_025_5N45N_95E150E.zarr`
- 波浪场：`Wind_Wave/data/zarr/era5_wave_050_5N45N_95E150E.zarr`
- 时间范围：2016-01-01 00:00 至 2025-12-31 23:00
- 总时长：87672 小时
- 风场网格：0.25 度，`161 x 221`
- 波浪网格：0.5 度，`81 x 111`

训练、验证、测试采用按时间顺序切分：

- 训练集约 70%
- 验证集约 15%
- 测试集约 15%

## 3. 训练参数

主要参数如下：

| 参数 | 值 |
|---|---:|
| model_variant | `m2-wave0-direct` |
| data_source | `zarr` |
| epochs | `20` |
| batch_size | `1024` |
| hidden_channels | `64` |
| spatial_stride | `8` |
| history_hours | `24` |
| lead_hours | `6,12,24,48,72` |
| learning_rate | `1e-3` |
| weight_decay | `1e-4` |
| dropout | `0.1` |
| early_stopping_patience | `3` |
| precision | `bf16` |

加速策略：

- 使用 BF16 混合精度，利用 Tensor Core
- 使用 Zarr 缓存作为数据源
- 使用 `fast-in-memory-dataset` 降低逐 batch 读取和预处理开销
- 使用空间降采样 `spatial_stride=8`，在精度和速度之间折中

## 4. 训练过程

训练共完成 20 轮，没有触发早停。

最优验证结果出现在第 19 轮：

| 指标 | 值 |
|---|---:|
| best_epoch | `19` |
| train_loss | `0.119178` |
| val_loss | `0.112539` |
| train_seconds | `43.28 s` |
| val_seconds | `6.99 s` |
| epoch_seconds | `50.27 s` |
| samples_per_second | `1416.44` |

第 20 轮最终结果：

| 指标 | 值 |
|---|---:|
| train_loss | `0.118232` |
| val_loss | `0.114914` |
| train_seconds | `43.13 s` |
| val_seconds | `6.92 s` |
| samples_per_second | `1421.51` |

从训练曲线看，前 5 轮下降最快，之后进入缓慢优化阶段。第 19 轮达到最低验证损失，第 20 轮验证损失略有回升，但幅度不大，属于正常波动。

## 5. 最优轮验证集指标

第 19 轮在验证集上的逐时效指标如下：

| Lead time | SWH RMSE (m) | MWP RMSE (s) | MWD MAE (deg) |
|---:|---:|---:|---:|
| +6h | 0.1694 | 0.3195 | 7.37 |
| +12h | 0.2139 | 0.4060 | 9.86 |
| +24h | 0.2589 | 0.5057 | 12.06 |
| +48h | 0.3014 | 0.6421 | 14.92 |
| +72h | 0.3275 | 0.7230 | 16.80 |

直观理解：

- 近时效 +6h 的浪高 RMSE 约 0.17 m，方向误差约 7.4 度。
- 到 +72h 时，浪高 RMSE 上升到约 0.33 m，方向误差约 16.8 度。
- 随着预报时效变长，误差增加是正常现象，但增长幅度比较平滑。

## 6. 与 persistence baseline 对比

Persistence baseline 指的是“直接把当前波浪场当成未来预报”。它是一个简单但很有参考价值的基线。

验证集上，`m2-wave0-direct` 相比 persistence baseline 有明显提升：

| Lead time | SWH RMSE: model / baseline | SWH 提升 | MWP RMSE: model / baseline | MWP 提升 | MWD MAE: model / baseline | MWD 提升 |
|---:|---:|---:|---:|---:|---:|---:|
| +6h | 0.1694 / 0.2381 | 28.8% | 0.3195 / 0.3823 | 16.4% | 7.37 / 9.56 | 22.9% |
| +12h | 0.2139 / 0.3983 | 46.3% | 0.4060 / 0.6068 | 33.1% | 9.86 / 17.15 | 42.5% |
| +24h | 0.2589 / 0.6046 | 57.2% | 0.5057 / 0.8822 | 42.7% | 12.06 / 28.49 | 57.7% |
| +48h | 0.3014 / 0.8035 | 62.5% | 0.6421 / 1.1535 | 44.3% | 14.92 / 41.67 | 64.2% |
| +72h | 0.3275 / 0.8657 | 62.2% | 0.7230 / 1.2487 | 42.1% | 16.80 / 46.29 | 63.7% |

按 5 个时效简单平均：

| 指标 | Model | Baseline | 平均提升 |
|---|---:|---:|---:|
| SWH RMSE | 0.2542 m | 0.5820 m | 56.3% |
| MWP RMSE | 0.5192 s | 0.8547 s | 39.3% |
| MWD MAE | 12.20 deg | 28.63 deg | 57.4% |

这个结果说明模型不是只在短时效上有效，而是在 +24h 到 +72h 的中长期时效上也明显优于 persistence baseline。

## 7. 结论

`spatial_stride=8` 是当前比较稳妥的一档实验设置。

它的优点：

- 训练速度仍然较快，稳定后约 `1400+ samples/s`
- 单轮训练加验证约 50 秒
- 验证集指标明显优于 persistence baseline
- 对波向 MWD 的提升尤其明显，长时效提升超过 60%

它的不足：

- 空间分辨率仍然经过降采样，不能代表最终高精度结果
- 当前 `metrics_by_lead.csv` 主要记录验证集逐时效指标，后续应补充 best checkpoint 在测试集上的正式评估
- 继续降低 stride 到 4 或 2 时，机器曾出现 WHEA 硬件级重启问题，因此高分辨率长时间训练前需要先确认硬件稳定性

当前建议：

- 把 `m2-wave0-direct, spatial_stride=8` 作为正式 baseline
- 下一步先补充 best checkpoint 的测试集评估
- 硬件稳定性确认后，再尝试 `spatial_stride=4`
- 如果 `stride=4` 继续导致系统重启，应优先排查 PCIe/WHEA、驱动、电源和散热，而不是先怀疑 Python 训练代码
