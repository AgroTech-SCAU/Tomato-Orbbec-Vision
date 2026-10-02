# 番茄成熟度智能检测与性能评估系统V1.0模型训练记录

> 文档状态：训练完成，训练侧数据已回填  
> 适用范围：软著V1.0所需的E13参数PyTorch模型训练  
> 不包含：ONNX、TensorRT、INT8、剪枝、蒸馏、多硬件Benchmark、RGB-D定位和机械臂控制

---

## 1. 文档与实验信息

| 项目 | 内容 |
|---|---|
| 文档版本 | V1.0（软著V1.0训练完成版） |
| 创建/更新日期 | 2026-09-21 |
| 项目名称 | 番茄成熟度智能检测与性能评估系统V1.0 |
| 项目根目录 | `E:\TOMATO\benchmark` |
| 实验编号 | E13 |
| 训练输出名称 | `v18_e13_mosaic025` |
| 当前状态 | 训练已完成；因`patience=40`在第95个epoch正常早停，最佳结果位于epoch 55，训练侧验收通过 |
| 执行人 | 王凯鑫 |
| 审核人 | 王凯鑫 |

## 2. V1.0目标与训练边界

### 2.1 本次训练目标

使用固定的Roboflow v18数据集、`yolo11m-seg.pt`初始权重和E13训练参数，产出可供软著V1.0软件稳定加载的PyTorch实例分割模型`best.pt`。由于当前设备使用`batch=4`时发生CUDA显存不足、多进程数据加载时发生OpenCV系统内存分配失败，本次V1.0训练将实际批次调整为`batch=2`、数据加载进程数调整为`workers=0`；输入尺寸及其他训练参数保持不变。

V1.0以“软件可运行、可演示、流程完整”为验收重点，不以论文级算法创新、极限精度或多后端性能对比为前置条件。成熟与未成熟番茄是主要业务类别，果梗作为候选区域辅助识别结果。

### 2.2 训练产物与软件功能的关系

| 内容 | 是否由本次训练直接提供 | V1.0要求 |
|---|---|---|
| E13参数PyTorch模型`best.pt` | 是 | 必须 |
| 训练参数、loss、valid基础指标和曲线 | 是 | 用于证明模型来源和基本可用性 |
| 图片、文件夹、视频、摄像头检测 | 否，由软件推理代码实现 | 必须 |
| 成熟、未成熟、果梗结果显示 | 模型提供三类输出，显示由软件实现 | 必须 |
| 数量和成熟率统计 | 否，由软件业务代码实现 | 必须 |
| 结果保存与导出 | 否，由软件业务代码实现 | 必须 |
| 阈值和类别显示开关 | 否，由软件界面实现 | 建议 |
| 历史记录、日志和错误提示 | 否，由软件业务代码实现 | 建议 |
| 基础推理时间和FPS | 否，由软件运行时统计 | 建议 |

### 2.3 明确延期到后续版本

- ONNX Runtime部署；
- TensorRT FP32、FP16和INT8；
- 正式的多后端或跨硬件Benchmark；
- INT8校准、模型剪枝和知识蒸馏；
- RGB-D三维定位、采摘点计算和机械臂控制；
- 论文级多随机种子或交叉验证实验。

以上内容缺失不影响软著V1.0模型训练和软件登记。

## 3. 训练环境

| 项目 | 固定值 |
|---|---|
| Python解释器 | `D:\envs\yolo\python.exe` |
| Python | 3.10.20 |
| PyTorch | 2.11.0+cu128 |
| torchvision | 0.26.0+cu128 |
| Ultralytics | 8.4.93 |
| NumPy | 2.2.6 |
| OpenCV | 5.0.0.93 |
| Pillow | 12.2.0 |
| PyYAML | 6.0.3 |
| 操作系统 | Windows 10 Home China，25H2，Build 26200.8875 |
| CPU | AMD Ryzen 9 8945HX，16核32线程 |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU，8151 MiB |
| NVIDIA驱动 | 592.01 |
| PyTorch CUDA Runtime | 12.8 |
| 当前电源方案 | 性能；插电 |

## 4. 数据集

### 4.1 数据身份

| 项目 | 内容 |
|---|---|
| 数据集 | Roboflow `tomato-h72eq` v18 |
| 任务 | 实例分割 |
| 数据目录 | `E:\TOMATO\benchmark\dataset` |
| 配置文件 | `E:\TOMATO\benchmark\data.yaml` |
| 配置文件SHA256 | `E22B9F898AD9F35A64B8CCACCA4A374D4669C2622CECC73B647C5399EA9AF381` |
| 图像尺寸 | 768×768 |
| 图像格式 | JPEG |
| 类别顺序 | `0 ripe`、`1 stem`、`2 unripe` |
| 数据许可 | CC BY 4.0 |

### 4.2 划分与实例数

| 划分 | 图片 | 标签 | ripe | stem | unripe | 总实例 |
|---|---:|---:|---:|---:|---:|---:|
| train | 763 | 763 | 6,420 | 659 | 3,515 | 10,594 |
| valid | 96 | 96 | 806 | 81 | 524 | 1,411 |
| test | 96 | 96 | 741 | 82 | 559 | 1,382 |
| 合计 | 955 | 955 | 7,967 | 822 | 4,598 | 13,387 |

### 4.3 训练前自动检查

`train.py`在加载GPU模型前检查：

- 三个划分的图片和标签数量；
- 图片与标签是否一一对应；
- 所有图片是否为768×768；
- 标签是否为空；
- 类别ID是否只包含0、1、2；
- 标签是否为合法分割多边形而非五列检测框；
- 坐标是否位于`[0,1]`；
- 每个划分的三类实例数是否与上述冻结统计一致。

已知限制：历史检查发现23个跨划分同源切片组，因此本数据集可能存在近重复或场景泄漏风险。该限制需要在说明书中客观披露，但不阻止V1.0软件完成。

## 5. 模型初始化

| 项目 | 固定值 |
|---|---|
| 任务 | 实例分割`segment` |
| 架构 | YOLO11m-seg |
| 初始权重 | `E:\TOMATO\benchmark\yolo11m-seg.pt` |
| 初始权重大小 | 45,400,152 bytes |
| 初始权重SHA256 | `EB9A06F63E2206C35D68D839B08C362429EBECF933AD54C1AD68B2FD001C17CF` |
| 输入通道 | 3 |
| 输出类别 | 3 |
| 冻结层 | `freeze=null`，不主动冻结网络层 |
| 模型结构修改 | 无 |

## 6. E13训练参数

参数的权威来源是`train.py`中的`e13_training_arguments()`；训练结束后以自动生成的`args.yaml`复核实际值。

| 参数 | 值 | 参数 | 值 |
|---|---:|---|---:|
| epochs | 200 | patience | 40 |
| batch | 2 | imgsz | 768 |
| device | `0` | workers | 0 |
| optimizer | SGD | lr0 | 0.01 |
| lrf | 0.01 | momentum | 0.937 |
| weight_decay | 0.0005 | nbs | 64 |
| seed | 42 | deterministic | true |
| amp | true | cos_lr | true |
| warmup_epochs | 3.0 | warmup_momentum | 0.8 |
| warmup_bias_lr | 0.1 | close_mosaic | 30 |
| mask_ratio | 2 | overlap_mask | true |
| degrees | 10.0 | translate | 0.1 |
| scale | 0.25 | shear | 0.0 |
| perspective | 0.0 | fliplr | 0.5 |
| flipud | 0.0 | mosaic | 0.25 |
| mixup | 0.0 | cutmix | 0.0 |
| copy_paste | 0.0 | multi_scale | 0.0 |
| hsv_h | 0.01 | hsv_s | 0.4 |
| hsv_v | 0.3 | cache | false |
| val | true | split | val |
| iou | 0.70 | max_det | 300 |
| save | true | plots | true |
| project | `E:\TOMATO\benchmark\runs\segment` | name | `v18_e13_mosaic025` |
| exist_ok | false | pretrained | true |

训练脚本：`E:\TOMATO\benchmark\train.py`  
训练脚本SHA256：`C8303EB8EA00137AA2D87D3B64B155FE4EC0E110347B5BD9E0A18594F0BED4CC`

## 7. 运行方式

首次训练：

```powershell
D:\envs\yolo\python.exe E:\TOMATO\benchmark\train.py
```

仅检查数据和参数，不训练：

```powershell
D:\envs\yolo\python.exe E:\TOMATO\benchmark\train.py --check-only
```

中断后从同一实验的`last.pt`恢复：

```powershell
D:\envs\yolo\python.exe E:\TOMATO\benchmark\train.py --resume
```

脚本拒绝覆盖已有的运行目录；若需要从已有检查点继续训练，可使用`--resume`。

## 8. 训练完成后自动产生的证据

| 产物 | 实际路径 | 用途 |
|---|---|---|
| 最佳权重 | `runs/segment/v18_e13_mosaic025/weights/best.pt` | V1.0正式候选模型 |
| 最后权重 | `runs/segment/v18_e13_mosaic025/weights/last.pt` | 中断恢复和训练档案 |
| 实际参数 | `runs/segment/v18_e13_mosaic025/args.yaml` | 核对实际训练配置 |
| 逐epoch结果 | `runs/segment/v18_e13_mosaic025/results.csv` | loss、学习率和valid指标 |
| 总体曲线 | `runs/segment/v18_e13_mosaic025/results.png` | 收敛情况 |
| Box/Mask PR曲线 | 实验目录中的`BoxPR_curve.png`、`MaskPR_curve.png` | 基础精度证据 |
| Box/Mask F1曲线 | 实验目录中的`BoxF1_curve.png`、`MaskF1_curve.png` | 阈值参考 |
| 混淆矩阵 | `confusion_matrix.png`及标准化版本 | 类别错误分析 |
| 训练批次图 | `train_batch*.jpg` | 数据与增强检查 |
| 验证图 | `val_batch*_labels.jpg`、`val_batch*_pred.jpg` | 直观效果检查 |
| 训练记录 | `training_run_record.json` | 时间、路径、参数及主要文件哈希 |

`training_run_record.json`由`train.py`在训练成功返回后生成，因此训练完成后，本文件第9节所需的时间、路径、配置和权重哈希都有来源可查。

## 9. 训练完成结果

以下字段必须使用本次新生成的文件填写，不得复制历史E13结果。

| 项目 | 本次结果 | 数据来源 |
|---|---|---|
| 训练开始时间 | 2026-09-20 18:05:18（UTC+08:00） | `training_run_record.json` |
| 训练结束时间 | 2026-09-20 21:58:00（UTC+08:00） | `training_run_record.json` |
| 总训练时间 | 13,963.59秒（约3小时52分44秒；终端显示3.875小时） | `training_run_record.json`、`test.md` |
| 计划epoch | 200 | `args.yaml` |
| 实际完成epoch | 95 | `training_run_record.json`中的`training_results` |
| 最佳epoch | 55 | `training_run_record.json`中的`training_results` |
| 是否早停 | 是；连续40个epoch无更优结果，属于`patience=40`的正常行为 | `training_run_record.json`、`test.md` |
| 最佳fitness | 1.38471 | `training_run_record.json`中的`training_results` |
| best.pt大小 | 45,191,414 bytes（约43.10 MiB；终端显示45.2 MB） | `training_run_record.json` |
| best.pt SHA256 | `CB05ED52B54E1060CDA6C04294227C8589AA3AB4B2175A759E4F15CB68671959` | `training_run_record.json` |
| last.pt大小 | 45,191,414 bytes（约43.10 MiB） | `training_run_record.json` |
| last.pt SHA256 | `8C120FFE13C71E22E7E15388D6C1B7754C4AD5B741F05EEF5B846D914FBD266D` | `training_run_record.json` |
| 最低train loss | box 0.37710；cls 0.27406；dfl 0.83201；seg 0.58716 | `training_run_record.json`中的`minimum_losses` |
| 最低valid loss | box 0.46875；cls 0.43608；dfl 0.88468；seg 0.80131 | `training_run_record.json`中的`minimum_losses` |
| 最佳valid Box mAP50-95 | 0.72317 | `training_run_record.json`中的`training_results` |
| 最佳valid Mask mAP50-95 | 0.66154 | `training_run_record.json`中的`training_results` |
| 训练是否出现异常或恢复 | 本次最终运行未中断、未恢复；此前`batch=4`预跑曾发生CUDA显存不足，多进程加载曾发生OpenCV系统内存不足，调整为`batch=2`、`workers=0`并重启设备后重新完整训练 | 控制台记录和实际操作 |

### 9.1 最佳权重最终验证结果

以下为训练结束后Ultralytics自动加载`best.pt`在valid集（96张图片、1,411个实例）上的最终验证结果。该步骤也证明`best.pt`可被当前Ultralytics 8.4.93环境正常加载。

| 类别 | 图片 | 实例 | Box P | Box R | Box mAP50 | Box mAP50-95 | Mask P | Mask R | Mask mAP50 | Mask mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 96 | 1,411 | 0.811 | 0.840 | 0.882 | 0.722 | 0.808 | 0.842 | 0.883 | 0.665 |
| ripe | 81 | 806 | 0.844 | 0.933 | 0.956 | 0.866 | 0.845 | 0.937 | 0.957 | 0.822 |
| stem | 70 | 81 | 0.736 | 0.691 | 0.765 | 0.499 | 0.729 | 0.691 | 0.767 | 0.416 |
| unripe | 66 | 524 | 0.853 | 0.896 | 0.925 | 0.803 | 0.852 | 0.899 | 0.926 | 0.756 |

最终验证速度：预处理0.4 ms/图、推理11.8 ms/图、后处理1.9 ms/图。该速度只代表本机当前PyTorch/CUDA环境下的valid批量验证输出，不作为正式跨硬件Benchmark结论。

## 10. 模型进入V1.0的最低验收条件

### 10.1 训练侧

- [x] 数据审计通过；
- [x] 训练正常结束，早停属于`patience=40`的正常行为；
- [x] `best.pt`、`last.pt`、`args.yaml`和`results.csv`完整存在；
- [x] `best.pt`已由训练流程使用Ultralytics 8.4.93正常加载并完成最终valid验证；
- [x] valid曲线整体收敛，预测样例未见阻断V1.0使用的明显异常；
- [x] 已记录`best.pt`文件大小和SHA256；
- [x] 本次未运行test集评估，未使用test结果反向修改训练参数。

### 10.2 软件侧

- [ ] 软件默认加载本次验收后的`best.pt`；
- [ ] 能完成单张图片检测；
- [ ] 能完成文件夹批量检测；
- [ ] 能完成视频检测；
- [ ] 能完成摄像头检测；
- [ ] 能显示成熟、未成熟和果梗候选区域；
- [ ] 能统计成熟和未成熟番茄数量；
- [ ] 成熟率按`ripe / (ripe + unripe)`计算，果梗不进入分母；
- [ ] 能保存检测图片或视频；
- [ ] 能导出至少一种统计文件格式；
- [ ] 果梗在界面和说明书中表述为“候选区域辅助识别”；
- [ ] 软件名称、V1.0版本号、说明书和截图保持一致。

## 11. V1.0结论

### 11.1 模型训练侧结论

本次E13训练于2026-09-20完成，使用`batch=2`、`workers=0`和768×768输入。训练因`patience=40`在epoch 95正常早停，最佳结果位于epoch 55。最终选择`best.pt`作为“番茄成熟度智能检测与性能评估系统V1.0”的PyTorch推理模型，文件SHA256为`CB05ED52B54E1060CDA6C04294227C8589AA3AB4B2175A759E4F15CB68671959`。

该模型能够输出ripe、stem和unripe三类实例分割结果，其中stem仅作为果梗候选区域辅助信息。训练产物、参数、哈希、曲线和valid结果均已留档，模型训练侧满足软著V1.0开发与演示所需的最低验收条件。软件功能侧仍应按第10.2节单独验收；ONNX、TensorRT、量化和正式跨硬件Benchmark延期至V1.1及后续版本。

## 12. 签字确认

| 角色 | 姓名 | 日期 | 结论 |
|---|---|---|---|
| 实验执行人 | [待填写] | [待填写] | [待填写] |
| 模型审核人 | [待填写] | [待填写] | [待填写] |
| 软件V1.0审核人 | [待填写] | [待填写] | [待填写] |
