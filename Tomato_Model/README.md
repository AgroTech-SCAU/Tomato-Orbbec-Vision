# 番茄采摘机器人（视觉）— 模型训练与优化

本目录为 Tomato-Orbbec-Vision 项目的模型模块，提供番茄成熟度与果梗候选区域的实例分割训练、推理及实验记录。

## 最新稳定版本

- **Stable：** `v1.0.0`
- **Release：** [GitHub Releases](https://github.com/AgroTech-SCAU/Tomato-Orbbec-Vision/releases)
- **状态：** 稳定维护
- **当前模型方案：** YOLO11m-seg，E13 训练配置，Roboflow v18 数据集

> 本 README 以当前 V1.0 模型方案为准；后续功能见仓库根目录的 [docs/plan.md](../docs/plan.md) 与对应 Issue / PR。

## 1. 项目简介

本模块面向农业采摘场景，主要用于番茄模型的训练、验证与推理，为后续视觉系统提供目标类别、检测框和实例分割掩膜。

### 主要能力

- 识别成熟番茄与未成熟番茄。
- 输出果梗候选区域，作为辅助识别信息。
- 对单张图片、图片文件夹、视频和摄像头画面进行推理。
- 显示类别、置信度、检测框、分割掩膜及当前帧各类别数量。
- 保存标注后的图片或视频，并显示推理耗时。
- 自动检查训练数据，保存训练参数、结果曲线、权重和训练记录。

### 类别定义

| 类别 ID | 数据集名称 | 含义 |
|---|---|---|
| 0 | `ripe` | 成熟番茄 |
| 1 | `stem` | 果梗候选区域 |
| 2 | `unripe` | 未成熟番茄 |

`detect.py` 将类别 1 的显示名称改为 `main_peduncle`；训练标签仍使用 `stem`，类别顺序不变。

### 当前边界

果梗识别仅用于候选区域辅助提示，尚不支持果梗精确定位、精确切割点或采摘点计算。当前脚本不包含 Orbbec 深度流接入、RGB-D 三维定位、手眼标定、ROS 通信和机械臂控制。ONNX、TensorRT、量化及跨硬件性能评测留待后续版本。

## 2. 环境要求

### 软件

以下版本来自已有 E13 训练记录，可作为复现参考；其他环境尚需自行验证。

| 项目 | 已记录环境 |
|---|---|
| OS | Windows；详细环境见训练记录 |
| Python | 3.10.20 |
| PyTorch | 2.11.0+cu128 |
| torchvision | 0.26.0+cu128 |
| Ultralytics | 8.4.93 |
| NumPy | 2.2.6 |
| OpenCV | 5.0.0.93 |
| Pillow | 12.2.0 |
| PyYAML | 6.0.3 |
| CUDA Runtime | 12.8（PyTorch 构建版本） |
| ROS / Compiler | 本目录脚本无需 ROS 或单独的编译步骤 |

### 硬件

- 已记录训练设备：AMD Ryzen 9 8945HX、NVIDIA GeForce RTX 5060 Laptop GPU，显存 8151 MiB。
- `train.py` 默认使用 CUDA 设备 `0`；训练配置为 `batch=2`、`workers=0`。
- `detect.py` 在 CUDA 可用时默认使用 GPU，否则使用 CPU；也可通过 `--device cpu` 指定 CPU。
- 图片和视频文件推理无需外接传感器。实时推理需要 OpenCV 可读取的摄像头；本模块不依赖执行器。

## 3. 安装

### 3.1 获取代码并创建环境

以下命令以 Windows PowerShell 为例：

```powershell
git clone https://github.com/AgroTech-SCAU/Tomato-Orbbec-Vision.git
cd Tomato-Orbbec-Vision/Tomato_Model

py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 3.2 安装依赖

复现已记录的 CUDA 12.8 环境：

```powershell
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install ultralytics==8.4.93 numpy==2.2.6 opencv-python==5.0.0.93 pillow==12.2.0 pyyaml==6.0.3
python -m pip check
```

其他硬件环境请根据 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/)选择适配版本，并记录实际依赖。Ultralytics 的安装方式见[官方文档](https://docs.ultralytics.com/quickstart/)。

检查 GPU 是否可用：

```powershell
python -c "import torch, ultralytics; print('torch:', torch.__version__); print('ultralytics:', ultralytics.__version__); print('CUDA available:', torch.cuda.is_available())"
```

## 4. 数据与模型准备

本模块为 Python 脚本，无需单独构建。训练前需要准备数据集和初始权重；仅推理时需要番茄任务训练后的权重及输入图片、视频或摄像头。

### 4.1 数据集

数据来源：[Roboflow tomato-h72eq v18](https://universe.roboflow.com/buzhidao-mkdfs/tomato-h72eq/dataset/18)。`data.yaml` 记录的数据许可为 CC BY 4.0，使用和再分发时应保留相应来源及署名信息。

将该版本的 **YOLO 实例分割格式**数据整理到当前目录下的 `dataset/`：

```text
dataset/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

| 划分 | 图片数 | ripe 实例 | stem 实例 | unripe 实例 |
|---|---:|---:|---:|---:|
| train | 763 | 6420 | 659 | 3515 |
| valid | 96 | 806 | 81 | 524 |
| test | 96 | 741 | 82 | 559 |
| 合计 | 955 | 7967 | 822 | 4598 |

训练脚本要求图片尺寸为 `768 × 768`，图片与标签一一对应，且各划分图片数、实例数与上述统计一致。标签必须为归一化的分割多边形，不能使用仅含五列的目标检测框标签。

> `train.py` 每次启动都会生成或覆盖 `data.yaml`，包括 `--check-only` 模式。它用于复现固定的 v18 数据和 E13 配置；仅修改 `data.yaml` 无法切换数据集。迁移到新数据集时，需要同步调整脚本中的配置生成逻辑及审计规则。

### 4.2 模型文件

| 文件 | 用途 |
|---|---|
| `yolo11m-seg.pt` | E13 训练使用的初始权重 |
| `yolo26n.pt` | 当前目录另附权重；现有 E13 训练与默认推理流程未使用 |
| `runs/segment/v18_e13_mosaic025/weights/best.pt` | E13 训练产出的最佳权重，供番茄实例分割推理使用 |

**仅有目录中的两个初始模型文件，不能直接复现训练好的番茄识别效果。** 如未附带 `best.pt`，请完成训练，或向维护者获取对应实验权重，并通过 `--weights` 指定路径。

已有训练记录中的 `best.pt` SHA256 为：

```text
CB05ED52B54E1060CDA6C04294227C8589AA3AB4B2175A759E4F15CB68671959
```

该哈希用于核对已有实验产物，不要求重新训练的文件具有相同哈希。

## 5. 快速开始

以下命令均在 `Tomato_Model/` 目录内执行。

### 5.1 检查数据与配置

```powershell
python train.py --check-only
```

预期输出包含 `Dataset audit passed`，汇总为 955 张图片、13387 个实例；随后打印 E13 参数并退出，不启动训练。

### 5.2 开始训练

```powershell
python train.py
```

默认输出目录为 `runs/segment/v18_e13_mosaic025/`。如该目录已存在，脚本会拒绝覆盖；新实验可指定名称：

```powershell
python train.py --name v18_e13_run02
```

中断后，从相应实验的 `last.pt` 恢复：

```powershell
python train.py --resume
# 自定义实验名时：
python train.py --name v18_e13_run02 --resume
```

`--resume` 用于恢复尚未完成的训练；已经正常结束或早停的实验，如需重新训练，应使用新的实验名称。

训练完成后生成 `weights/best.pt`、`weights/last.pt`、`args.yaml`、`results.csv`、曲线及 `training_run_record.json`。

### 5.3 图片与文件夹推理

准备好训练后的 `best.pt`，将单图命令中的路径替换为实际文件：

```powershell
python detect.py --source "path/to/tomato.jpg"
python detect.py --source dataset/test/images
```

使用其他位置的权重，或在 CPU 上推理：

```powershell
python detect.py --source "path/to/tomato.jpg" --weights "path/to/best.pt" --device cpu
```

### 5.4 视频与摄像头推理

```powershell
python detect.py --source "path/to/tomato.mp4" --show
python detect.py --source 0
```

摄像头默认显示预览，视频默认不显示；在预览窗口按 `Q` 或 `Esc` 退出。图片预览模式下，每张图片等待按键后继续。

无预览、限制处理帧数且不保存结果：

```powershell
python detect.py --source 0 --no-show --max-frames 100 --no-save
```

### 预期结果

- 输出画面包含检测框、分割掩膜、类别名称、置信度与当前帧类别数量。
- 标注图片默认保存在 `outputs/e13_detect/images/`。
- 标注视频默认保存在 `outputs/e13_detect/videos/`。
- 控制台输出处理进度和平均推理耗时。

脚本显示的 FPS 按单次 `model.predict()` 耗时换算，不代表包含采集、绘制、显示和保存的完整系统帧率。重复处理同名输入时，建议使用不同的 `--output-dir` 保存各次结果。

## 6. 配置说明

### 6.1 训练配置

训练参数在 `train.py` 的 `e13_training_arguments()` 中定义；实际执行值以输出目录中的 `args.yaml` 为准。

| 配置项 | 默认值 | 说明 |
|---|---|---|
| `task` | `segment` | 实例分割 |
| `epochs` | `200` | 最大训练轮数 |
| `patience` | `40` | 连续未改善时提前停止 |
| `batch` | `2` | 批次大小 |
| `imgsz` | `768` | 输入尺寸 |
| `device` | `0` | CUDA 设备编号 |
| `workers` | `0` | 数据加载子进程数 |
| `optimizer` | `SGD` | 优化器 |
| `lr0` | `0.01` | 初始学习率 |
| `seed` | `42` | 随机种子 |
| `amp` | `True` | 自动混合精度 |
| `mosaic` | `0.25` | Mosaic 增强概率 |
| `close_mosaic` | `30` | 最后阶段关闭 Mosaic 的配置 |
| `mask_ratio` | `2` | 掩膜下采样比例 |
| `name` | `v18_e13_mosaic025` | 实验输出名称 |

训练脚本仅提供 `--name`、`--check-only`、`--resume` 三个业务命令行参数。调整 `batch`、`epochs` 或 `device` 等参数需要修改上述函数，不能直接传入 `--batch` 或 `--device`。

### 6.2 推理配置

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--source` | 必填 | 图片、图片目录、视频文件或摄像头编号 |
| `--weights` | `runs/segment/v18_e13_mosaic025/weights/best.pt` | 模型权重，默认相对脚本目录定位 |
| `--output-dir` | `outputs/e13_detect` | 输出目录，默认位于脚本目录下 |
| `--imgsz` | `768` | 推理输入尺寸 |
| `--conf` | `0.15` | 置信度阈值 |
| `--iou` | `0.70` | IoU 阈值 |
| `--device` | CUDA 可用时为 `0`，否则 `cpu` | 推理设备 |
| `--show` | 未开启 | 显示预览；摄像头默认显示 |
| `--no-show` | 未开启 | 禁止显示预览，不能与 `--show` 同用 |
| `--no-save` | 未开启 | 不保存标注图片或视频 |
| `--camera-width` / `--camera-height` | `0` / `0` | 摄像头分辨率请求；0 表示不主动设置 |
| `--max-frames` | `0` | 视频或摄像头处理帧数上限；0 表示不限 |

## 7. 目录结构

```text
Tomato_Model/
├── README.md                          # 本文档
├── BENCHMARK_FULL_RECORD_TEMPLATE.md   # 已回填的 E13 完整训练记录
├── data.yaml                          # 数据集配置，训练脚本会重新生成
├── detect.py                          # 图片、视频及摄像头推理
├── test.md                            # 训练终端日志及最终 valid 验证输出
├── train.py                           # 数据审计、E13 训练及产物记录
├── yolo11m-seg.pt                      # E13 初始权重
├── yolo26n.pt                          # 另附权重，当前 E13 流程未使用
├── dataset/                           # 需另行准备
├── runs/                              # 训练后生成
│   └── segment/
│       └── v18_e13_mosaic025/
│           ├── weights/
│           │   ├── best.pt
│           │   └── last.pt
│           ├── args.yaml
│           ├── results.csv
│           ├── results.png
│           └── training_run_record.json
└── outputs/                           # 保存推理结果时生成
    └── e13_detect/
        ├── images/
        └── videos/
```

## 8. 文档与已记录结果

- [项目规划](../docs/plan.md)：仓库级版本计划与后续工作。
- [完整训练记录](BENCHMARK_FULL_RECORD_TEMPLATE.md)：环境、数据、E13 参数、产物哈希和验收状态；虽然文件名包含 `TEMPLATE`，目前已回填训练结果。
- [训练日志](test.md)：训练终端输出及最佳权重的最终验证结果。
- [数据集配置](data.yaml)：数据路径与类别映射。

### E13 验证集结果

已有记录显示：训练计划为 200 轮，在第 95 轮正常早停，最佳轮次为第 55 轮，训练耗时约 3.88 小时。以下为训练结束后加载 `best.pt`，在 **valid 集（96 张图片、1411 个实例）**上的最终验证结果，来源为 [test.md](test.md)。

| 类别 | Box mAP50 | Box mAP50-95 | Mask mAP50 | Mask mAP50-95 |
|---|---:|---:|---:|---:|
| all | 0.882 | 0.722 | 0.883 | 0.665 |
| ripe | 0.956 | 0.866 | 0.957 | 0.822 |
| stem | 0.765 | 0.499 | 0.767 | 0.416 |
| unripe | 0.925 | 0.803 | 0.926 | 0.756 |

这里统一引用最终验证输出，未混用训练过程中 CSV 的最佳轮次统计。`test.md` 的文件名不代表独立 test 集评估；已有记录明确说明本次尚未进行 test 集评估。

已有训练记录还披露了 23 个跨划分同源切片组，可能存在近重复或场景泄漏，因此上述指标不能直接代表独立真实场景中的泛化性能。果梗类别表现弱于番茄类别，仍按候选区域辅助信息使用。

## 9. 常见问题

### 找不到 `best.pt`

`detect.py` 默认读取训练产物，而不是目录中的 `yolo11m-seg.pt`。先完成训练或获取对应 E13 权重；若使用自定义实验名，需通过 `--weights` 指向该实验的 `best.pt`。

### 数据检查失败

检查是否使用固定 v18 数据集、验证目录是否为 `valid`、图片尺寸是否为 768×768、标签是否为分割多边形，以及图片数与实例数是否匹配。脚本会拒绝空标签、标签缺失及仅含检测框的标注。

### CUDA 显存不足或数据加载失败

已有实验中 `batch=4` 曾出现显存不足，多进程加载曾出现系统内存不足，因此当前配置使用 `batch=2`、`workers=0`。先关闭占用资源的程序；如仍需调小批次，应在训练配置函数中修改并记录实验差异。

### 没有 GPU 能否运行？

推理可使用 `python detect.py --source "path/to/tomato.jpg" --device cpu`。训练脚本固定使用设备 `0`，不会自动回退 CPU；如需 CPU 训练，应修改训练配置，且耗时可能明显增加。



### 摄像头无法打开或没有预览

检查设备连接、系统摄像头权限、是否被其他程序占用，并尝试其他编号，例如 `--source 1`。确认未使用 `--no-show`；视频文件需要显式添加 `--show` 才会显示预览。

### 能否直接输出三维坐标或机械臂采摘点？

当前输出为图像中的识别与分割结果。三维定位和采摘点计算需要进一步接入深度数据、相机标定及机器人坐标转换，不属于当前脚本功能。

## 10. 版本与发布

正式稳定版本采用 Git Tag + GitHub Release 发布，版本历史见 [Releases](https://github.com/AgroTech-SCAU/Tomato-Orbbec-Vision/releases)。

模型发布时应同时提供对应的 `best.pt`、训练配置、类别映射、验证结果、依赖版本及权重 SHA256，便于复现和核对。后续模型优化与系统功能安排见 [docs/plan.md](../docs/plan.md)。

## 11. 维护者

- Maintainer / 项目负责人：[@alexwang0529](https://github.com/alexwang0529)
- 问题反馈与改进建议：[GitHub Issues](https://github.com/AgroTech-SCAU/Tomato-Orbbec-Vision/issues)

## 12. 以当前模型为基线的系统开发

### 12.1 基线约定

后续应用默认加载 E13 训练产出的 `best.pt`，保留 `ripe / stem / unripe` 的类别 ID 和顺序。初始权重 `yolo11m-seg.pt` 用于训练初始化，不能替代业务推理基线。

模型、应用和结果协议分别管理版本。新增界面、任务管理或统计功能时保持基线模型不变；新模型使用独立编号和产物目录，通过评估后再显式切换，保留 E13 回退能力。果梗始终表述为候选区域辅助识别。

### 12.2 五套系统范围

以下为《软著.docx》提出的开发范围，当前均需完成独立的软件功能验收。

| 编号 | 系统 | 核心输入 | 核心输出 | 当前可复用基础 |
|---|---|---|---|---|
| S1 | 训练数据增强效果仿真与参数可视化系统 V1.0 | 图片、YOLO 标注、类别配置 | 增强预览、标注检查、参数方案 | E13 增强参数与数据格式 |
| S2 | 模型训练与实验管理系统 V1.0 | 数据集、增强配置、初始权重 | 训练任务、模型、指标和报告 | `train.py` 与训练记录 |
| S3 | 番茄多源影像智能识别系统 V1.0 | 图片、目录、视频、摄像头、E13 模型 | 可视化结果与标准识别结果包 | `detect.py` 推理流程 |
| S4 | 番茄成熟度统计分析与结果管理系统 V1.0 | S3 识别结果包或兼容 CSV | 数量、成熟率、历史记录与报表 | 类别定义；统计系统待开发 |
| S5 | 番茄目标识别结果复核与样本质量管理系统 V1.0 | 原图、预测结果、掩膜、可选已有标注 | 复核队列、审阅记录与样本清单 | 分割输出；复核系统待开发 |

S1 只预览增强效果、检查标注和导出参数，不执行训练或识别；S2 负责训练；S3 负责识别；S4 负责统计；S5 负责人工作业与样本质量管理。

