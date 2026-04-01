# FlashAvatar-DepthFusion

面向头部数字人重建的单目深度正则化 FlashAvatar 扩展版本。

本仓库是在原始 FlashAvatar 基础上构建的公开研究分支，核心目标是在保留 FLAME 条件高斯头像重建主流程的同时，引入一条实用的单目深度监督路径，以缓解 RGB-only 训练在侧脸几何与轮廓稳定性上的欠约束问题。

## 项目概览

- 研究重点：在保持 RGB 重建为主监督信号的前提下，用单目深度增强几何稳定性
- 深度来源：逐帧 VideoDepth Anything 风格深度图
- 深度损失：尺度不变 Pearson 相关性损失
- 监督策略：只在保守的人脸核心区域使用深度，不监督嘴部内部与轮廓边缘
- 当前状态：面向受控实验的工程原型，不是官方上游发布版本

## 为什么要做这个分支

仅使用 RGB 训练时，头像几何在侧视角下通常会出现欠约束问题，最常见的表现包括：

- 下颌线与脸颊区域的深度顺序不稳定
- 头发和轮廓边缘容易抖动或漂移
- 表情变化时下半脸几何容易失真
- 多帧之间的侧视图一致性不足

这个分支并不试图强行插入一个显式 3D 中间重建阶段，而是把单目深度作为软约束，辅助原始 FlashAvatar 管线学习更稳定的几何表示。

## 相比原始 FlashAvatar 的改动

- `train.py` 增加了可选深度监督参数
- `Scene_mica` 支持读取 `dataset/<id>/depth/*.npy`
- [`utils/loss_utils.py`](utils/loss_utils.py) 中实现了尺度不变深度相关性损失
- 在施加深度监督前，使用有效区域掩码过滤高风险区域
- 预处理文档补充了 `mp4 -> imgs / alpha / parsing / depth / checkpoint` 的数据约定

## 方法摘要

这里的深度并不被视为严格的度量真值，而是作为相关性约束信号参与训练：

```text
L_depth = 1 - corr(render_depth[valid], mono_depth[valid])
```

有效监督区域采用保守策略：

```text
valid_depth_mask = erode(head_mask, 2~3 px) - mouth_mask
```

这样做主要是为了规避单目深度最常见的两个误差来源：

- 头发、下巴、耳朵与轮廓边界附近的深度串色
- 张嘴区域内部的错误深度估计

## 结构示意

当前设计保持 FlashAvatar 原有的 FLAME 驱动高斯头像表示，只把深度作为辅助正则项注入训练过程。

![Depth Fusion Architecture](figures/depth_fusion_architecture/depth_fusion_architecture_preview.png)

## 定性对比

当前工作区中已经整理了若干对比卡片：

| 身份 | 结果 |
| --- | --- |
| Obama | ![Obama Comparison](_tmp_compare/Obama_contact.png) |
| Mead2 | ![Mead2 Comparison](_tmp_compare/Mead2_contact.png) |
| Luoxiang | ![Luoxiang Comparison](_tmp_compare/luoxiang_contact.png) |

## 仓库导读

- [docs/overview.md](docs/overview.md)：分支整体说明
- [docs/depth_supervision.md](docs/depth_supervision.md)：深度损失动机与掩码策略
- [docs/preprocessing.md](docs/preprocessing.md)：预处理流程与输出约定
- [CONTRIBUTING.md](CONTRIBUTING.md)：协作与仓库规范
- [Agents.zh-CN.md](Agents.zh-CN.md)：更详细的中文工程说明
- [pseudocode.md](pseudocode.md)：预处理伪代码
- [data_schema.json](data_schema.json)：机器可读的数据结构定义

## 快速开始

### 1. 创建环境

```bash
conda env create --file environment.yml
conda activate FlashAvatar
```

### 2. 安装 PyTorch3D

```bash
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
conda install pytorch3d -c pytorch3d
```

### 3. 编译原生扩展

如果环境中尚未构建，需要额外编译以下模块：

- `submodules/simple-knn`
- `submodules/diff-gaussian-rasterization`

### 4. baseline 训练

```bash
python train.py --idname <id_name>
```

### 5. 启用深度监督训练

```bash
python train.py \
  --idname <id_name> \
  --use_depth_supervision \
  --depth_loss_weight 0.05 \
  --depth_start_iter 0 \
  --depth_erode_kernel 3 \
  --min_depth_samples 256
```

### 6. 测试与新视角生成

```bash
python test.py --idname <id_name> --checkpoint dataset/<id_name>/log/ckpt/chkpnt.pth
python novel_view.py --idname <id_name> --checkpoint dataset/<id_name>/log/ckpt/chkpnt.pth
```

## 数据组织方式

每个身份目录建议满足如下结构：

```text
dataset/
  <id>/
    imgs/
    alpha/
    parsing/
    depth/          # 启用深度监督时需要提供

metrical-tracker/
  output/
    <id>/
      checkpoint/
```

帧对齐规则为：

```text
00000.frame  <->  00001.jpg
00001.frame  <->  00002.jpg
...
```

深度文件约定：

- `dataset/<id>/depth/00001.npy`
- `float32`
- 与 `imgs/*.jpg` 具有相同分辨率和编号
- 推荐来源：VideoDepth Anything

## 当前实现边界

深度相关代码目前已经接入：

- [train.py](train.py)
- [scene/__init__.py](scene/__init__.py)
- [scene/cameras.py](scene/cameras.py)
- [utils/loss_utils.py](utils/loss_utils.py)

当前实现采用“将高斯中心投影到图像平面，再在那里采样单目深度”的方式施加监督。它还没有把“稠密渲染深度图”作为默认渲染器输出完全暴露出来。

这让本分支更容易集成到现有 FlashAvatar 流程中，但它仍然应被视为“实验性的深度正则化路径”，而不是一个已经完全定型的 benchmark 版本。

## 预处理相关资源

仓库中还附带了若干预处理与可视化资源：

- [preprocess_flashavatar_mp4.py](preprocess_flashavatar_mp4.py)
- [pseudocode.md](pseudocode.md)
- [data_schema.json](data_schema.json)
- [test_flashavatar_schema.py](test_flashavatar_schema.py)
- [generate_depth_fusion_architecture.py](generate_depth_fusion_architecture.py)
- [draw_depth_fusion_flowchart.py](draw_depth_fusion_flowchart.py)
- [make_view_stability_comparison.py](make_view_stability_comparison.py)

## 未包含的外部资源

出于体积与授权原因，仓库中没有附带所有运行依赖。你仍然需要自行准备：

- `flame/` 所需的 FLAME 模型资源
- 如环境依赖，还需准备 FLAME mask 资源
- 启用深度监督时需要准备单目深度图
- `metrical-tracker/output/<id>/checkpoint` 下的跟踪输出
- 你自己的 `dataset/<id>` 数据

## 当前限制

- 这不是官方上游 FlashAvatar 仓库
- 运行仍依赖 CUDA、PyTorch3D 与高斯渲染相关原生扩展
- 当前深度监督是稀疏、投影式的，不是稠密深度渲染监督
- 公共示例素材有限，真正训练前仍需要本地准备数据与依赖

## 后续计划

- 从高斯渲染器中显式暴露稠密深度分支
- 对比稀疏中心投影损失与稠密深度渲染损失
- 增加更清晰的公开预处理示例
- 支持多段 identity 的分段训练与采样策略

## 致谢

本仓库建立在以下工作基础之上：

- FlashAvatar
- 3D Gaussian Splatting
- metrical-tracker / MICA
- BiSeNet face parsing
- MatAnyone
- VideoDepth Anything

## 引用

如果你的工作基于本仓库，请同时引用原始 FlashAvatar 论文：

```bibtex
@inproceedings{xiang2024flashavatar,
  author    = {Jun Xiang and Xuan Gao and Yudong Guo and Juyong Zhang},
  title     = {FlashAvatar: High-fidelity Head Avatar with Efficient Gaussian Embedding},
  booktitle = {CVPR},
  year      = {2024}
}
```
