# FlashAvatar 接手说明（中文精简版）

## 1. 仓库要做什么

这个仓库实现论文 **FlashAvatar** 的核心训练与渲染部分，目标是：

- 从**单目视频的预处理结果**重建一个个体化头部 avatar
- 用 **FLAME 网格 + 固定数量 3D Gaussians** 表示头像
- 用 **表情 / 眼球 / 下巴 / 眼睑参数** 驱动高斯形变
- 用 **3D Gaussian Splatting** 做快速渲染

它**不是**原始视频到结果的一站式流水线。仓库默认你已经有：

- 抽帧结果
- alpha 前景掩码
- parsing 语义掩码
- 每帧 FLAME + 相机跟踪结果 `*.frame`

## 2. 入口文件

- [`train.py`](/root/AvatarStack/modules/FlashAvatar-code/train.py)：单身份训练
- [`test.py`](/root/AvatarStack/modules/FlashAvatar-code/test.py)：同视角测试渲染
- [`novel_view.py`](/root/AvatarStack/modules/FlashAvatar-code/novel_view.py)：本工作区新增的新视角渲染工具
- [`scene/__init__.py`](/root/AvatarStack/modules/FlashAvatar-code/scene/__init__.py)：把磁盘数据装成训练/测试样本
- [`src/deform_model.py`](/root/AvatarStack/modules/FlashAvatar-code/src/deform_model.py)：FLAME 条件下的高斯 offset 网络
- [`scene/gaussian_model.py`](/root/AvatarStack/modules/FlashAvatar-code/scene/gaussian_model.py)：高斯参数状态
- [`gaussian_renderer/__init__.py`](/root/AvatarStack/modules/FlashAvatar-code/gaussian_renderer/__init__.py)：高斯渲染
- [`flame/flame_mica.py`](/root/AvatarStack/modules/FlashAvatar-code/flame/flame_mica.py)：FLAME 几何模型

## 3. 输入输出定义

### 输入目录

每个身份需要：

- `dataset/<id>/imgs/*.jpg`
- `dataset/<id>/alpha/*.jpg`
- `dataset/<id>/parsing/*_neckhead.png`
- `dataset/<id>/parsing/*_mouth.png`
- `metrical-tracker/output/<id>/checkpoint/*.frame`

设计扩展里还建议支持：

- `dataset/<id>/depth/*.npy`

其中 `depth/*.npy` 表示逐帧单目深度图，推荐由 **VideoDepth Anything** 预先计算得到，并保持：

- 与 `imgs/*.jpg` 同分辨率
- 与 `imgs/*.jpg` 同编号
- 每帧一个 `float32` depth map

其中 `.frame` 至少要包含：

- `flame.exp`: `(1, 100)`
- `flame.shape`: `(1, 300)`
- `flame.eyes`: `(1, 12)`
- `flame.eyelids`: `(1, 2)`
- `flame.jaw`: `(1, 6)`
- `opencv.R`: `(1, 3, 3)`
- `opencv.t`: `(1, 3)`
- `opencv.K`: `(1, 3, 3)`
- `img_size`: `(2,)`

### 输出目录

训练时：

- `dataset/<id>/log/train/*.jpg`
- `dataset/<id>/log/ckpt/chkpnt*.pth`

测试时：

- `dataset/<id>/log/test.avi`
- 如有转码，则额外有 `test.mp4`

## 4. 核心数据流

### 第 1 步：样本加载

[`scene/__init__.py`](/root/AvatarStack/modules/FlashAvatar-code/scene/__init__.py) 做的事情：

- 读取 RGB 图像，得到 `gt_image: (3, H, W)`，通常是 `(3, 512, 512)`
- 读取 `alpha: (1, H, W)`
- 读取 `head_mask: (1, H, W)`
- 读取 `mouth_mask: (1, H, W)`
- 读取 `.frame` 中的 FLAME 参数和相机参数
- 构造 `Camera` 对象

注意这里有一个关键约定：

- `frame_delta = 1`
- 也就是 `00000.frame` 对应 `00001.jpg`

如果你的图像和 tracker 输出不是这个偏移关系，结果会错位。

### 第 2 步：身份级固定信息

从第一帧 `.frame` 中取：

- `shape_param: (1, 300)`

它代表身份 shape，后续整个身份共享。

### 第 3 步：UV 空间高斯初始化

[`src/deform_model.py`](/root/AvatarStack/modules/FlashAvatar-code/src/deform_model.py) 里的 `example_init()` 对应论文里的“surface-embedded Gaussian initialization”。

关键过程：

1. 用 FLAME 生成中性几何：
   - `geometry_shape: (B, 5023, 3)`
2. 把顶点转换成面表示：
   - `face_vertices_shape: (B, F, 3, 3)`
3. 在 UV 空间做一次 rasterization：
   - `rast_out: (B, 4, 128, 128)`
4. 抽取可见 UV 采样点：
   - `uv_vertices_shape: (1, V_uv, 3)`
5. 做位置编码：
   - `uv_vertices_shape_embeded: (1, V_uv, 51)`
6. 再用 FLAME mask 过滤掉边界，只保留 head+neck 区域

当前工作区实测：

- UV 分辨率：`128`
- 过滤前采样点数：`14876`
- 过滤后真正高斯数：`13453`

这也是后面高斯的固定数量。

### 第 4 步：表达驱动 offset

论文里的条件 `psi` 在代码里实现为：

- `expr: (1, 100)`
- `jaw: (1, 6)`
- `eyes: (1, 12)`
- `eyelids: (1, 2)`

拼起来是：

- `condition: (1, 120)`

位置编码维度是：

- `51`

所以每个高斯送进 MLP 的输入维度是：

- `171 = 51 + 120`

MLP 输出 10 维：

- 前 3 维：位置残差 `Delta mu`
- 中间 4 维：旋转增量
- 后 3 维：缩放系数

解码后的主要张量：

- `verts_final: (1, 13453, 3)`
- `rot_delta: (1, 13453, 4)`
- `scale_coef: (1, 13453, 3)`

### 第 5 步：构造 Gaussian 状态

第一次训练时：

- `gaussians.create_from_verts(verts_final[0])`

初始化后主要参数形状：

- `xyz: (13453, 3)`
- `features_dc: (13453, 1, 3)`
- `features_rest: (13453, 15, 3)`
- `scaling_base: (13453, 3)`
- `rotation_base: (13453, 4)`
- `opacity: (13453, 1)`

后续每帧更新：

- `xyz = verts_final`
- `rotation = rotation_base * rot_delta`（四元数乘）
- `scaling = scaling_base * scale_coef`

核心理解：

- 这个仓库里的高斯位置不是自由漂移优化出来的
- 它主要由 **FLAME 几何 + offset 网络** 驱动

### 第 6 步：渲染

[`gaussian_renderer/__init__.py`](/root/AvatarStack/modules/FlashAvatar-code/gaussian_renderer/__init__.py) 接收：

- 相机参数
- 高斯中心 / 旋转 / 缩放 / 不透明度 / SH 颜色
- 背景色

输出：

- `render: (3, H, W)`

训练和测试里最终都把它和 GT 拼接成图或视频。

补充说明：

- 当前渲染器默认只返回 RGB
- 如果要接入单目深度监督，还需要在训练侧额外暴露每帧的 `render_depth`
- 这属于**设计扩展**，不是当前 `train.py` 已经启用的默认行为

## 5. 训练目标

[`train.py`](/root/AvatarStack/modules/FlashAvatar-code/train.py) 用的损失：

- Huber 重建损失
- 嘴部区域加权 Huber，权重 `40`
- LPIPS perceptual loss

LPIPS 的启用策略：

- 前 `15000` iter 不开
- 之后权重是 `0.05`

优化对象主要有两类：

- Gaussian 的外观相关属性
- offset MLP 参数

### 5.1 设计扩展：尺度不变深度损失

如果后续接入单目深度监督，建议采用：

- **Scale-Invariant Depth Loss, SIDL**
- 在本工作区中建议具体实现为：**Pearson Correlation Loss**

核心思想：

- 不把单目深度显式整合成 3D mesh
- 直接读取逐帧单目深度图 `dataset/<id>/depth/*.npy`
- 让渲染深度 `render_depth` 与单目深度 `mono_depth` 在有效区域上做相关性约束

一个直接可用的形式是：

```text
L_depth = 1 - corr(render_depth[valid], mono_depth[valid])
```

其中：

- `corr` 是皮尔逊相关系数
- 这个形式天然对全局尺度和偏移更稳健
- 很适合直接约束单目深度这种“相对深度”信号

### 5.2 深度监督的 valid mask 设计

深度损失不能直接在整张图上算，必须结合高置信度有效区域。

建议的 valid mask 构造方式：

1. 从 `neckhead mask` 出发，只保留核心头颈区域。
2. 对 mask 边界向内腐蚀 `2-3` 个像素。
3. 再把 `mouth` 内部区域抠掉，令其深度权重为 `0`。

可以理解成：

```text
valid_depth_mask
  = erode(neckhead_mask, 2~3 px)
    - mouth_inner_mask
```

这样做的目的，是让深度监督只落在：

- 脸颊
- 鼻梁
- 额头
- 稳定的下半脸实体区域

而不去强约束：

- 头发边缘
- 下巴与背景交界
- 张嘴时的口腔内部

### 5.3 为什么必须裁掉边界和嘴部

#### A. 边缘大出血（Edge Depth Bleeding）

单目深度大模型在头发边缘、下颌线边缘，往往会输出平滑过渡的“斜坡深度”。

但真实几何在这些地方更接近：

- 深度突变
- 清晰遮挡边界

如果直接拿这种“斜坡深度”监督 3DGS，常见风险是：

- 侧脸被向外拉扯
- 下巴边缘变尖
- 发际线和轮廓处出现锥化、鼓包或错误外扩

所以文档层面推荐：

- 只在 mask 内收后的核心区域计算深度损失
- 边缘区域交给 RGB 多视角监督和 FLAME 先验去收敛

#### B. 张嘴等非刚性运动的深度冲突

嘴巴张开时，口腔内部真实深度会突然变深。

但单目深度模型经常把：

- 嘴唇
- 牙齿
- 口腔阴影

错误地压成一个近似平面。

如果强行监督，会和下面这些先验冲突：

- FLAME 下颌骨运动
- RGB 重建信号
- 嘴部局部高频外观

所以建议：

- 嘴内部区域深度权重设为 `0`
- 口腔几何主要由 FLAME jaw 先验与 RGB 驱动
- 深度只负责稳定外轮廓和核心脸部实体结构

## 6. 关键约束

### 数据层

- 必须有预处理后的 `imgs/alpha/parsing/checkpoint`
- 不能只有原始 `mp4`
- `00000.frame` 必须存在
- 图像、mask、frame 数量和命名必须严格对齐

如果启用深度监督扩展，还要额外满足：

- `dataset/<id>/depth/*.npy` 存在
- depth 与 `imgs` 同分辨率、同编号
- depth 不能整帧为常数或大面积 NaN / Inf

### 分辨率层

- 代码里很多地方默认按 `512x512` 工作
- 背景图直接初始化成 `(3, 512, 512)`
- 训练/测试输出也默认围绕 `512` 构造

所以如果你换输入分辨率，最先检查：

- [`scene/__init__.py`](/root/AvatarStack/modules/FlashAvatar-code/scene/__init__.py)
- [`train.py`](/root/AvatarStack/modules/FlashAvatar-code/train.py)
- [`test.py`](/root/AvatarStack/modules/FlashAvatar-code/test.py)

### 运行层

- 实际上依赖 CUDA
- 依赖 `pytorch3d`
- 依赖 `simple-knn`
- 依赖 `diff-gaussian-rasterization`

CPU 路径基本不可用。

## 7. 边界条件和常见坑

### 短序列

原始逻辑默认测试集取最后 `500` 帧，短视频会出负索引问题。

当前工作区里：

- [`scene/__init__.py`](/root/AvatarStack/modules/FlashAvatar-code/scene/__init__.py) 已做短序列修补

如果你在别处复现原仓库，需要注意这个差异。

### 跟踪误差

FlashAvatar 对 FLAME 跟踪质量依赖很强，尤其是：

- 全局头部姿态
- 表情参数
- 相机外参

如果这些偏了，后面即使训练收敛，也会表现成：

- 细节糊
- 脸和背景/轮廓对不齐
- 边缘伪影

### 头发和大幅非刚性运动

论文和代码都默认：

- 头部主体运动可由 FLAME + 小 offset 表达

因此它不擅长：

- 大幅飘动头发
- 严重非刚性附件运动
- 非头部主导的复杂运动

## 8. 改代码时优先看哪里

如果你要排查问题，建议按问题类型切入口：

- 数据对齐 / split / 路径问题：
  - [`scene/__init__.py`](/root/AvatarStack/modules/FlashAvatar-code/scene/__init__.py)
- 相机和视角问题：
  - [`scene/cameras.py`](/root/AvatarStack/modules/FlashAvatar-code/scene/cameras.py)
  - [`utils/graphics_utils.py`](/root/AvatarStack/modules/FlashAvatar-code/utils/graphics_utils.py)
  - [`novel_view.py`](/root/AvatarStack/modules/FlashAvatar-code/novel_view.py)
- 表情驱动 / offset 网络问题：
  - [`src/deform_model.py`](/root/AvatarStack/modules/FlashAvatar-code/src/deform_model.py)
  - [`flame/flame_mica.py`](/root/AvatarStack/modules/FlashAvatar-code/flame/flame_mica.py)
- 高斯参数 / checkpoint 恢复问题：
  - [`scene/gaussian_model.py`](/root/AvatarStack/modules/FlashAvatar-code/scene/gaussian_model.py)
- 渲染问题：
  - [`gaussian_renderer/__init__.py`](/root/AvatarStack/modules/FlashAvatar-code/gaussian_renderer/__init__.py)

## 9. 当前工作区额外说明

这不是完全原始的上游仓库，当前工作区还有一些本地内容：

- [`scene/__init__.py`](/root/AvatarStack/modules/FlashAvatar-code/scene/__init__.py) 有短序列补丁
- [`novel_view.py`](/root/AvatarStack/modules/FlashAvatar-code/novel_view.py) 是本地新增工具
- 论文 PDF 也放在仓库内，便于对照阅读

如果你的目标是“严格复现实验”，要把这些本地补丁和原论文仓库实现区分开看。

## 10. 训练新人像的实操流程

如果输入只有一个 `mp4`，要先把它变成 FlashAvatar 能读的训练集。推荐按下面理解：

### 目标

你最后必须得到两类结果：

1. `dataset/<new_id>/...`
2. `metrical-tracker/output/<new_id>/checkpoint/*.frame`

只有这两类都齐，`train.py --idname <new_id>` 才能真正开始训练。

### 推荐流程

#### 第 1 步：准备原始视频

对输入 `mp4` 的建议标准：

- 单人头肩为主，尽量不要多人同框
- 人脸尽量始终可见，不要长时间遮挡
- 头部不要频繁出画
- 尽量避免大幅运动模糊
- 建议包含：
  - 一段相对中性表情
  - 一些张嘴、转头、眨眼等表情变化

如果视频很短、姿态变化很少、嘴基本不开，能训，但泛化和细节通常会差。

#### 第 2 步：从 `mp4` 抽帧并做 matting

当前工作区里可用的本地链路是 [`VHAP`](/root/AvatarStack/modules/VHAP/README.md)。

它提供了：

- [`vhap/preprocess_video.py`](/root/AvatarStack/modules/VHAP/vhap/preprocess_video.py)
- `mp4 -> images/*.jpg`
- 可选生成 `alpha_maps/*.jpg`

这一步的产物本质上要有：

- RGB 帧序列
- alpha 前景掩码

#### 第 3 步：做 FLAME 跟踪

FlashAvatar 自己不带 raw-video tracker，所以要依赖外部跟踪流程。

当前工作区里本地可用的是：

- [`vhap/staged_track_sequence.py`](/root/AvatarStack/modules/VHAP/vhap/staged_track_sequence.py)

它负责从抽帧结果里估计：

- 相机参数
- FLAME shape / expression / jaw / eyes / eyelids

然后再导出成 NeRF/3DGS 风格的数据。

#### 第 4 步：导出成可转换结果

从本工作区经验看，最容易衔接 FlashAvatar 的中间格式是 VHAP 导出的：

- `images/*.png`
- `fg_masks/*.png`
- `flame_param/*.npz`
- `transforms.json`
- `canonical_flame_param.npz`

这类目录我们已经验证过可以转换成 FlashAvatar 所需格式并用于训练/测试。

#### 第 5 步：转换成 FlashAvatar 训练格式

FlashAvatar 最终需要：

- `dataset/<new_id>/imgs/*.jpg`
- `dataset/<new_id>/alpha/*.jpg`
- `dataset/<new_id>/parsing/*_neckhead.png`
- `dataset/<new_id>/parsing/*_mouth.png`
- `metrical-tracker/output/<new_id>/checkpoint/*.frame`

其中：

- `imgs` 是训练图像
- `alpha` 是前景区域
- `neckhead` 是头颈区域 mask
- `mouth` 是嘴部区域 mask
- `*.frame` 是逐帧 FLAME + 相机参数

#### 第 6 步：启动训练

准备完成后，训练入口就是：

```bash
python train.py --idname <new_id>
```

如果要从已有 checkpoint 续训：

```bash
python train.py --idname <new_id> --start_checkpoint <path_to_ckpt>
```

### “支持重训”的预处理标准

这里的“支持重训”不是“文件存在就行”，而是至少满足下面三层标准。

#### A. 文件完整性标准

必须满足：

- `imgs / alpha / parsing / checkpoint` 四类文件都存在
- 第 1 帧 `.frame` 存在，即 `00000.frame`
- 图像和 tracker 帧数一致
- 遵守当前代码的对齐方式：
  - `00000.frame -> 00001.jpg`
- 每一帧都能找到：
  - 图像
  - alpha
  - neckhead mask
  - mouth mask

只要缺一类，训练通常会直接报错。

#### B. 参数合法性标准

每个 `.frame` 至少要能读出这些张量：

- `flame.exp`: `(1, 100)`
- `flame.shape`: `(1, 300)`
- `flame.eyes`: `(1, 12)`
- `flame.eyelids`: `(1, 2)`
- `flame.jaw`: `(1, 6)`
- `opencv.R`: `(1, 3, 3)`
- `opencv.t`: `(1, 3)`
- `opencv.K`: `(1, 3, 3)`
- `img_size`: `(2,)`

如果维度不对，即使文件名齐，也不算“支持重训”。

#### C. 质量可用性标准

这部分最重要，决定“能不能训”和“训出来好不好”。

建议至少检查：

- 人脸在大多数帧里都被正确跟踪
- 相机姿态连续，没有明显跳变
- alpha 没有大面积漏抠
- head/neck mask 没把脸截坏，也没把大块背景吞进来
- 嘴部 mask 基本覆盖嘴区，否则嘴部重建会差
- 帧间没有明显错帧或命名错位

快速人工检查时，建议至少抽看：

- 开头 10 帧
- 中间 10 帧
- 结尾 10 帧

重点看三件事：

- 轮廓是否对齐
- 嘴部是否稳定
- 头转动时相机和网格是否跟着走

### 一个最实用的判断原则

如果一份预处理结果满足下面条件，就可以认为“达到可重训标准”：

- `Scene_mica` 能完整加载，不报缺文件 / 维度错误
- `test.py` 能在已有 checkpoint 下跑通，不报 shape / IO / camera 错误
- 随机抽帧看 GT、alpha、mask、跟踪姿态，大体是对齐的

换句话说：

- **能被 `Scene_mica` 正确消费**
- **能被 renderer 正常走通**
- **人工检查看起来没有明显错位**

这三条同时满足，就已经足够作为“支持重训”的工程标准。

## 11. 从 `mp4` 到可训练数据的流程图

下面这张“文字流程图”可以直接当作 FlashAvatar 预处理总览。

```text
原始输入
mp4

  |
  v

步骤 1：抽帧
mp4 -> images/*.jpg
目标：
- 得到连续帧
- 帧号稳定、无跳帧
- 后续所有结果都按这套帧号对齐

  |
  v

步骤 2：前景抠图
images/*.jpg -> alpha/*.jpg
目标：
- 生成前景 alpha mask
- 尽量去掉背景
- 不要漏掉头发、脖子、耳朵等轮廓

  |
  v

步骤 3：语义分割
images/*.jpg -> parsing/*_neckhead.png + parsing/*_mouth.png
目标：
- neckhead：限制头颈监督区域
- mouth：训练时加大嘴部损失权重

  |
  v

步骤 3.5：可选单目深度估计（设计扩展）
images/*.jpg -> depth/*.npy
目标：
- 直接读取逐帧单目深度，不转成显式 mesh
- 为后续 SIDL / Pearson depth loss 提供监督信号
- 保持与 `imgs` 完全同分辨率、同编号

  |
  v

步骤 4：FLAME 跟踪 + 相机估计
images/*.jpg -> checkpoint/*.frame
每帧输出至少包含：
- shape
- expression
- jaw
- eyes
- eyelids
- camera R / t / K

  |
  v

步骤 5：整理成 FlashAvatar 标准目录
dataset/<id>/
  imgs/
  alpha/
  depth/      # optional, planned depth supervision extension
  parsing/

metrical-tracker/output/<id>/
  checkpoint/

  |
  v

步骤 6：训练
python train.py --idname <id>
```

### 最终交付标准

真正可以开始训练时，磁盘上应该长这样：

```text
dataset/
  <id>/
    imgs/
      00001.jpg
      00002.jpg
      ...
    alpha/
      00001.jpg
      00002.jpg
      ...
    depth/
      00001.npy
      00002.npy
      ...
    parsing/
      00001_neckhead.png
      00001_mouth.png
      00002_neckhead.png
      00002_mouth.png
      ...

metrical-tracker/
  output/
    <id>/
      checkpoint/
        00000.frame
        00001.frame
        ...
```

并且满足对齐关系：

```text
00000.frame  <->  00001.jpg
00001.frame  <->  00002.jpg
...
```

### 预处理阶段最建议人工检查的 5 件事

1. 图像是否清晰，是否有大量运动模糊。
2. alpha 是否把头发、耳朵、脖子抠坏。
3. neckhead mask 是否只覆盖头颈，而不是吞入大片背景。
4. mouth mask 是否真正覆盖嘴部区域。
5. tracker 输出的姿态和表情是否和原图同步，没有明显跳变。

如果启用深度监督扩展，建议再额外检查：

6. depth 是否与 RGB 严格对齐，没有缩放、平移或插值错位。
7. 下巴边缘、头发边缘是否存在明显的深度 bleeding。
8. 张嘴帧里嘴内部区域是否已从深度监督里排除。

如果这 5 项基本成立，这份预处理数据大概率就能支持 FlashAvatar 重训。

### FlashAvatar 对抠图算法的要求

适合 FlashAvatar 的抠图算法，应满足以下要求：

- 时序稳定：随机抽查开头、中间、结尾帧时，人物轮廓无明显闪烁、跳变或忽大忽小。
- 关键区域完整：头发、耳朵、脖子、下巴等细长或薄弱区域不能持续性漏抠。
- 背景干净：人物外轮廓附近不能残留大块背景、噪声或碎片，否则容易被模型误学为前景。
- 严格对齐：alpha 必须与原图逐帧保持同尺寸、同编号、同位置，不能有平移、缩放、裁切偏差。
- 训练可用：接入 [`Scene_mica`](/root/AvatarStack/modules/FlashAvatar-code/scene/__init__.py) 以及训练/测试流程后，不应出现明显边缘发虚、轮廓漂移或背景被学进去的现象。

## 12. Mead2 Tracker 加速配置记录（2026-03-25）

### 背景

在 `Mead2` 这条 40 秒视频上，默认 `metrical-tracker` 配置耗时较长。为满足“约 1 小时内完成”目标，工作区采用了“同输入帧率 + 每帧轻量优化”的加速配置。

### 生效配置

配置文件：
[`Mead2.yml`](/root/AvatarStack/modules/metrical-tracker/configs/actors/Mead2.yml)

当前关键参数：

- `fps: 30`
- `image_size: [384, 384]`
- `pyr_levels: [[1.0, 24], [0.5, 8], [1.0, 8]]`
- `optimize_shape: false`
- `optimize_jaw: false`
- `begin_frames: 1`

### 与默认配置的差异

默认基线（见 [`configs/config.py`](/root/AvatarStack/modules/metrical-tracker/configs/config.py)）通常是：

- `fps: 25`
- `image_size: [512, 512]`
- `pyr_levels: [[1.0,160],[0.25,40],[0.5,40],[1.0,70]]`
- `optimize_shape: false`
- `optimize_jaw: false`

这次加速的核心不在帧率，而在“每帧迭代量”和“分辨率”：

- `pyr_levels` 每帧总迭代从 `310` 降到 `40`
- 图像从 `512` 降到 `384`

这两项一起大幅降低了单帧优化成本。

### 关键说明

- `fps` 改为 `30` 是为了与输入视频保持一致（输入视频约 `39.9s / 1197` 帧）。
- 帧率一致本身不会提速，甚至会增加总帧数；提速来自每帧更轻的优化配置。
- 该模式是“速度优先”工程配置，重建质量通常会低于默认慢速高迭代配置。

### 运行与备份约定

在切换加速配置并重跑时，工作区采用“重命名备份”而非删除：

- 旧输入中间件目录会重命名为 `*_speed30_backup_*` 或 `*_shortvideo_backup_*`
- 旧输出目录会重命名为 `Mead2_speed30_backup_*` 或 `Mead2_shortvideo_backup_*`

这样可以保留历史结果并支持回滚排查。
