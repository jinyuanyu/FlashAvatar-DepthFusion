# FlashAvatar mp4 预处理伪代码

## 范围

本文件只描述以下预处理环节：

1. 输入一个 `mp4`
2. 调用 `/root/AvatarStack/modules/BiSeNet-face-parsing` 对首帧做人脸解析
3. 生成两类首帧种子掩码：
   - 人物整体前景种子
   - 嘴部种子
4. 调用 `/root/AvatarStack/modules/MatAnyone` 对上述两类目标做逐帧跟踪
5. 按 `/root/AvatarStack/modules/FlashAvatar-code` 需要的目录格式整理 `imgs / alpha / parsing`
6. 调用 `/root/AvatarStack/modules/metrical-tracker` 生成 `checkpoint/*.frame`
7. 可选：写出逐帧单目深度 `depth/*.npy` 供后续 SIDL 使用

注意：

- `FlashAvatar` 训练真正必需的 `.frame` 文件由 `metrical-tracker` 生成。
- `metrical-tracker` 不是只靠视频就能运行，它还依赖 `identity.npy` 作为 MICA identity / shape 先验。
- 本阶段现在覆盖视频帧、mask、以及 tracker 输入输出目录的标准化组织。
- 由于 `BiSeNet-face-parsing` 是人脸解析模型，这里的“人物整体”在工程上等价于“头部主体前景 matte 种子”，不是通用全身实例分割。

## 关键约定

- 原始抽帧编号使用 5 位补零：`00000.jpg`, `00001.jpg`, ...
- `MatAnyone --save_image` 默认输出 4 位编号，需要在整理阶段统一重命名为 5 位编号。
- `FlashAvatar` 目录侧建议保留从 `00000` 开始的图像和 mask。
- `metrical-tracker` 配置文件名应直接命名为 `<id>.yml`，因为它的输出目录名来自配置文件 stem。
- `metrical-tracker` 需要 `begin_frames: 1`，这样它会从 `images/00001.*` 开始跟踪，最终满足：
  - `00000.frame <-> 00001.jpg`
  - 即 `len(imgs) = len(checkpoint) + 1`
- `alpha/*.jpg` 存灰度软掩码
- `parsing/*_neckhead.png` 和 `parsing/*_mouth.png` 存二值 mask
- 设计扩展里推荐增加 `depth/*.npy`，保存逐帧 `float32` 单目深度图

## BiSeNet 标签约定

```text
0  background
1  skin
2  l_brow
3  r_brow
4  l_eye
5  r_eye
6  eye_g
7  l_ear
8  r_ear
9  ear_r
10 nose
11 mouth
12 u_lip
13 l_lip
14 neck
15 neck_l
16 cloth
17 hair
18 hat
```

本方案中的首帧种子定义：

- `person_seed = union(class_id in [1..18])`
- `mouth_seed = union(class_id in [11, 12, 13])`

说明：

- `person_seed` 用于驱动人物主体 matte 跟踪，并最终生成 `alpha`
- 当前阶段的 `neckhead` 由人物主体 matte 二值化得到，用于兼容 `FlashAvatar` 目录协议
- 如果后续需要更严格的头颈语义，可在不改目录协议的前提下，把 `neckhead` 替换为更精细的 head-neck 分支

## 伪代码

```text
FUNCTION preprocess_mp4_for_flashavatar(
    input_mp4_path,
    identity_id,
    work_root,
    flashavatar_dataset_root,
    flashavatar_tracker_root,
    metrical_tracker_root,
    bisenet_model_name = "resnet18",
    bisenet_weight_path,
    matanyone_ckpt_path = "pretrained_models/matanyone.pth",
    mica_identity_path,
    mouth_threshold = 127,
    neckhead_threshold = 127
):

    DEFINE raw_frame_dir =
        work_root / identity_id / "frames_raw"

    DEFINE seed_dir =
        work_root / identity_id / "seed_masks"

    DEFINE matting_dir =
        work_root / identity_id / "matanyone"

    DEFINE final_dataset_dir =
        flashavatar_dataset_root / identity_id

    DEFINE final_imgs_dir =
        final_dataset_dir / "imgs"

    DEFINE final_alpha_dir =
        final_dataset_dir / "alpha"

    DEFINE final_parsing_dir =
        final_dataset_dir / "parsing"

    DEFINE final_checkpoint_dir =
        flashavatar_tracker_root / identity_id / "checkpoint"

    DEFINE tracker_actor_dir =
        metrical_tracker_root / "input" / identity_id

    DEFINE tracker_cfg_path =
        metrical_tracker_root / "configs" / "actors" / identity_id + ".yml"


    STEP 1: 抽帧
        read input_mp4_path
        decode all frames in original order
        save each frame as JPEG to raw_frame_dir
        filename rule: 00000.jpg, 00001.jpg, ...
        assert frame_count > 0


    STEP 2: 对首帧做 BiSeNet 解析
        first_frame_path = raw_frame_dir / "00000.jpg"

        run BiSeNet inference on first_frame_path
        obtain first_frame_parsing_raw with class ids in [0..18]

        save raw label map to:
            seed_dir / "00000_parsing_raw.png"


    STEP 3: 从首帧解析结果生成两张种子 mask
        person_seed =
            first_frame_parsing_raw != 0

        mouth_seed =
            first_frame_parsing_raw in {11, 12, 13}

        save person_seed to:
            seed_dir / "00000_person_seed.png"

        save mouth_seed to:
            seed_dir / "00000_mouth_seed.png"

        assert person_seed contains non-zero pixels
        assert mouth_seed contains non-zero pixels


    STEP 4: 用 MatAnyone 跟踪人物主体 matte
        run MatAnyone with:
            input_path = input_mp4_path
            mask_path = seed_dir / "00000_person_seed.png"
            output_path = matting_dir / "person"
            save_image = True

        expect outputs:
            matting_dir / "person" / <video_name> / "pha" / 0000.png ...
            matting_dir / "person" / <video_name> / "fgr" / 0000.png ...


    STEP 5: 用 MatAnyone 跟踪嘴部 matte
        run MatAnyone with:
            input_path = input_mp4_path
            mask_path = seed_dir / "00000_mouth_seed.png"
            output_path = matting_dir / "mouth"
            save_image = True

        expect outputs:
            matting_dir / "mouth" / <video_name> / "pha" / 0000.png ...
            matting_dir / "mouth" / "fgr" is optional for this pipeline


    STEP 6: 创建 FlashAvatar 目录
        mkdir -p final_imgs_dir
        mkdir -p final_alpha_dir
        mkdir -p final_parsing_dir
        mkdir -p final_checkpoint_dir


    STEP 7: 整理原始图像到 imgs
        FOR each frame_index in [0, frame_count - 1]:
            src = raw_frame_dir / zfill5(frame_index) + ".jpg"
            dst = final_imgs_dir / zfill5(frame_index) + ".jpg"
            copy or re-encode src -> dst


    STEP 8: 整理人物主体 matte 到 alpha 与 neckhead
        FOR each frame_index in [0, frame_count - 1]:
            person_pha_src =
                matting_dir / "person" / <video_name> / "pha" / zfill4(frame_index) + ".png"

            person_pha =
                read grayscale image from person_pha_src

            alpha_soft =
                keep uint8 grayscale range [0, 255]

            neckhead_binary =
                person_pha >= neckhead_threshold

            save alpha_soft as JPEG to:
                final_alpha_dir / zfill5(frame_index) + ".jpg"

            save neckhead_binary as binary PNG to:
                final_parsing_dir / zfill5(frame_index) + "_neckhead.png"


    STEP 9: 整理嘴部 matte 到 parsing
        FOR each frame_index in [0, frame_count - 1]:
            mouth_pha_src =
                matting_dir / "mouth" / <video_name> / "pha" / zfill4(frame_index) + ".png"

            mouth_pha =
                read grayscale image from mouth_pha_src

            mouth_binary =
                mouth_pha >= mouth_threshold

            save mouth_binary as binary PNG to:
                final_parsing_dir / zfill5(frame_index) + "_mouth.png"


    STEP 10: 准备 metrical-tracker 输入目录
        mkdir -p tracker_actor_dir

        copy input_mp4_path to:
            tracker_actor_dir / "video.mp4"

        copy mica_identity_path to:
            tracker_actor_dir / "identity.npy"

        note:
            metrical-tracker will internally generate:
                source/
                images/
                kpt/
                kpt_dense/


    STEP 11: 生成 metrical-tracker 配置
        write tracker_cfg_path with at least:
            actor: "./input/<id>"
            save_folder: "./output/"
            begin_frames: 1
            optimize_shape: true
            optimize_jaw: true

        ensure:
            basename_without_ext(tracker_cfg_path) == identity_id

        because:
            tracker output folder name = config file stem


    STEP 12: 运行 metrical-tracker
        cd metrical_tracker_root
        run:
            python tracker.py --cfg ./configs/actors/<id>.yml

        expect outputs:
            metrical_tracker_root / "output" / identity_id / "checkpoint" / 00000.frame ...
            metrical_tracker_root / "output" / identity_id / "mesh" / 00000.ply ...
            metrical_tracker_root / "output" / identity_id / "depth" / ...


    STEP 13: 整理 checkpoint 到 FlashAvatar 路径
        tracker_checkpoint_src =
            metrical_tracker_root / "output" / identity_id / "checkpoint"

        copy all *.frame from tracker_checkpoint_src to final_checkpoint_dir

        validate:
            checkpoint_count + 1 == frame_count
            00000.frame aligns with 00001.jpg
            00000.frame contains flame, camera, opencv, img_size


    STEP 14: 一致性校验
        assert count(final_imgs_dir/*.jpg) == frame_count
        assert count(final_alpha_dir/*.jpg) == frame_count
        assert count(final_parsing_dir/*_neckhead.png) == frame_count
        assert count(final_parsing_dir/*_mouth.png) == frame_count
        assert count(final_checkpoint_dir/*.frame) == frame_count - 1

        FOR each frame_index in [0, frame_count - 1]:
            assert img exists
            assert alpha exists
            assert neckhead exists
            assert mouth exists
            assert image size of all four files is identical

        sample-check frames:
            first
            middle
            last

        visually verify:
            alpha edge stability
            mouth coverage stability
            neckhead foreground does not collapse
            tracker pose and expression are synchronized


    STEP 15: 输出预处理结果
        RETURN {
            "identity_id": identity_id,
            "frame_count": frame_count,
            "dataset_dir": final_dataset_dir,
            "tracker_checkpoint_dir": final_checkpoint_dir,
            "status": "flashavatar_ready"
        }
```

## 最终目录目标

```text
dataset/
  <id>/
    imgs/
      00000.jpg
      00001.jpg
      ...
    alpha/
      00000.jpg
      00001.jpg
      ...
    parsing/
      00000_neckhead.png
      00000_mouth.png
      00001_neckhead.png
      00001_mouth.png
      ...

metrical-tracker/
  output/
    <id>/
      checkpoint/
        00000.frame
        00001.frame
        ...
```

## 阶段完成定义

### 状态 A: `mask_ready_tracker_pending`

满足：

- `imgs / alpha / parsing` 已完整生成
- 编号、尺寸、数量一致
- 但 `identity.npy` 缺失，或 `metrical-tracker` 尚未运行完成

说明：

- 可直接进入 MICA identity 准备或 tracker 执行阶段

### 状态 B: `flashavatar_ready`

满足：

- `imgs / alpha / parsing / checkpoint` 全部存在
- `checkpoint` 由 `/root/AvatarStack/modules/metrical-tracker` 生成
- `checkpoint_count + 1 == image_count`
- `00000.frame <-> 00001.jpg` 对齐规则成立
- 可被 `FlashAvatar` 的 `Scene_mica` 正常消费

## 可选扩展：单目深度监督准备

如果后续要引入 **Scale-Invariant Depth Loss (SIDL)**，推荐在预处理阶段额外生成：

```text
dataset/
  <id>/
    depth/
      00000.npy
      00001.npy
      ...
```

约定：

- 深度来源：`VideoDepth Anything`
- 文件格式：`float32 .npy`
- 编号：与 `imgs/*.jpg` 一一对应
- 分辨率：与 `imgs / alpha / parsing` 完全一致

推荐的深度监督有效区域：

```text
valid_depth_mask
  = erode(neckhead_mask, 2~3 px)
    - mouth_inner_mask
```

设计意图：

- 不把单目深度拟合成显式 mesh
- 只在核心脸部实体区域施加深度相关性约束
- 回避头发边缘、下颌线边缘的 depth bleeding
- 回避张嘴时口腔内部的错误深度监督
