## mmdetection训练Cascade RCNN
## Cascade RCNN 训练自己的数据

**Xu Jing**

商汤科技（2018 COCO 目标检测挑战赛冠军）和香港中文大学最近开源了一个基于Pytorch实现的深度学习目标检测工具箱mmdetection，支持Faster-RCNN，Mask-RCNN，Fast-RCNN，Cascade-RCNN等主流目标检测框架。可以快速部署自己的模型。

项目地址：<https://github.com/open-mmlab/mmdetection>

官方教程：<https://mmdetection.readthedocs.io](https://mmdetection.readthedocs.io/>

paper: <https://arxiv.org/abs/1906.07155>

![](coco_test_12510.jpg)

### 2.环境要求

1. Linux (官方不支持windows，但是我们可以看到网上关于在windows安装mmdetection的教程)

2. Python 3.5+

3. \>=PyTorch 1.1.0, torchvision 0.3.0

4. \>=CUDA 9.0

5. NCCL 2

6. \>=GCC 4.9

7. mmcv


### 1.环境安装

按照官方文档建议先安装Anaconda,创建python虚拟环境,使用conda进行安装,这里我们使用virtualenv安装

1.virtualenv创建一个虚拟环境

```
virtualenv -p python3 mmlab
cd mmlab/bin
source activate

```

2.安装pytorch和torchvision

```
https://pytorch.org/ 下载安装
https://download.pytorch.org/whl/torch_stable.html
pip install torch==1.1.0 torchvision==0.3.0 -f https://download.pytorch.org/whl/torch_stable.html
# conda install pytorch==1.1.0 torchvision==0.3.0
```

3.下载mmdetection

```
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
```

4.安装mmdetection

```
pip3 install mmcv cython -i https://pypi.tuna.tsinghua.edu.cn/simple
#安装mmcv和cython
pip3 install albumentations>=0.3.2 imagecorruptions pycocotools six terminaltables 
#安装依赖包
python3 setup.py develop 
# 在root用户下做，发现自己ubuntu16.04不在root用户下做报错
# 必须先安装mmcv，再运行setup.py编译,不然会报错。
```

### 2.验证是否安装成功

下载一个faster_rcnn_r50_fpn_1x的[预训练模型](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth)，保存到mmdetection/checkpoints目录下,运行下面的代码,如果能显示图片，说明安装成功了。

```python
from mmdet.apis import init_detector, inference_detector, show_result
import mmcv

config_file = 'configs/faster_rcnn_r50_fpn_1x.py'
checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
img = 'test.jpg'  # or img = mmcv.imread(img), which will only load it once
result = inference_detector(model, img)
# visualize the results in a new window
# show_result(img, result, model.CLASSES)
# or save the visualization results to image files
show_result(img, result, model.CLASSES,score_thr=0.90,show=False,out_file='result.jpg')
```

![](test_fast_rcnn.jpg)

这样我们就完成mmdetection的安装！

### 3.构建训练集

1.创建相应文件夹

+ ./config/bingzao： 模型训练的配置文件存放地址
  - 将cascade_rcnn_r101_fpn_1x.py文件存放在此，并对其进行修改

+ data： 训练数据的的存放地址

  ```
  ./data
  ├─coco
  │  ├─annotations   # 存放train.json,val.json,test.json
  │  ├─test          # 测试或集
  │  │  ├─annotations  # 测试或验证xml标注
  │  │  └─JPEGImages  # 测试或验证图片
  │  └─train         # 训练集
  │      ├─annotations # 训练集的xml标注
  │      └─JPEGImages  # 训练集的图片
  ├─pretrained    # 预训练模型的存放地址
  ├─results       # 测试结果的存放地址，用于测试
  └─source        # 待处理的数据存放地址，将最终的数据检查后存放在coco文件夹
      ├─test
      │  ├─annotations
      │  └─JPEGImages
      └─train
          ├─annotations
          └─JPEGImages
  ```

+ work_dirs： 用于保存训练模型的模型文件和训练log

+ checkpoint: 用于保存预训练的模型（这里我们并没有使用）

2.训练集准备

往往我们拿到的数据集都是基于VOC数据格式的数据，有xml标注文件和图像源文件，我们将获得的源数据存放在`./data/source/`文件下。

3.修改代码将VOC数据转为COCO数据

可以参考[这个代码](https://github.com/spytensor/prepare_detection_dataset)，将自己的数据转换为coco格式，它支持：

+ csv to coco
+ csv to voc
+ labelme to coco
+ labelme to voc
+ csv to json

**A.新建修改mmdetection/mmdet/datasets/bingzao.py**

```
# 其结构与mmdetection/mmdet/datasets/coco.py相似，但类名和CLASSES不同
@DATASETS.register_module
class bingzao(CustomDataset):  # 类名修改成bingzao
	# 修改CLASSES，修改成自己的
    CLASSES = ("Barrett","CX","FLXSGY","HJQ","JCJZQA","JCXR",
        "JCZA","JZQWA","JS","KYXJCY","MXWSXWY","QP","QG","QTMH",
        "QTQPGY","SGJMQZ","SGZA","TW","WKY","WZA","YD","ZZ")

```

同时修改同级目录下的`__init__.py`

```
from .builder import build_dataset
from .cityscapes import CityscapesDataset
from .coco import CocoDataset
from .custom import CustomDataset
from .dataset_wrappers import ConcatDataset, RepeatDataset
from .loader import DistributedGroupSampler, GroupSampler, build_dataloader
from .registry import DATASETS
from .voc import VOCDataset
from .wider_face import WIDERFaceDataset
from .xml_style import XMLDataset
from .bingzao import bingzao

__all__ = [
    'CustomDataset', 'XMLDataset', 'CocoDataset', 'VOCDataset',
    'CityscapesDataset', 'GroupSampler', 'DistributedGroupSampler',
    'build_dataloader', 'ConcatDataset', 'RepeatDataset', 'WIDERFaceDataset',
    'DATASETS', 'build_dataset',"bingzao" #在此添加
]

```

**B.新建修改mmdetection/mmdet/tools/data_process**

```
./tools
└─data_process
        00_img_rename.py  # 通过uuid重命名训练集，测试集copy到coco文件夹
        01_check_img.py  # 检查数据的合规性
        02_check_box.py  # 检查标注的合规性
        03_xml2coco.py   # VOC数据转COCO数据
        generate_test_json.py  # test无xml,随机生成COCO,方便后期测试测试集
```



**C.修改mmdetection/mmdet/core/evaluation下的`__init__.py`,class_names.py**

```
# class_names.py
# 新增类
def bingzao_classes():
    return [
        "Barrett","CX","FLXSGY","HJQ","JCJZQA","JCXR","JCZA","JZQWA",
        "JS","KYXJCY","MXWSXWY","QP","QG","QTMH","QTQPGY","SGJMQZ",
        "SGZA","TW","WKY","WZA","YD","ZZ"
    ]
```

```
# __init__.py 修改
from .class_names import (cityscapes_classes, coco_classes, dataset_aliases,
                          get_classes, imagenet_det_classes,
                          imagenet_vid_classes, voc_classes,bingzao_classes)
from .eval_hooks import DistEvalHook
from .mean_ap import average_precision, eval_map, print_map_summary
from .recall import (eval_recalls, plot_iou_recall, plot_num_recall,
                     print_recall_summary)

__all__ = [
    'voc_classes', 'imagenet_det_classes', 'imagenet_vid_classes',
    'coco_classes', 'cityscapes_classes', 'dataset_aliases', 'get_classes',
    'DistEvalHook', 'average_precision', 'eval_map', 'print_map_summary',
    'eval_recalls', 'print_recall_summary', 'plot_num_recall',
    'plot_iou_recall',"bingzao_classes" # 添加这个类
]
```



4.下载预训练的模型

在[model Zoo](https://github.com/open-mmlab/mmdetection/blob/master/docs/MODEL_ZOO.md)下载我们需要的模型，下载好的预训练模型将其存放在新建的`./checkpoint`（官方推荐）文件夹或`./data/pretrained`文件夹，这个取决我们在Section4中配置文件的配置，我们将下载的`cascade_rcnn_r101_fpn_1x_20181129-d64ebac7.pth` COCO预训练的模型存放在`./data/pretrained`文件夹。

5. 生成COCO数据

   准备好上述数据后，运行

   ```
   python ./tools/data_process/00_img_rename.py  
   python ./tools/data_process/01_check_img.py  
   python ./tools/data_process/02_check_box.py  
   python ./tools/data_process/03_xml2coco.py
   ```

   最终在`./data/coco/annotations/`下生成了train.json和test.json,用以训练模型做最后的数据准备！

### 4.修改mmdetection的模型config文件

修改`./configs/bingzao/cascade_rcnn_r101_fpn_1x.py`

<details>
  <summary>展开我查看：cascade_rcnn_r101_fpn_1x.py</summary>
  <pre><blockcode> 
# model settings
model = dict(
    type='CascadeRCNN',
    num_stages=3,
    pretrained='torchvision://resnet101',
    backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_scales=[8],
        anchor_ratios=[0.5, 1.0, 2.0],
        anchor_strides=[4, 8, 16, 32, 64],
        target_means=[.0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0],
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    bbox_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32]),
    bbox_head=[
        dict(
            type='SharedFCBBoxHead',
            num_fcs=2,
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=23,  #----------- 修改类别个数81 类别数量+1----------
            target_means=[0., 0., 0., 0.],
            target_stds=[0.1, 0.1, 0.2, 0.2],
            reg_class_agnostic=True,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
        dict(
            type='SharedFCBBoxHead',
            num_fcs=2,
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=81,
            target_means=[0., 0., 0., 0.],
            target_stds=[0.05, 0.05, 0.1, 0.1],
            reg_class_agnostic=True,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
        dict(
            type='SharedFCBBoxHead',
            num_fcs=2,
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=81,
            target_means=[0., 0., 0., 0.],
            target_stds=[0.033, 0.033, 0.067, 0.067],
            reg_class_agnostic=True,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
    ])
# model training and testing settings
train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    rpn_proposal=dict(
        nms_across_levels=False,
        nms_pre=2000,
        nms_post=2000,
        max_num=2000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=[
        dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False),
        dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.6,
                neg_iou_thr=0.6,
                min_pos_iou=0.6,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False),
        dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.7,
                min_pos_iou=0.7,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)
    ],
    stage_loss_weights=[1, 0.5, 0.25])
test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=1000,
        nms_post=1000,
        max_num=1000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(  #-------------修改一些后处理的参数NMS,WBF,Soft NMS-------------
        score_thr=0.0001, nms=dict(type='soft_nms', iou_thr=0.5,min_score=0.0001), max_per_img=200))
# dataset settings
dataset_type = 'bingzao'  #---------修改数据集名称-----------
data_root = 'data/coco/'  #---------修改数据的根目录---------
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    #-------------训练数据增强在此添加操作---------------------
    # https://blog.csdn.net/Mr_health/article/details/103552617?depth_1-utm_source=distribute.pc_relevant.none-task&utm_source=distribute.pc_relevant.none-task
    dict(type='Resize', img_scale=[(1920,1080),(1280, 1024),(1024,768),(1528,1036),(720,576)], keep_ratio=True,multiscale_mode='value'),  
#-----修改多尺度训练dict(type='Resize', img_scale=[(4096, 600), (4096, 1000)],multiscale_mode='range', keep_ratio=True),--------
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        #---------多尺度推断在此修改------------
        img_scale=[(1920,1080),(1280, 1024),(1024,768),(1528,1036),(720,576)],  
        #--------img_scale=[(4096, 600), (4096, 800), (4096, 1000)],--------
        flip=True, # 默认是False
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    imgs_per_gpu=4,    #  -----每个GPU计算的图像数量-----
    workers_per_gpu=2, # -----每个GPU分配的线程数-----
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/train.json',  # ---标注的annotation路径---
        img_prefix=data_root + 'train/JPEGImages',  #---数据就的图片路径---
        pipeline=train_pipeline),
    #-----这里我们没有分配验证集，如果分配可加入----------------
    # val=dict(
    #     type=dataset_type,
    #     ann_file=data_root + 'annotations/instances_val2017.json',
    #     img_prefix=data_root + 'val2017/',
    #     pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/test.json',  # ----没有的话可以随机生成的---
        img_prefix=data_root + 'test/JPEGImages',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')
# ------optimizer------
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',  # -----warmup的策略，这里设置为线性增加------
    warmup_iters=500,  # ----在初始的500次迭代中学习率逐渐增加-----
    warmup_ratio=1.0 / 30, # -----起始的学习率1.0/3-------
    step=[70, 90])   #-----在第8和11个epoch时降低学习率------
checkpoint_config = dict(interval=20) #-----每n个epoch存储一次模型------
# yapf:disable
log_config = dict(
    interval=20,  #-----每20iter报告一次训练的log-------
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook') #-----打开可以使用tensorboard------
    ])
# yapf:enable
# runtime settings
total_epochs = 100  # ------训练的epoch-----
dist_params = dict(backend='nccl') # -----分布式参数-----
log_level = 'INFO'
work_dir = './work_dirs/cascade_rcnn_r101_fpn_1x' #----训练过程中模型和训练log的保存地址--
# load_from = None   # ----加载模型的路径，None表示从预训练模型加载---
#-----预训练模型的地址，我们在section3已经下载存放好
#load_form = "./checkpoint/cascade_rcnn_r101_fpn_1x.py"
load_from = "data/pretrained/cascade_rcnn_r101_fpn_1x_20181129-d64ebac7.pth"
resume_from = None         # -----恢复训练模型的路径,用于断点训练-----
workflow = [('train', 1)]  # ------当前工作区的名称-------
  </blockcode></pre>
</details>





### 5.训练Cascade RCNN

1.单GPU训练

```
#python tools/train.py ${模型配置文件}
source ./mmlab/bin/activate
python tools/train.py configs/bingzao/cascade_rcnn_r101_fpn_1x.py
```

2.多GPU训练

```
#./tools/dist_train.sh ${模型配置文件} ${GPU数量} [可选]
./tools/dist_trian.sh  configs/bingzao/cascade_rcnn_r101_fpn_1x.py 4
```

训练完之后work_dirs文件夹中会保存训练过程中的log日志文件、保存的间隔周期的pth文件（这个文件将会用于后面的test测试）

### 6.测试Cascade RCNN

**TODO**

- [ ]  单尺度推断
- [ ] 多尺度推断
- [ ] 单GPU测试
- [ ] 多GPU测试

### 7.Citation

```
@article{mmdetection,
  title   = {{MMDetection}: Open MMLab Detection Toolbox and Benchmark},
  author  = {Chen, Kai and Wang, Jiaqi and Pang, Jiangmiao and Cao, Yuhang and
             Xiong, Yu and Li, Xiaoxiao and Sun, Shuyang and Feng, Wansen and
             Liu, Ziwei and Xu, Jiarui and Zhang, Zheng and Cheng, Dazhi and
             Zhu, Chenchen and Cheng, Tianheng and Zhao, Qijie and Li, Buyu and
             Lu, Xin and Zhu, Rui and Wu, Yue and Dai, Jifeng and Wang, Jingdong
             and Shi, Jianping and Ouyang, Wanli and Loy, Chen Change and Lin, Dahua},
  journal= {arXiv preprint arXiv:1906.07155},
  year={2019}
}
```

```
https://github.com/python-bookworm/mmdetection-new
```

```
https://github.com/zhengye1995/underwater-objection-detection
```

