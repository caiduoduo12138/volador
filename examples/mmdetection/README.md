# MMDetection

我们提供了一些示例的yaml(实验配置文件)作为示例，可以在[这里](https://github.com/caiduoduo12138/volador/blob/master/examples/mmdetection.zip)找到。

## 概述

[mmdetection](https://github.com/open-mmlab/mmdetection)是一个开源的目标检测框架，包含了大量的目标检测与分割算法。它通过配置文件的方式，可以完成模型的搭建，实现快速组合backbone、neck、head以及训练中的optimizer和学习率调度等其他参数。为了保留这一特性，飞鱼集群的作业系统尽可能地进行兼容，减少代码修改。如果用户熟悉mmdetection，可快速地利用飞鱼集群的作业系统调用mmdetction的相关API进行实验。如果用户对mmdetection不熟悉，请参考[官方文档](https://mmdetection.readthedocs.io/en/latest/)进行了解。

## 指定MMDetection的配置文件

以我们提供的示例文件[fasterrcnn.yaml](https://github.com/caiduoduo12138/volador/blob/master/examples/mmdetection.zip)文件为例，用户可以通过更改yaml中的字段`config_file`来选择mmdetection的配置文件。运行作业的容器中，mmdetetion的目录被配置在`/mmdetection`下例如：

```
hyperparameters:
  config_file: /mmdetection/configs/faster_rcnn/faster_rccn_r50_fpn_1x_coco.py
```

## 覆盖一个配置文件

熟悉mmdetection的用户知道，mmdetection可以通过覆盖配置文件的方式对模型的各个结构或参数进行修改。假设新建的配置文件名为`faster_rcnn_r101.py`，该文件放置在工程文件目录下()，，通过如下方式进行覆盖：

```
hyperparameters:
  global_batch_size: 16
  config_file: /mmdetection/configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_1x_coco.py
  merge_config: ./faster_rcnn_r101.py
```

```
注意：原始的配文件faster_rcnn_r50_caffe_fpn_1x_coco.py在作业容器内部的路径
/mmdetection/configs/faster_rcnn/下，而新建的配置文件在工程目录下，一般是在
飞鱼集群的分布式存储中，示例的faster_rcnn_r101.py是与yaml在同一级目录。
```

## 其他yaml字段

`global_batch_size`必须被指定，用来创建实验。如果你想与mmdetection配置文件中的设置保持一致，请修改配置文件。用户可进作业镜像进行修改，例如，增加自定义数据集、设置新的模块结构等(不推荐刚使用mmdetection和不熟悉飞鱼集群作业镜像的用户进行该操作，以防止不可预知的错误)。

更多mmdetection的用法请参考[官方文档](https://mmdetection.readthedocs.io/en/latest/)。


