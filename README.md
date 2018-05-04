# A MXNet/Gluon implementation of MobileNetV2

This is a Gluon implementation of MobileNetV2 architecture as described in the paper [Inverted Residuals and Linear Bottlenecks: Mobile Networks for Classification, Detection and Segmentation](https://arxiv.org/pdf/1801.04381).

### Pretrained Models on ImageNet

We provide pretrained MobileNetV2(width_mult: 1.0, crop_size: 224) models on ImageNet, which achieve slightly better accuracy rates than the original ones reported in the paper. 

The top-1/5 accuracy rates by using single center crop (crop size: 224x224, image size: 256xN):

Network|Top-1|Top-5|
:---:|:---:|:---:|
MobileNet v2| 71.72 | 90.13

### Training Script
For training, please refer to [liangfu/mxnet-mobilenet-v2](https://github.com/liangfu/mxnet-mobilenet-v2). Our pretrained model is converted from that.
The MXNet officially offer a pretrained model (and training script) [here](https://github.com/apache/incubator-mxnet/blob/master/docs/api/python/gluon/model_zoo.md) too.

### Normalization

The input images are substrated by mean RGB = [ 123.68, 116.78, 103.94 ].
