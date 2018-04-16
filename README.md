# A MXNet/Gluon implementation of MobileNetV2

This is a Gluon implementation of MobileNetV2 architecture as described in the paper [Inverted Residuals and Linear Bottlenecks: Mobile Networks for Classification, Detection and Segmentation](https://arxiv.org/pdf/1801.04381).

### Pretrained Models on ImageNet

We provide pretrained MobileNet models on ImageNet, which achieve slightly better accuracy rates than the original ones reported in the paper. 

The top-1/5 accuracy rates by using single center crop (crop size: 224x224, image size: 256xN):

Network|Top-1|Top-5|
:---:|:---:|:---:|
MobileNet v2| 72.45| 90.78

### Tranning Script
For tranning, please refer to [this repo](https://github.com/liangfu/mxnet-mobilenet-v2).Our pretrained model is converted from that.

### Normalization

The input images are substrated by mean RGB = [ 123.68, 116.78, 103.94 ].