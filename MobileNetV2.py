import mxnet as mx
import mxnet.ndarray as nd
import mxnet.gluon as gluon
import mxnet.gluon.nn as nn
import mxnet.autograd as ag

class RELU6(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(RELU6, self).__init__(**kwargs)

    def hybrid_forward(self, F, x):
        return F.clip(x, 0, 6, name="fwd")

def ConvBlock(channels, kernel_size, strides, **kwargs):
    out = nn.HybridSequential(**kwargs)
    with out.name_scope():
        out.add(
            nn.Conv2D(channels, kernel_size, strides=strides, padding=1, use_bias=False),
            nn.BatchNorm(scale=True),
            RELU6(prefix="relu6_")
        )
    return out

def Conv1x1(channels, is_linear=False, **kwargs):
    out = nn.HybridSequential(**kwargs)
    with out.name_scope():
        out.add(
            nn.Conv2D(channels, 1, padding=0, use_bias=False),
            nn.BatchNorm(scale=True)
        )
        if not is_linear:
            out.add(RELU6(prefix="relu6_"))
    return out

def DWise(channels, strides, **kwargs):
    out = nn.HybridSequential(**kwargs)
    with out.name_scope():
        out.add(
            nn.Conv2D(channels, 3, strides=strides, padding=1, groups=channels, use_bias=False),
            nn.BatchNorm(scale=True),
            RELU6(prefix="relu6_")
        )
    return out

class ExpandedConv(nn.HybridBlock):
    def __init__(self, inp, oup, t, strides, same_shape=True, **kwargs):
        super(ExpandedConv, self).__init__(**kwargs)
        self.same_shape = same_shape
        self.strides = strides
        with self.name_scope(): 
            self.bottleneck = nn.HybridSequential()
            self.bottleneck.add(
                Conv1x1(inp*t, prefix="expand_"),
                DWise(inp*t, self.strides, prefix="dwise_"),
                Conv1x1(oup, is_linear=True, prefix="linear_")
            )
    def hybrid_forward(self, F, x):
        out = self.bottleneck(x)
        if self.strides == 1 and self.same_shape:
            out = F.elemwise_add(out, x)
        return out

def ExpandedConvSequence(t, inp, oup, repeats, first_strides, **kwargs):
    seq = nn.HybridSequential(**kwargs)
    with seq.name_scope():
        seq.add(ExpandedConv(inp, oup, t, first_strides, same_shape=False))
        curr_inp = oup
        for _ in range(1, repeats):
            seq.add(ExpandedConv(curr_inp, oup, t, 1))
            curr_inp = oup
    return seq

class MobilenetV2(nn.HybridBlock):
    def __init__(self, num_classes=1000, width_mult=1.0, **kwargs):
        super(MobilenetV2, self).__init__(**kwargs)
        
        self.w = width_mult

        self.first_oup = 32 * self.w
        self.interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1, "stage0_"],  # -> 112x112
            [6, 24, 2, 2, "stage1_"],  # -> 56x56
            [6, 32, 3, 2, "stage2_"],  # -> 28x28
            [6, 64, 4, 2, "stage3_0_"],  # -> 14x14
            [6, 96, 3, 1, "stage3_1_"],  # -> 14x14
            [6, 160, 3, 2, "stage4_0_"], # -> 7x7
            [6, 320, 1, 1, "stage4_1_"], # -> 7x7          
        ]
        self.last_channels = int(1280*self.w) if self.w > 1.0 else 1280

        with self.name_scope():
            self.features = nn.HybridSequential()
            self.features.add(ConvBlock(self.first_oup, 3, 2, prefix="stage0_"))
            inp = self.first_oup
            for t, c, n, s, prefix in self.interverted_residual_setting:
                oup = c * self.w
                self.features.add(ExpandedConvSequence(t, inp, oup, n, s, prefix=prefix))
                inp = oup

            self.features.add(Conv1x1(self.last_channels, prefix="stage4_2_"))
            self.features.add(nn.GlobalAvgPool2D())
            self.features.add(nn.Flatten())
            self.output = nn.Dense(num_classes)
    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x

net = MobilenetV2(1000,1, prefix="")

# save as symbol
data =mx.sym.var('data')
sym = net(data)

# print network summary
#mx.viz.print_summary(sym, shape={'data':(8,3,224,224)})

# plot network graph
#mx.viz.plot_network(sym,shape={'data':(8,3,224,224)}, node_attrs={'shape':'oval','fixedsize':'fasl==false'}).view()


