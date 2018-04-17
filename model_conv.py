import json
import mxnet as mx
from mxnet import nd
from MobileNetV2 import MobilenetV2

# convert model from https://github.com/yuantangliang/MobileNet-v2-Mxnet

net = MobilenetV2(1000,1, prefix="")

data = mx.sym.var('data')
sym = net(data)

sym2, arg_params, aux_params = mx.model.load_checkpoint('mobilenetv2-1_0',0)


conf = json.loads(sym.tojson())
nodes = conf["nodes"]

conf2 = json.loads(sym2.tojson())
nodes2 = conf2["nodes"]
new_arg_params = {}
new_aux_params = {}
for i, n in enumerate(nodes):
    if nodes2[i]['name'] == 'prob_label':
        break
    if n['op'] == 'Convolution' or n['op'] == 'FullyConnected':
        #print(n['op'], nodes[i]['name'][:-4], nodes2[i]['name'])
        new_arg_params[nodes[i]['name'][:-4]+'_weight'] = arg_params[nodes2[i]['name']+'_weight']
        if not 'no_bias' in n['attrs'] or n['attrs']['no_bias'] == 'False':
            new_arg_params[nodes[i]['name'][:-4]+'_bias'] = arg_params[nodes2[i]['name']+'_bias']
    elif n['op'] == 'BatchNorm':
        #print(nodes[i]['name'])
        new_arg_params[nodes[i]['name'][:-4]+'_gamma'] = arg_params[nodes2[i]['name']+'_gamma']
        new_arg_params[nodes[i]['name'][:-4]+'_beta'] = arg_params[nodes2[i]['name']+'_beta']
        new_aux_params[nodes[i]['name'][:-4]+'_running_mean'] = aux_params[nodes2[i]['name']+'_moving_mean']
        new_aux_params[nodes[i]['name'][:-4]+'_running_var'] = aux_params[nodes2[i]['name']+'_moving_var']
    else:
        pass

new_arg_params['dense0_weight'] = arg_params['fc_weight'], axis=(2,3)
new_arg_params['dense0_bias'] = arg_params['fc_bias']


model = mx.mod.Module(symbol=sym, data_names=['data'], label_names=None)
model.bind(data_shapes=[('data', (1, 3, 224, 224))])
model.set_params(arg_params=new_arg_params, aux_params=new_aux_params, allow_missing=False)
model.save_checkpoint('mbnv2', 0)