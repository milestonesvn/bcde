from kaos.distributions import Distribution as Dist
from kaos.distributions import gaussian_sampler, infer_ladder
from kaos.layers import BatchNormalization
from keras.layers import Input, Dense, merge, Lambda, Merge, Activation
from keras.regularizers import l2 as l2_reg
from keras.models import Sequential, Model

def concat(*args):
    return merge(args, mode='concat')

def linear(output_size, activation=None, bn=False, l2=0.0, input_dim=None):
    if l2 == 0.0:
        sequence = [Dense(output_size, input_dim=input_dim)]
    else:
        sequence = [Dense(output_size, W_regularizer=l2_reg(l2), input_dim=input_dim)]
    if bn:
        sequence += [BatchNormalization()]
    if activation is not None:
        sequence += [Activation(activation)]
    return sequence

def build_mlp(layers, activation='relu', bn=False, l2=0.0, headless=False):
    seq = []
    for i in xrange(1, len(layers[:-1])):
        seq += linear(layers[i], activation=activation, bn=bn, l2=l2, input_dim=layers[i-1])
    if headless:
        seq += linear(layers[-1], activation=None, bn=False, l2=l2, input_dim=layers[-2])
    else:
        seq += linear(layers[-1], activation=activation, bn=bn, l2=l2, input_dim=layers[-2])
    return Sequential(seq)

def build_inference():
    dist = Dist()
    dist.sampler = gaussian_sampler
    def call(self, l_par, p_par):
        return infer_ladder(l_par, p_par)
    dist.set_callback(call, bind=True)
    return dist

def build_distribution(layers, dist_type, activation='relu', bn=False, dist_bn=False, l2=0.0):
    dist = Dist()
    seq = []
    for i in xrange(1, len(layers[:-1])):
        seq += linear(layers[i], activation=activation, bn=bn, l2=l2, input_dim=layers[i-1])

    if layers is () and dist_type == 'gaussian':
        dist.sampler = gaussian_sampler
        dist.set_callback(lambda: (0, 1))

    elif dist_type == 'gaussian':
        mu_seq = linear(layers[-1], activation=None, bn=dist_bn, l2=l2, input_dim=layers[-2])
        var_seq = linear(layers[-1], activation='softplus', bn=dist_bn, l2=l2, input_dim=layers[-2])

        dist.sampler = gaussian_sampler
        dist.net = Sequential(seq) if len(seq) > 0 else None
        dist.mu = Sequential(mu_seq)
        dist.var = Sequential(var_seq)
        def call(self, x):
            h_out = x if self.net is None else self.net(x)
            mu = self.mu(h_out)
            var = self.var(h_out)
            return (mu, var)
        dist.set_callback(call, bind=True)

    elif dist_type == 'bernoulli':
        seq += linear(layers[-1], activation='sigmoid', bn=False, l2=l2, input_dim=layers[-2])
        dist.net = Sequential(seq)
        dist.set_callback(dist.net)

    return dist

def create_name(dic, keys):
    assert dic['folder'] is not None
    ignore = dic['ignore']
    ignore.update(['folder', 'ignore'])
    name = ''
    for k in keys:
        if k not in dic or k in ignore:
            continue
        name += '{:s}='.format(k)
        val = dic[k]
        if isinstance(val, bool):
            name += '{:b}'
        elif isinstance(val, int):
            name += '{:d}'
        elif isinstance(val, float):
            name += '{:.1e}'
        elif isinstance(val, str):
            name += '{:s}'
        else:
            raise Exception('Instance of type {:s} not expected'.format(type(val)))
        name = name.format(val)
        name += '_'
    return name[:-1]
