from models.utils import build_mlp, build_inference, build_distribution, create_name
from models.single import Single, LikelihoodSingle
from models.double import LadderDouble, LikelihoodLadderDouble
from models.vae import VAE, LikelihoodVAE, LadderDoubleVAE
from kaos.utils import file_handle, Session
from kaos.callbacks import NegativeLogLikelihood, LossLog
from keras.optimizers import Adam, SGD
import numpy as np
import sys
import os
SCRATCH = '/scratch/users/rshu15/Documents/github/cde2/07_double'

def single_network(factored, bn, dist_bn, z, x, y):
    mlp = lambda shape: build_mlp(shape, bn=bn)
    dist = lambda shape, dist_type: build_distribution(shape, dist_type, bn=bn, dist_bn=dist_bn)
    shapex = {'x': (x,)}
    shapey = {'x': (y,)}
    shape = {'x': (x,), 'y': (y,)}
    u, l, p, q = {}, {}, {}, {}
    u['x'] = mlp((x, 500, 500))
    u['y'] = mlp((y, 500, 500))
    p['z'] = dist((), 'gaussian')
    p['y'] = dist((z, 500, 500, y), 'bernoulli')
    p['x'] = dist((z, 500, 500, x), 'bernoulli')
    if not factored:
        u['xy'] = mlp((784, 500, 500))
        q['z|xy'] = dist((500, z), 'gaussian')
        q['z|x'] = dist((500, z), 'gaussian')
        q['z|y'] = dist((500, z), 'gaussian')
        x_nets = {'u_net': {'x': u['x']},
                  'q_net': {'z': q['z|x']},
                  'p_net': {'z': p['z'], 'x': p['x']}}
        y_nets = {'u_net': {'x': u['y']},
                  'q_net': {'z': q['z|y']},
                  'p_net': {'z': p['z'], 'x': p['y']}}
        c_nets = {'u_net': u, 'p_net': p, 'q_net': q}
        vaex = VAE(validate_network=False, shape=shapex, **x_nets)
        vaey = VAE(validate_network=False, shape=shapey, **y_nets)
        cvae = Single(validate_network=False, shape=shape, **c_nets)
    else:
        l['z|y'] = dist((500, z), 'gaussian')
        q['z|x'] = dist((500, z), 'gaussian')
        q['z|xy'] = build_inference()
        q['z|y'] = build_inference()
        x_nets = {'u_net': {'x': u['x']},
                  'q_net': {'z': q['z|x']},
                  'p_net': {'z': p['z'], 'x': p['x']}}
        y_nets = {'u_net': {'x': u['y']},
                  'l_net': {'z': l['z|y']},
                  'q_net': {'z': q['z|y']},
                  'p_net': {'z': p['z'], 'x': p['y']}}
        c_nets = {'u_net': u, 'l_net': l, 'p_net': p, 'q_net': q}
        vaex = VAE(validate_network=False, shape=shapex, **x_nets)
        vaey = LikelihoodVAE(validate_network=False, shape=shapey, **y_nets)
        cvae = LikelihoodSingle(validate_network=False, shape=shape, **c_nets)
    return vaex, vaey, cvae, (u, l, p, q)

def double_network(factored, bn, dist_bn, z, x, y):
    if factored:
        return ladder_factored_double_network(factored, bn, dist_bn, z, x, y)
    else:
        return ladder_double_network(factored, bn, dist_bn, z, x, y)

def ladder_double_network(factored, bn, dist_bn, z, x, y):
    assert not factored
    mlp = lambda shape: build_mlp(shape, bn=bn)
    dist = lambda shape, dist_type: build_distribution(shape, dist_type, bn=bn, dist_bn=dist_bn)
    shapex = {'x': (x,)}
    shapey = {'x': (y,)}
    shape = {'x': (x,), 'y': (y,)}
    u, l, p, q = {}, {}, {}, {}
    # upstream det
    u['xy'] = mlp((784, 500, 500))
    u['x'] = mlp((x, 500, 500))
    u['z2'] = mlp((y, 500, 500))
    u['z1'] = mlp((500, 500, 500))
    # generation
    p['z1'] = dist((), 'gaussian')
    p['z2'] = dist((z, 500, 500, z), 'gaussian')
    p['x'] = dist((z, 500, 500, x), 'bernoulli')
    p['y'] = dist((z, 500, 500, y), 'bernoulli')
    # likelihood
    l['z2|y'] = dist((500, z), 'gaussian')
    # inference
    q['z1|xy'] = dist((500, z), 'gaussian')
    q['z1|x'] = dist((500, z), 'gaussian')
    q['z1|y'] = dist((500, z), 'gaussian')
    q['z2'] = build_inference()
    # define nets
    x_nets = {'u_net': {'x': u['x']},
              'q_net': {'z': q['z1|x']},
              'p_net': {'z': p['z1'], 'x': p['x']}}
    y_nets = {'u_net': {'z2': u['z2'], 'z1': u['z1']},
              'l_net': {'z2': l['z2|y']},
              'q_net': {'z2': q['z2'], 'z1': q['z1|y']},
              'p_net': {'z1': p['z1'], 'z2': p['z2'], 'x': p['y']}}
    c_nets = {'u_net': u, 'l_net': l, 'p_net': p, 'q_net': q}
    vaex = VAE(validate_network=False, shape=shapex, **x_nets)
    vaey = LadderDoubleVAE(validate_network=False, shape=shapey, **y_nets)
    cvae = LadderDouble(validate_network=False, shape=shape, **c_nets)
    return vaex, vaey, cvae, (u, l, p, q)

def ladder_factored_double_network(factored, bn, dist_bn, z, x, y):
    assert factored
    mlp = lambda shape: build_mlp(shape, bn=bn)
    dist = lambda shape, dist_type: build_distribution(shape, dist_type, bn=bn, dist_bn=dist_bn)
    shapex = {'x': (x,)}
    shapey = {'x': (y,)}
    shape = {'x': (x,), 'y': (y,)}
    u, l, p, q = {}, {}, {}, {}
    # upstream det
    u['x'] = mlp((x, 500, 500))
    u['z2'] = mlp((y, 500, 500))
    u['z1'] = mlp((500, 500, 500))
    # generation
    p['z'] = dist((), 'gaussian')
    p['z1'] = dist((), 'gaussian')
    p['z2'] = dist((z, 500, 500, z), 'gaussian')
    p['x'] = dist((z, 500, 500, x), 'bernoulli')
    p['y'] = dist((z, 500, 500, y), 'bernoulli')
    # likelihood
    l['z1|y'] = dist((500, z), 'gaussian')
    l['z2|y'] = dist((500, z), 'gaussian')
    # inference
    q['z1|x'] = dist((500, z), 'gaussian')
    q['z1'] = build_inference()
    q['z2'] = build_inference()
    # define nets
    c_nets = {'u_net': u, 'l_net': l, 'p_net': p, 'q_net': q}
    vaex = None
    vaey = None
    cvae = LikelihoodLadderDouble(validate_network=False, shape=shape, **c_nets)
    return vaex, vaey, cvae, (u, l, p, q)
