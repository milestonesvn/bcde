from models.utils import build_mlp, build_inference, build_distribution, create_name
from data_loaders.unified_quad import QnistVAEX as XL
from data_loaders.unified_quad import QnistVAEY as YL
from data_loaders.unified_quad import QnistUnified as CL
from kaos.utils import file_handle, Session
from kaos.callbacks import NegativeLogLikelihood, LossLog
from keras.optimizers import Adam, SGD
from networks import *
import numpy as np
import sys
import os

def use_network(model_spec, factored, bn, dist_bn, z, x, y):
    args = (factored, bn, dist_bn, z, x, y)
    if model_spec == 'single':
        return single_network(*args)
    elif model_spec == 'double':
        return double_network(*args)
    else:
        raise Exception("No model specified")

def run_model(model, dataloader, fh, scratch, nlabel=50000, run_every=1):
    save_model_path = os.path.join(scratch, fh.name.replace('.out', '.mod'))

    with Session(fh, save_model_path) as sess:
        losslog = LossLog(fh=fh)
        nll = NegativeLogLikelihood(dataloader,
                                    n_samples=50,
                                    run_every=run_every,
                                    run_training=False,
                                    run_validation=True,
                                    run_test=True,
                                    display_best_val=True,
                                    display_nelbo=True,
                                    display_epoch=True,
                                    patience=10,
                                    step_decay=0.1,
                                    save_model_path=save_model_path,
                                    fh=fh,
                                    end_line=True)
        callback_ops = [losslog, nll]
        callbacks = [op for op in callback_ops if op is not None]
        print "Begin running model..."
        if 'QnistUnified' in str(dataloader.__class__):
            iter_per_epoch = len(dataloader.x_label)/100
        elif 'QnistVAE' in str(dataloader.__class__):
            iter_per_epoch = len(dataloader.x_train)/100
        else:
            raise Exception('Undefined dataloader class')
        if '/joint/' in fh.name:
            iter_per_epoch = len(dataloader.x_train)/100
        print 'Iter per epoch is:', iter_per_epoch
        model.fit(dataloader,
                  nb_epoch=10000,
                  iter_per_epoch=iter_per_epoch,
                  callbacks=callbacks)

def get_optimizer(opt):
    if opt == 'adam':
        Opt = lambda lr: Adam(lr=lr)
    elif opt == 'sgdm':
        Opt = lambda lr: SGD(lr=lr, momentum=0.9, nesterov=False)
    elif opt == 'sgdn':
        Opt = lambda lr: SGD(lr=lr, momentum=0.9, nesterov=True)
    else:
        raise Exception('opt not specified')
    return Opt

def main(nlabel=60000, task=None, shift='0', model_spec='single', factored=False, bn=True,
         dist_bn=True, z=50, lc=1.0, lxy=0.0, lx=0.0, ly=0.0, lr=1e-3,
         run_x=False, run_y=False, run_joint=False, joint_lr=1e-3,
         small_val=False, run_every=1, folder=None, scratch=None, ignore=set()):
    version = create_name(locals(), main.func_code.co_varnames)
    if task in {'td', 'q2'}:
        x, y = 392, 392
    elif task == 'q1':
        x, y = 196, 588
    elif task == 'q3':
        x, y = 588, 196
    else:
        raise Exception('Task not implemented')
    vaex, vaey, cvae, dicts = use_network(model_spec, factored, bn, dist_bn, z, x, y)

    if model_spec == 'single' or 'double' in model_spec:
        L = CL
        loss_weights = [lc, lxy, lx, ly]
    elif model_spec == 'link':
        L = LinkL
        assert lxy == 0.0
        assert lx == 0.0
        assert ly == 0.0
        loss_weights = [lc]
    else:
        raise Exception('model not specified')

    Opt = get_optimizer('adam')
    cvae_fh, cvae_seed = file_handle(folder, 'cvae', version, get_seed=True)

    if cvae_seed > 1003:
        print "Program complete! Removing file"
        os.remove(cvae_fh.name)
        print "Quitting"
        quit()
    dataloader = L(nlabel, cvae_seed, task, shift=shift, small_val=small_val)

    if run_x:
        fh = file_handle(folder, 'vaex', version, seed=cvae_seed, overwrite=True)
        vaex.compile('adam', compute_log_likelihood=True, verbose=1)
        run_model(vaex, XL(task, shift=shift), fh, scratch)
    if run_y:
        fh = file_handle(folder, 'vaey', version, seed=cvae_seed, overwrite=True)
        vaey.compile('adam', compute_log_likelihood=True, verbose=1)
        run_model(vaey, YL(task, shift=shift), fh, scratch)
    if run_joint:
        w = float(len(dataloader.x_label))/len(dataloader.x_train)
        print '\nJoint training weight:', w
        JointOpt = get_optimizer('adam')
        fh = file_handle(folder, 'joint', version, seed=cvae_seed, overwrite=True)
        cvae.compile(JointOpt(lr=joint_lr), compute_log_likelihood=True,
                     loss_weights=[0.0, w, 1.0, 1.0], verbose=1)
        run_model(cvae, dataloader, fh, scratch, nlabel, run_every)

    # Run CVAE
    cvae.compile(Opt(lr=lr), compute_log_likelihood=True,
                 loss_weights=loss_weights, verbose=1)
    run_model(cvae, dataloader, cvae_fh, scratch, nlabel, run_every)
