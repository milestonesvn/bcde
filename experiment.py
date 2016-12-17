from main import main
import sys

# Task settings:
# nlabel: no. labeled data points to use. If fully-supervised, set nlabel = 60000
# small_val: indicates whether to use small validation set or not. Set true for fully-sup
# task: q1, q2, q3, or td
# shift: 0, 5, 5+, 5--, or 5-. Only use with td.
nlabel = 5000
small_val = True
task = 'q2'
shift = '0'

# Model selection
# model_spec: double or single. Indicates architecture choice
# factored: indicates whether to use factored inference or not
# bn: indicates whether to use batchnorm
# dist_bn: indicates whether to use batchnorm on last layer of distribution
# z: no. of latent vars
# run_x: whether to pretrain x
# run_y: whether to pretrain y
# run_joint: whether to pretrain on joint (setting to True implies Hybrid)
model_spec = 'single'
factored = True
run_joint = True

# folder settings:
# folder: location of log output
# scratch: location of model output
# ignore: suppress parameters from file name
folder = 'hybrid'
scratch = '/scratch/users/rshu15/Documents/github/cde2/07_double'
ignore = {'bn', 'dist_bn', 'lc', 'lxy', 'lx', 'ly', 'lr', 'joint_lr', 'run_every', 'scratch'}

for _ in xrange(5):
    main(nlabel=nlabel, task=task, shift=shift,
         model_spec=model_spec, factored=factored, bn=True, dist_bn=True, z=50,
         run_x=False, run_y=False, run_joint=run_joint,
         small_val=small_val, folder=folder, scratch=scratch, ignore=ignore)
