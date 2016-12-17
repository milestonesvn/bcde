from models.utils import concat
from kaos.bayes import BayesNet
from kaos.distributions import log_bernoulli
from kaos.softplus import log_normal, kl_normal
from kaos.utils import rename
from keras.layers import Input, merge
from keras import backend as K
import numpy as np

class SingleBase(BayesNet):
    def _define_log_importance_likelihood(self, inputs, y_param, loss):
        x, y = inputs
        log_imp = -loss(y, y_param)
        self._log_importance_likelihood = K.function([x, y, K.learning_phase()], log_imp)

    def log_importance_likelihood(self, data_input, data_output, n_samples):
        x, y = data_input[:2]
        ln_imp = np.empty((len(y), 0))

        for i in xrange(n_samples):
            sample = self._log_importance_likelihood([x, y, 0]).reshape(-1, 1)
            ln_imp = np.hstack((ln_imp, sample))

        return ln_imp

    def _define_io_loss(self):
        x = self._define_io_loss_x()
        y = self._define_io_loss_y()
        xy = self._define_io_loss_xy()
        return self._zip_io_losses(xy, x, y)

class Single(SingleBase):
    def _define_io_loss_x(self):
        x = Input(shape=self.shape['x'])
        u_net, q_net, p_net = self.u_net, self.q_net, self.p_net
        u, q, p, s = {}, {}, {}, {}

        # upstream
        u['x'] = u_net['x'](x)
        # inference
        q['z'], s['z'] = q_net['z|x'](u['x'], sample=True)
        # generation
        p['z'] = p_net['z']()
        p['x'] = p_net['x'](s['z'])

        @rename('x loss')
        def loss(x, x_param):
            loss = -log_bernoulli(x, p['x'])
            loss += log_normal(s['z'], q['z']) - log_normal(s['z'], p['z'])
            return loss

        return self._standardize_io_loss(x, p['x'], loss)

    def _define_io_loss_y(self):
        y = Input(shape=self.shape['y'])
        u_net, q_net, p_net = self.u_net, self.q_net, self.p_net
        u, q, p, s = {}, {}, {}, {}

        # upstream
        u['y'] = u_net['y'](y)
        # inference
        q['z'], s['z'] = q_net['z|y'](u['y'], sample=True)
        # generation
        p['z'] = p_net['z']()
        p['y'] = p_net['y'](s['z'])

        @rename('y loss')
        def loss(y, y_param):
            loss = -log_bernoulli(y, p['y'])
            loss += log_normal(s['z'], q['z']) - log_normal(s['z'], p['z'])
            return loss

        return self._standardize_io_loss(y, p['y'], loss)

    def _define_io_loss_xy(self):
        x, y = Input(shape=self.shape['x']), Input(shape=self.shape['y'])
        u_net, q_net, p_net = self.u_net, self.q_net, self.p_net
        u, q, p, s = {}, {}, {}, {}

        # upstream
        u['x'] = u_net['x'](x)
        u['y'] = u_net['y'](y)
        u['xy'] = u_net['xy'](concat(x, y))
        # inference
        q['z'], s['z'] = q_net['z|xy'](u['xy'], sample=True)
        # generation
        p['z'] = p_net['z']()
        p['z|x'] = q_net['z|x'](u['x'])
        p['x'] = p_net['x'](s['z'])
        p['y'] = p_net['y'](s['z'])

        @rename('y|x loss')
        def lossc(y, y_param):
            loss = -log_bernoulli(y, p['y'])
            loss += log_normal(s['z'], q['z']) - log_normal(s['z'], p['z|x'])
            return loss
        if self._compute_log_likelihood:
            self._define_log_importance_likelihood([x, y], p['y'], lossc)

        @rename('xy loss')
        def lossxy(x, x_param):
            loss = -log_bernoulli(x, p['x'])
            loss -= log_bernoulli(y, p['y'])
            loss += log_normal(s['z'], q['z']) - log_normal(s['z'], p['z'])
            return loss

        return self._standardize_io_loss([x, y],
                                         [p['y'], p['x']],
                                         [lossc, lossxy])

class LikelihoodSingle(SingleBase):
    def _define_io_loss_x(self):
        x = Input(shape=self.shape['x'])
        u_net, l_net, q_net, p_net = self.u_net, self.l_net, self.q_net, self.p_net
        u, l, q, p, s = {}, {}, {}, {}, {}

        # upstream
        u['x'] = u_net['x'](x)
        # inference
        q['z'], s['z'] = q_net['z|x'](u['x'], sample=True)
        # generation
        p['z'] = p_net['z']()
        p['x'] = p_net['x'](s['z'])

        @rename('x loss')
        def loss(x, x_param):
            loss = -log_bernoulli(x, p['x'])
            loss += log_normal(s['z'], q['z']) - log_normal(s['z'], p['z'])
            return loss

        return self._standardize_io_loss(x, p['x'], loss)

    def _define_io_loss_y(self):
        y = Input(shape=self.shape['y'])
        u_net, l_net, q_net, p_net = self.u_net, self.l_net, self.q_net, self.p_net
        u, l, q, p, s = {}, {}, {}, {}, {}

        # upstream
        u['y'] = u_net['y'](y)
        # likelihood
        l['z'] = l_net['z|y'](u['y'])
        # generation and inference
        p['z'] = p_net['z']()
        q['z'], s['z'] = q_net['z|y'](l['z'], p['z'], sample=True)
        p['y'] = p_net['y'](s['z'])

        @rename('y loss')
        def loss(y, y_param):
            loss = -log_bernoulli(y, p['y'])
            loss += log_normal(s['z'], q['z']) - log_normal(s['z'], p['z'])
            return loss

        return self._standardize_io_loss(y, p['y'], loss)

    def _define_io_loss_xy(self):
        x, y = Input(shape=self.shape['x']), Input(shape=self.shape['y'])
        u_net, l_net, q_net, p_net = self.u_net, self.l_net, self.q_net, self.p_net
        u, l, q, p, s = {}, {}, {}, {}, {}

        # upstream
        u['x'] = u_net['x'](x)
        u['y'] = u_net['y'](y)
        # likelihoods
        l['z|y'] = l_net['z|y'](u['y'])
        # generation and inference
        p['z'] = p_net['z']()
        p['z|x'] = q_net['z|x'](u['x'])
        q['z'], s['z'] = q_net['z|xy'](l['z|y'], p['z|x'], sample=True)
        p['x'] = p_net['x'](s['z'])
        p['y'] = p_net['y'](s['z'])

        @rename('y|x loss')
        def lossc(y, y_param):
            loss = -log_bernoulli(y, p['y'])
            loss += log_normal(s['z'], q['z']) - log_normal(s['z'], p['z|x'])
            return loss
        if self._compute_log_likelihood:
            self._define_log_importance_likelihood([x, y], p['y'], lossc)

        @rename('xy loss')
        def lossxy(x, x_param):
            loss = -log_bernoulli(x, p['x'])
            loss -= log_bernoulli(y, p['y'])
            loss += log_normal(s['z'], q['z']) - log_normal(s['z'], p['z'])
            return loss

        return self._standardize_io_loss([x, y],
                                         [p['y'], p['x']],
                                         [lossc, lossxy])
