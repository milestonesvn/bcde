from models.utils import concat
from kaos.bayes import BayesNet
from kaos.distributions import log_bernoulli, log_normal, kl_normal
from kaos.utils import rename
from keras.layers import Input, merge
from keras import backend as K
import numpy as np

class DoubleBase(BayesNet):
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


class LadderDouble(DoubleBase):
    def _define_io_loss_x(self):
        x = Input(shape=self.shape['x'])
        u_net, q_net, p_net = self.u_net, self.q_net, self.p_net
        u, q, p, s = {}, {}, {}, {}

        # upstream
        u['x'] = u_net['x'](x)
        # inference
        q['z1'], s['z1'] = q_net['z1|x'](u['x'], sample=True)
        # generation
        p['z1'] = p_net['z1']()
        p['x'] = p_net['x'](s['z1'])

        @rename('x loss')
        def loss(x, x_param):
            loss = -log_bernoulli(x, p['x'])
            loss += log_normal(s['z1'], q['z1']) - log_normal(s['z1'], p['z1'])
            return loss

        return self._standardize_io_loss(x, p['x'], loss)

    def _define_io_loss_y(self):
        y = Input(shape=self.shape['y'])
        u_net, l_net, q_net, p_net = self.u_net, self.l_net, self.q_net, self.p_net
        u, l, q, p, s = {}, {}, {}, {}, {}

        # upstream
        u['z2'] = u_net['z2'](y)
        u['z1'] = u_net['z1'](u['z2'])
        # likelihood
        l['z2'] = l_net['z2|y'](u['z2'])
        # generation and inference
        p['z1'] = p_net['z1']()
        q['z1'], s['z1'] = q_net['z1|y'](u['z1'], sample=True)
        p['z2'] = p_net['z2'](s['z1'])
        q['z2'], s['z2'] = q_net['z2'](l['z2'], p['z2'], sample=True)
        p['y'] = p_net['y'](s['z2'])

        @rename('y loss')
        def loss(y, y_param):
            loss = -log_bernoulli(y, p['y'])
            loss += log_normal(s['z1'], q['z1']) - log_normal(s['z1'], p['z1'])
            loss += log_normal(s['z2'], q['z2']) - log_normal(s['z2'], p['z2'])
            return loss

        return self._standardize_io_loss(y, p['y'], loss)

    def _define_io_loss_xy(self):
        x, y = Input(shape=self.shape['x']), Input(shape=self.shape['y'])
        u_net, l_net, q_net, p_net = self.u_net, self.l_net, self.q_net, self.p_net
        u, l, q, p, s = {}, {}, {}, {}, {}

        # upstream
        u['x'] = u_net['x'](x)
        u['z2'] = u_net['z2'](y)
        u['xy'] = u_net['xy'](concat(x, y))
        # likelihood
        l['z2'] = l_net['z2|y'](u['z2'])
        # generation and inference
        p['z1'] = p_net['z1']()
        p['z1|x'] = q_net['z1|x'](u['x'])
        q['z1'], s['z1'] = q_net['z1|xy'](u['xy'], sample=True)
        p['z2'] = p_net['z2'](s['z1'])
        q['z2'], s['z2'] = q_net['z2'](l['z2'], p['z2'], sample=True)
        p['x'] = p_net['x'](s['z1'])
        p['y'] = p_net['y'](s['z2'])

        @rename('y|x loss')
        def lossc(y, y_param):
            loss = -log_bernoulli(y, p['y'])
            loss += log_normal(s['z1'], q['z1']) - log_normal(s['z1'], p['z1|x'])
            loss += log_normal(s['z2'], q['z2']) - log_normal(s['z2'], p['z2'])
            return loss
        if self._compute_log_likelihood:
            self._define_log_importance_likelihood([x, y], p['y'], lossc)

        @rename('xy loss')
        def lossxy(x, x_param):
            loss = -log_bernoulli(x, p['x'])
            loss -= log_bernoulli(y, p['y'])
            loss += log_normal(s['z1'], q['z1']) - log_normal(s['z1'], p['z1'])
            loss += log_normal(s['z2'], q['z2']) - log_normal(s['z2'], p['z2'])
            return loss

        return self._standardize_io_loss([x, y],
                                         [p['y'], p['x']],
                                         [lossc, lossxy])

class LikelihoodLadderDouble(DoubleBase):
    def _define_io_loss_x(self):
        x = Input(shape=self.shape['x'])
        u_net, q_net, p_net = self.u_net, self.q_net, self.p_net
        u, q, p, s = {}, {}, {}, {}

        # upstream
        u['x'] = u_net['x'](x)
        # inference
        q['z1'], s['z1'] = q_net['z1|x'](u['x'], sample=True)
        # generation
        p['z1'] = p_net['z1']()
        p['x'] = p_net['x'](s['z1'])

        @rename('x loss')
        def loss(x, x_param):
            loss = -log_bernoulli(x, p['x'])
            loss += log_normal(s['z1'], q['z1']) - log_normal(s['z1'], p['z1'])
            return loss

        return self._standardize_io_loss(x, p['x'], loss)

    def _define_io_loss_y(self):
        y = Input(shape=self.shape['y'])
        u_net, l_net, q_net, p_net = self.u_net, self.l_net, self.q_net, self.p_net
        u, l, q, p, s = {}, {}, {}, {}, {}

        # upstream
        u['z2'] = u_net['z2'](y)
        u['z1'] = u_net['z1'](u['z2'])
        # likelihood
        l['z2'] = l_net['z2|y'](u['z2'])
        l['z1'] = l_net['z1|y'](u['z1'])
        # generation and inference
        p['z1'] = p_net['z1']()
        q['z1'], s['z1'] = q_net['z1'](l['z1'], p['z1'], sample=True)
        p['z2'] = p_net['z2'](s['z1'])
        q['z2'], s['z2'] = q_net['z2'](l['z2'], p['z2'], sample=True)
        p['y'] = p_net['y'](s['z2'])

        @rename('y loss')
        def loss(y, y_param):
            loss = -log_bernoulli(y, p['y'])
            loss += log_normal(s['z1'], q['z1']) - log_normal(s['z1'], p['z1'])
            loss += log_normal(s['z2'], q['z2']) - log_normal(s['z2'], p['z2'])
            return loss

        return self._standardize_io_loss(y, p['y'], loss)

    def _define_io_loss_xy(self):
        x, y = Input(shape=self.shape['x']), Input(shape=self.shape['y'])
        u_net, l_net, q_net, p_net = self.u_net, self.l_net, self.q_net, self.p_net
        u, l, q, p, s = {}, {}, {}, {}, {}

        # upstream
        u['x'] = u_net['x'](x)
        u['z2'] = u_net['z2'](y)
        u['z1'] = u_net['z1'](u['z2'])
        # likelihood
        l['z2'] = l_net['z2|y'](u['z2'])
        l['z1'] = l_net['z1|y'](u['z1'])
        # generation and inference
        p['z1'] = p_net['z1']()
        p['z1|x'] = q_net['z1|x'](u['x'])
        q['z1'], s['z1'] = q_net['z1'](l['z1'], p['z1|x'], sample=True)
        p['z2'] = p_net['z2'](s['z1'])
        q['z2'], s['z2'] = q_net['z2'](l['z2'], p['z2'], sample=True)
        p['x'] = p_net['x'](s['z1'])
        p['y'] = p_net['y'](s['z2'])

        @rename('y|x loss')
        def lossc(y, y_param):
            loss = -log_bernoulli(y, p['y'])
            loss += log_normal(s['z1'], q['z1']) - log_normal(s['z1'], p['z1|x'])
            loss += log_normal(s['z2'], q['z2']) - log_normal(s['z2'], p['z2'])
            return loss
        if self._compute_log_likelihood:
            self._define_log_importance_likelihood([x, y], p['y'], lossc)

        @rename('xy loss')
        def lossxy(x, x_param):
            loss = -log_bernoulli(x, p['x'])
            loss -= log_bernoulli(y, p['y'])
            loss += log_normal(s['z1'], q['z1']) - log_normal(s['z1'], p['z1'])
            loss += log_normal(s['z2'], q['z2']) - log_normal(s['z2'], p['z2'])
            return loss

        return self._standardize_io_loss([x, y],
                                         [p['y'], p['x']],
                                         [lossc, lossxy])
