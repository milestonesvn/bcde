from kaos.data import MnistLoader
import numpy as np

class Qnist(MnistLoader):
    def __init__(self, nlabel, seed, quad_type, batchsize=100, resample=False, small_val=False, shift='0'):
        super(Qnist, self).__init__(batchsize)
        self.resample = resample
        self.ssl_compatible = True
        self.small_val = small_val
        self.quad_type = quad_type
        self.x_orig = self.x_train
        self.y_orig = self.y_train
        self.binarize()
        self.convert_to_ssl(nlabel, seed)
        if shift == '5':
            assert quad_type == 'td'
            self.shift()
            self.split()
        elif shift == '5+':
            assert quad_type == 'td'
            self.x_train = self._binarize(np.tile(self.x_orig, (5, 1)), seed=42)
            self.y_train = np.tile(self.y_train, 5)
            print "Expanding training set by 5 times. New size: ", len(self.x_train)
            self.shift()
            self.split()
        elif shift == '5--':
            assert quad_type == 'td'
            print "No expansion allowed. Splitting first, then shift."
            self.split()
            self.shift()
        elif shift == '5-':
            assert quad_type == 'td'
            self.x_train = self._binarize(np.tile(self.x_orig, (5, 1)), seed=42)
            self.y_train = np.tile(self.y_train, 5)
            print "Expanding training set by 5 times. New size: ", len(self.x_train)
            print "Splitting first, then shift."
            self.split()
            self.shift()
        elif shift == '0':
            self.split()
        else:
            raise Exception('Unknown shift parameter')

    def shift(self):
        state = np.random.get_state()
        np.random.seed(42)
        def transform(x):
            H = len(x)/28
            y = np.zeros((H, 28))
            s = np.random.randint(0, 5)
            if s == 0:
                return x, 0
            if np.random.choice([0, 1]):
                y[:, :-s] = x.reshape(H, 28)[:, s:]
                return y.reshape(-1), -s
            else:
                y[:, s:] = x.reshape(H, 28)[:, :-s]
                return y.reshape(-1), s
        def batch_transform(x, s):
            for i in xrange(len(x)):
                x[i], s[i] = transform(x[i])
        self.s_label = np.zeros(len(self.y_label))
        self.s_train = np.zeros(len(self.y_train))
        self.s_valid = np.zeros(len(self.y_valid))
        self.s_test = np.zeros(len(self.y_test))
        batch_transform(self.x_label, self.s_label)
        batch_transform(self.x_train, self.s_train)
        batch_transform(self.x_valid, self.s_valid)
        batch_transform(self.x_test, self.s_test)
        np.random.set_state(state)

    def convert_to_ssl(self, nlabel, seed):
        assert self.ssl_compatible, "Compatability flag is off"
        state = np.random.get_state()
        np.random.seed(seed)
        x, y = self.balanced_sampler(self.x_train, self.y_train, nlabel)
        self.x_label = x
        self.y_label = y
        if self.small_val:
            print "Using small validation set of size", nlabel/5
            x, y = self.balanced_sampler(self.x_valid, self.y_valid, nlabel/5)
            self.x_valid = x
            self.y_valid = y
        np.random.set_state(state)
        return self

    def split(self):
        self.z_train, self.z_valid, self.z_test, self.z_orig = self.y_train, self.y_valid, self.y_test, self.y_orig
        self.x_train, self.y_train = self._split(self.x_train)
        self.x_label, self.y_label = self._split(self.x_label)
        self.x_valid, self.y_valid = self._split(self.x_valid)
        self.x_test, self.y_test = self._split(self.x_test)
        self.x_orig, self.y_orig = self._split(self.x_orig)

    def _split(self, data):
        assert len(data.shape) == 2
        assert data.shape[1] == 784
        size = data.shape[0]
        spatial_idxs = np.arange(784).reshape(28, 28)
        if self.quad_type == 'q1':
            x_idx = spatial_idxs[14:, :14].reshape(-1)
            y_idx = np.delete(spatial_idxs.reshape(-1), x_idx)
        elif self.quad_type == 'q2':
            x_idx = spatial_idxs[:, :14].reshape(-1)
            y_idx = np.delete(spatial_idxs.reshape(-1), x_idx)
        elif self.quad_type == 'q3':
            y_idx = spatial_idxs[14:, 14:].reshape(-1)
            x_idx = np.delete(spatial_idxs.reshape(-1), y_idx)
        elif self.quad_type == 'td':
            x_idx = spatial_idxs[:14, :].reshape(-1)
            y_idx = np.delete(spatial_idxs.reshape(-1), x_idx)
        else:
            raise Exception('quadrant not specified')
        data_x = data[:, x_idx]
        data_y = data[:, y_idx]
        # data_x = data_x.reshape(size, -1)
        # data_y = data_y.reshape(size, -1)
        return data_x, data_y

    def stitch(self, xs, ys):
        assert len(xs.shape) == 2
        assert len(ys.shape) == 2
        assert len(xs) == len(ys)
        imgs = np.empty((len(xs), 784))
        spatial_idxs = np.arange(784).reshape(28,28)
        if self.quad_type == 'q1':
            x_idx = spatial_idxs[14:, :14].reshape(-1)
            y_idx = np.delete(spatial_idxs.reshape(-1), x_idx)
        elif self.quad_type == 'q2':
            x_idx = spatial_idxs[:, :14].reshape(-1)
            y_idx = np.delete(spatial_idxs.reshape(-1), x_idx)
        elif self.quad_type == 'q3':
            y_idx = spatial_idxs[14:, 14:].reshape(-1)
            x_idx = np.delete(spatial_idxs.reshape(-1), y_idx)
        elif self.quad_type == 'td':
            x_idx = spatial_idxs[:14, :].reshape(-1)
            y_idx = np.delete(spatial_idxs.reshape(-1), x_idx)
        else:
            raise Exception('quadrant not specified')
        imgs[:, x_idx] = xs
        imgs[:, y_idx] = ys
        return imgs

    def get_training_batch(self):
        if self.resample:
            idx = np.random.choice(len(self.x_orig), self.batchsize, replace=False)
            x, y = self.x_orig[idx], self.y_orig[idx]
            x = self._binarize(x)
            y = self._binarize(y)
            return x, y
        else:
            return super(Qnist, self).get_training_batch()

class QnistVAEX(Qnist):
    def __init__(self, quad_type, batchsize=100, nlabel=60000, seed=42, resample=False, shift=False):
        super(QnistVAEX, self).__init__(nlabel, seed, quad_type, batchsize, resample, shift=shift)

    def get_training_batch(self):
        x, y = super(QnistVAEX, self).get_training_batch()
        return x, x

    def get_training_set(self):
        x, y = super(QnistVAEX, self).get_training_set()
        return x, x

    def get_validation_set(self):
        x, y = super(QnistVAEX, self).get_validation_set()
        return x, x

    def get_test_set(self):
        x, y = super(QnistVAEX, self).get_test_set()
        return x, x

class QnistVAEY(Qnist):
    def __init__(self, quad_type, batchsize=100, nlabel=60000, seed=42, resample=False, shift=False):
        super(QnistVAEY, self).__init__(nlabel, seed, quad_type, batchsize, resample, shift=shift)

    def get_training_batch(self):
        x, y = super(QnistVAEY, self).get_training_batch()
        return y, y

    def get_training_set(self):
        x, y = super(QnistVAEY, self).get_training_set()
        return y, y

    def get_validation_set(self):
        x, y = super(QnistVAEY, self).get_validation_set()
        return y, y

    def get_test_set(self):
        x, y = super(QnistVAEY, self).get_test_set()
        return y, y

class QnistLink(Qnist):
    def get_training_batch(self):
        xl, yl = super(QnistLink, self).get_labeled_training_batch()
        xu, yu = super(QnistLink, self).get_training_batch()
        return [xl, yl], yl

    def get_training_set(self):
        x, y = super(QnistLink, self).get_labeled_training_set()
        return [x, y], y

    def get_validation_set(self):
        x, y = super(QnistLink, self).get_validation_set()
        return [x, y], y

    def get_test_set(self):
        x, y = super(QnistLink, self).get_test_set()
        return [x, y], y

class QnistUnified(Qnist):
    def get_training_batch(self):
        xl, yl = super(QnistUnified, self).get_labeled_training_batch()
        xu, yu = super(QnistUnified, self).get_training_batch()
        return [xl, yl, xu, yu], [yl, xl, xu, yu]

    def get_training_set(self):
        x, y = super(QnistUnified, self).get_labeled_training_set()
        return [x, y, x, y], [y, x, x, y]

    def get_validation_set(self):
        x, y = super(QnistUnified, self).get_validation_set()
        return [x, y, x, y], [y, x, x, y]

    def get_test_set(self):
        x, y = super(QnistUnified, self).get_test_set()
        return [x, y, x, y], [y, x, x, y]
