"""Separate script, to avoid slow import."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import matplotlib.pyplot as plt
from datetime import datetime


class AutoEncoder(nn.Module):
    def __init__(self, L, P):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(L, L//2, bias=True),
            nn.Tanh(),
            nn.Linear(L//2, L//4, bias=True),
            nn.Tanh(),
            nn.Linear(L//4, P, bias=True),
            nn.Softmax(dim=1),
        )
        self.decoder = nn.Linear(P, L, bias=False)

    def forward(self, y):
        code = self.encoder(y)
        output = self.decoder(code)
        return code, output


class TimeReminder:
    def __init__(self):
        self.start = datetime.now()

    def remind(self, now_cnt, max_cnt):
        return (datetime.now()-self.start) / (now_cnt+1) * (max_cnt - now_cnt + 1e-5)

    def consuming(self):
        return datetime.now() - self.start


def smooth_matrix(height, width):
    """Return smooth matrix h (numpy array)"""
    N = height * width
    h = np.eye(N)
    for i in range(N):
        if i - width >= 0 and i + width <= N - 1 and i % width != 0 and (i + 1) % width != 0:
            h[i + 1, i] = -0.25
            h[i - 1, i] = -0.25
            h[i + width, i] = -0.25
            h[i - width, i] = -0.25
        elif i - width < 0:
            if i == 0 or i == width - 1:
                h[1, 0] = -0.5
                h[width, 0] = -0.5
                h[width - 2, width - 1] = -0.5
                h[2 * width - 1, width - 1] = -0.5
            else:
                h[i - 1, i] = -1 / 3
                h[i + 1, i] = -1 / 3
                h[i + width, i] = -1 / 3
        elif i + width > N - 1:
            if i == N - 1 or i == N - width:
                h[N - 2, N - 1] = -0.5
                h[N - width, N - 1] = -0.5
                h[N - width + 1, N - width] = -0.5
                h[N - 2 * width, N - width] = -0.5
            else:
                h[i - 1, i] = -1 / 3
                h[i + 1, i] = -1 / 3
                h[i - width, i] = -1 / 3
        elif i % width == 0:
            h[i - width, i] = -1 / 3
            h[i + width, i] = -1 / 3
            h[i + 1, i] = -1 / 3
        elif (i + 1) % width == 0:
            h[i - width, i] = -1 / 3
            h[i + width, i] = -1 / 3
            h[i - 1, i] = -1 / 3
    return h


def norm2squ(x):
    """Return L2norm^2"""
    return torch.sum(torch.pow(x, exponent=2))


class AutoUnmix:
    """Unmix by autoencoder"""
    def __init__(self, L, P, init_edm=None, height=None, width=None, version='com', seed=30):
        """
        :param L: bands number
        :param P: endmember number
        :param init_edm: initialized endmembers for decoder, shape(L, P)
        """
        if seed:
            torch.manual_seed(seed)
        self.reg_layer = None
        self.reg_norm = None
        self.frozen_layer = None
        self.lr_decay = 1.
        self.reg_decay = 1e-5
        self.unfreeze_time = 0.9
        self.input_shape = (L,)
        self.loss_code = self._loss_code(mode='0', decay=1e-5)
        if version == 'com':
            self.model = AutoEncoder(L, P).cuda()
            self.lr_decay = 0.9  
            self.reg_norm: list = [2, 2, 2]  
            self.reg_layer: list = ['encoder.0.weight', 'encoder.2.weight', 'encoder.4.weight'] 
            self.edm_layer = 'decoder.weight'
            self._load_endmember(edm=init_edm, layer_name=self.edm_layer)
            self.frozen_layer = [self.edm_layer]
            self.loss_func = self._loss_function(mode='sad')
            self.loss_code = self._loss_code(mode='l1-2', decay=1e-5)
            h = torch.from_numpy(smooth_matrix(height=height, width=width)).float()
            self.s0 = torch.abs(h) > 0.1
            self.s1 = h < -0.1
            self.s2 = torch.eye(height*width)
            self.h = h.cuda()
            self.h.requires_grad = True
            self.s0 = self.s0.cuda()
            self.s1 = self.s1.cuda()
            self.s2 = self.s2.cuda()
            self.unfreeze_time = 0.9
            self.opt = torch.optim.Adam([{'params': self.model.encoder.parameters(), 'lr': 1e-3},
                                         {'params': self.model.decoder.parameters(), 'lr': 1e-2},
                                         {'params': self.h, 'lr': 1e-1}])
        else:
            raise Exception('Version Error. com/cnn')

        self.L, self.P = L, P
        self.new_edm = 0
        self.height = height
        self.width = width
        self.version = version
        self.edm = init_edm
        self.timer = TimeReminder()
        self._list_layer()
        summary(self.model, self.input_shape)

    def _load_endmember(self, edm, layer_name):
        edm = torch.from_numpy(edm)
        model_dict = self.model.state_dict()
        model_dict[layer_name] = edm
        self.model.load_state_dict(model_dict)

    def _list_layer(self):
        for name, para in self.model.named_parameters():
            print(name, end='/')
        print()

    def _freeze(self):
        if self.frozen_layer is None:
            return
        for name, value in self.model.named_parameters():
            if name in self.frozen_layer:
                value.requires_grad = False

    def _unfreeze(self):
        if self.frozen_layer is None:
            return
        for name, para in self.model.named_parameters():
            if name in self.frozen_layer:
                para.requires_grad = True

    def _regularization_loss(self, layer_name=None, weight_decay=1e-5, p=None):
        if layer_name is None:
            return 0
        r_loss = 0
        for name, param in self.model.named_parameters():
            if name in layer_name:
                norm_num = p[layer_name.index(name)]
                if norm_num == 1 or norm_num == 2:
                    r_loss += weight_decay * torch.norm(param, p=norm_num)
        return r_loss

    def _special_loss(self, code, output):
        if self.version == 'com':
            return 1e-6 * norm2squ(torch.mm(self.h*self.s0, code)) +\
                   5 * (norm2squ(torch.sum(self.h*self.s1, dim=0)+torch.tensor(1).cuda()) +
                        norm2squ(self.h*self.s2-self.s2))
        else:
            return 0

    @staticmethod
    def _loss_code(decay=1e-3, mode='l1'):
        if mode == 'l1-2':
            return lambda code: decay * torch.mean(torch.sum(torch.sqrt(torch.abs(code)), dim=-1))
        if mode == '0':
            return lambda code: 0
        else:
            raise Exception('Mode Error! l1-2/0')

    @staticmethod
    def _loss_function(mode='sad'):
        if mode == 'sad':
            return lambda output, target: \
                torch.mean(torch.acos(torch.sum(output * target, dim=-1) /
                           (torch.norm(output, dim=-1, p=2)*torch.norm(target, dim=-1, p=2))))
        else:
            raise Exception('Version Error! sad/')

    def fit(self, y, max_iter=500, verbose=True):
        """
        :param y: shape should be (L, N)
        :param max_iter: max epochs
        :param verbose: verbose
        :return: abundance with shape (P, N)
        """
        pix = torch.from_numpy(y.T).float().cuda()
        self._freeze()
        loss_record = 1e5
        epoch = 0
        while epoch < max_iter:
            code, output = self.model(pix)
            base_loss = self.loss_func(output, pix)
            c_loss = self.loss_code(code)
            r_loss = self._regularization_loss(layer_name=self.reg_layer, p=self.reg_norm, weight_decay=self.reg_decay)
            s_loss = self._special_loss(code, output)
            loss = base_loss + r_loss + c_loss + s_loss

            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10, norm_type=1)
            self.opt.step()
            epoch += 1

            if (epoch+1) % 200 == 0:
                for i, dic in enumerate(self.opt.param_groups):
                    dic['lr'] *= self.lr_decay

            if (epoch+1) % (max_iter//10) == 0:
                print('epoch [{}/{}], loss:{:.8f}, '.format(epoch + 1, max_iter, loss), end=' ')
                print('ETA:', self.timer.remind(now_cnt=epoch + 1, max_cnt=max_iter))

                if abs(loss_record-float(loss)) <= 1e-6 and epoch > max_iter*0.5:
                    if epoch < max_iter*self.unfreeze_time:
                        epoch = int(max_iter*self.unfreeze_time) + 1 
                    else:
                        break
                else:
                    loss_record = float(loss)

                if epoch >= max_iter * self.unfreeze_time:
                    self._unfreeze()

        if verbose and self.height and self.width:
            est_y = output.cpu()
            est_y = est_y.data.numpy()
            est_y = np.reshape(est_y, (self.height, self.width, -1))
            plt.subplot(121)
            plt.title('Reconstructed')
            plt.imshow(est_y[:, :, [90, 60, 30]])
            plt.subplot(122)
            plt.title('Real')
            y_plot = np.reshape(y.T, (self.height, self.width, -1))
            plt.imshow(y_plot[:, :, [90, 60, 30]])
            plt.show()

        m = None
        for name, para in self.model.named_parameters():
            if name == self.edm_layer:
                m = para.cpu().data.numpy()


        a = code.cpu()
        if len(a.shape) == 2:
            a = a.data.numpy().T 
        else:
            a = a.data.numpy()
            a = np.squeeze(a)
            a = a.reshape(a.shape[0], -1)
        a = a / np.sum(a, axis=0)

        if m is not None:
            self.new_edm = m[:self.L]
        else:
            self.new_edm = self.edm[:self.L]
        return a


if __name__ == '__main__':
    im = np.load('jasperRidge_R198.npy')
    edm = np.load('edm_extracted.npy')
    model = AutoUnmix(L=198, P=4, init_edm=edm.T, height=100, width=100)
    abd = model.fit(im.reshape(-1, 198).T, max_iter=1000, verbose=True).T

    for i in range(4):
        plt.subplot(1, 4, i+1)
        plt.imshow(np.reshape(abd[:,i], (100,100)))
    plt.show()


