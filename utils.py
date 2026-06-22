import visdom
import numpy as np
import torch
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from torch import nn


class Visualizer(object):
    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)
        self.index = {}

    def plot_lines(self, name, y, **kwargs):
        '''
        self.plot('loss', 1.00)
        '''
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]), X=np.array([x]),
                      win=str(name),
                      opts=dict(title=name),
                      update=None if x == 0 else 'append',
                      **kwargs
                      )
        self.index[name] = x + 1
    def disp_image(self, name, img):
        self.vis.image(img=img, win=name, opts=dict(title=name))
    def lines(self, name, line, X=None):
        if X is None:
            self.vis.line(Y=line, win=name)
        else:
            self.vis.line(X=X, Y=line, win=name)
    def scatter(self, name, data):
        self.vis.scatter(X=data, win=name)


def align_dvs_to_rgb(features_dvs, features_rgb):
    t_dvs, _, f_dvs = features_dvs.shape
    t_rgb, _, f_rgb = features_rgb.shape

    if t_dvs < t_rgb:
        # 插值扩展 DVS
        x_old = np.linspace(0, 1, t_dvs)
        x_new = np.linspace(0, 1, t_rgb)
        features_dvs_interp = np.zeros((t_rgb, 10, f_dvs), dtype=np.float32)

        for i in range(10):
            for j in range(f_dvs):
                interp_func = interp1d(x_old, features_dvs[:, i, j], kind='linear', fill_value='extrapolate')
                features_dvs_interp[:, i, j] = interp_func(x_new)

    elif t_dvs > t_rgb:
        # 软对齐 DVS - 通过高斯平滑 + 下采样
        sigma = (t_dvs - t_rgb) / 2.0  # 根据差异动态调整平滑程度
        features_dvs_smooth = gaussian_filter1d(features_dvs, sigma=sigma, axis=0)

        x_old = np.linspace(0, 1, t_dvs)
        x_new = np.linspace(0, 1, t_rgb)
        features_dvs_interp = np.zeros((t_rgb, 10, f_dvs), dtype=np.float32)

        for i in range(10):
            for j in range(f_dvs):
                interp_func = interp1d(x_old, features_dvs_smooth[:, i, j], kind='linear', fill_value='extrapolate')
                features_dvs_interp[:, i, j] = interp_func(x_new)

    else:
        features_dvs_interp = features_dvs  # T 维度已经对齐

    return features_dvs_interp


def process_feat(feat, length):
    new_feat = np.zeros((length, feat.shape[1])).astype(np.float32)
    
    # r = np.linspace(0, len(feat), length+1, dtype=np.int)   old
    r = np.linspace(0, len(feat), length + 1, dtype=int)
    for i in range(length):
        if r[i]!=r[i+1]:
            new_feat[i,:] = np.mean(feat[r[i]:r[i+1],:], 0)
        else:
            new_feat[i,:] = feat[r[i],:]
    return new_feat


def minmax_norm(act_map, min_val=None, max_val=None):
    if min_val is None or max_val is None:
        relu = torch.nn.ReLU()
        max_val = relu(torch.max(act_map, dim=0)[0])
        min_val = relu(torch.min(act_map, dim=0)[0])

    delta = max_val - min_val
    delta[delta <= 0] = 1
    ret = (act_map - min_val) / delta

    ret[ret > 1] = 1
    ret[ret < 0] = 0

    return ret


def modelsize(model, input, type_size=4):
    # check GPU utilisation
    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    print('Model {} : params: {:4f}M'.format(model._get_name(), para * type_size / 1000 / 1000))

    input_ = input.clone()
    input_.requires_grad_(requires_grad=False)

    mods = list(model.modules())
    out_sizes = []

    for i in range(1, len(mods)):
        m = mods[i]
        if isinstance(m, nn.ReLU):
            if m.inplace:
                continue
        out = m(input_)
        out_sizes.append(np.array(out.size()))
        input_ = out

    total_nums = 0
    for i in range(len(out_sizes)):
        s = out_sizes[i]
        nums = np.prod(np.array(s))
        total_nums += nums


    print('Model {} : intermedite variables: {:3f} M (without backward)'
          .format(model._get_name(), total_nums * type_size / 1000 / 1000))
    print('Model {} : intermedite variables: {:3f} M (with backward)'
          .format(model._get_name(), total_nums * type_size*2 / 1000 / 1000))


def save_best_record(test_info, file_path):
    fo = open(file_path, "w")
    fo.write("epoch: {}\n".format(test_info["epoch"][-1]))
    fo.write(str(test_info["test_AUC"][-1]))
    fo.close()