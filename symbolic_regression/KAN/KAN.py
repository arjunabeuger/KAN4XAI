import torch
import torch.nn as nn
import numpy as np

from KAN.KANLayer import *
from KAN.Symbolic_KANLayer import *
from KAN.LBFGS import *
from KAN.utils import *

import os
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import copy
from sklearn.metrics import r2_score


class KAN(nn.Module):
    # Class attributes and docstrings as previously defined

    def __init__(self, width=None, grid=3, k=3, noise_scale=0.1, noise_scale_base=0.1, base_fun=torch.nn.SiLU(), symbolic_enabled=True, bias_trainable=True, grid_eps=1.0, grid_range=[-1, 1], sp_trainable=True, sb_trainable=True,
                 device='cpu', seed=0):
        super(KAN, self).__init__()

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        ### initializeing the numerical front ###

        self.biases = []
        self.act_fun = []
        self.depth = len(width) - 1
        self.width = width

        for l in range(self.depth):
            scale_base = 1 / np.sqrt(width[l]) + (torch.randn(width[l] * width[l + 1], ) * 2 - 1) * noise_scale_base
            sp_batch = KANLayer(in_dim=width[l], out_dim=width[l + 1], num=grid, k=k, noise_scale=noise_scale, scale_base=scale_base, scale_sp=1., base_fun=base_fun, grid_eps=grid_eps, grid_range=grid_range, sp_trainable=sp_trainable,
                                sb_trainable=sb_trainable, device=device)
            self.act_fun.append(sp_batch)

            bias = nn.Linear(width[l + 1], 1, bias=False, device=device).requires_grad_(bias_trainable)
            bias.weight.data *= 0.
            self.biases.append(bias)

        self.biases = nn.ModuleList(self.biases)
        self.act_fun = nn.ModuleList(self.act_fun)

        self.grid = grid
        self.k = k
        self.base_fun = base_fun

        ### initializing the symbolic front ###
        self.symbolic_fun = []
        for l in range(self.depth):
            sb_batch = Symbolic_KANLayer(in_dim=width[l], out_dim=width[l + 1], device=device)
            self.symbolic_fun.append(sb_batch)

        self.symbolic_fun = nn.ModuleList(self.symbolic_fun)
        self.symbolic_enabled = symbolic_enabled
        
        self.device = device

    def initialize_from_another_model(self, another_model, x):
        another_model(x.to(another_model.device))
        batch = x.shape[0]

        self.initialize_grid_from_another_model(another_model, x.to(another_model.device))

        for l in range(self.depth):
            spb = self.act_fun[l]
            spb_parent = another_model.act_fun[l]

            preacts = another_model.spline_preacts[l]
            postsplines = another_model.spline_postsplines[l]
            self.act_fun[l].coef.data.copy_(curve2coef(preacts.reshape(batch, spb.size).permute(1, 0), postsplines.reshape(batch, spb.size).permute(1, 0), spb.grid, k=spb.k, device=self.device))
            self.act_fun[l].scale_base.data.copy_(spb_parent.scale_base.data)
            self.act_fun[l].scale_sp.data.copy_(spb_parent.scale_sp.data)
            self.act_fun[l].mask.data.copy_(spb_parent.mask.data)

        for l in range(self.depth):
            self.biases[l].weight.data.copy_(another_model.biases[l].weight.data)

        for l in range(self.depth):
            self.symbolic_fun[l] = another_model.symbolic_fun[l]

        return self

    def update_grid_from_samples(self, x):
        for l in range(self.depth):
            self.forward(x)
            self.act_fun[l].update_grid_from_samples(self.acts[l])

    def initialize_grid_from_another_model(self, model, x):
        model(x)
        for l in range(self.depth):
            self.act_fun[l].initialize_grid_from_parent(model.act_fun[l], model.acts[l])

    def forward(self, x):
        self.acts = []
        self.spline_preacts = []
        self.spline_postsplines = []
        self.spline_postacts = []
        self.acts_scale = []
        self.acts_scale_std = []

        self.acts.append(x)

        for l in range(self.depth):
            x_numerical, preacts, postacts_numerical, postspline = self.act_fun[l](x)

            if self.symbolic_enabled:
                x_symbolic, postacts_symbolic = self.symbolic_fun[l](x)
            else:
                x_symbolic = 0.
                postacts_symbolic = 0.

            x = x_numerical + x_symbolic
            postacts = postacts_numerical + postacts_symbolic

            grid_reshape = self.act_fun[l].grid.reshape(self.width[l + 1], self.width[l], -1)
            input_range = grid_reshape[:, :, -1] - grid_reshape[:, :, 0] + 1e-4
            output_range = torch.mean(torch.abs(postacts), dim=0)
            self.acts_scale.append(output_range / input_range)
            self.acts_scale_std.append(torch.std(postacts, dim=0))
            self.spline_preacts.append(preacts.detach().clone())
            self.spline_postacts.append(postacts.detach().clone())
            self.spline_postsplines.append(postspline.detach().clone())

            x = x + self.biases[l].weight
            self.acts.append(x)

        return x

    def set_mode(self, l, i, j, mode, mask_n=None):
        if mode == "s":
            mask_n = 0.
            mask_s = 1.
        elif mode == "n":
            mask_n = 1.
            mask_s = 0.
        elif mode in ["sn", "ns"]:
            if mask_n is None:
                mask_n = 1.
            mask_s = 1.
        else:
            mask_n = 0.
            mask_s = 0.

        self.act_fun[l].mask.data[j * self.act_fun[l].in_dim + i] = mask_n
        self.symbolic_fun[l].mask.data[j, i] = mask_s

    def fix_symbolic(self, l, i, j, fun_name, fit_params_bool=True, a_range=(-10, 10), b_range=(-10, 10), verbose=True, random=False):
        self.set_mode(l, i, j, mode="s")
        if not fit_params_bool:
            self.symbolic_fun[l].fix_symbolic(i, j, fun_name, verbose=verbose, random=random)
            return None
        else:
            x = self.acts[l][:, i]
            y = self.spline_postacts[l][:, j, i]
            r2 = self.symbolic_fun[l].fix_symbolic(i, j, fun_name, x, y, a_range=a_range, b_range=b_range, verbose=verbose)
            return r2

    def unfix_symbolic(self, l, i, j):
        self.set_mode(l, i, j, mode="n")

    def unfix_symbolic_all(self):
        for l in range(len(self.width) - 1):
            for i in range(self.width[l]):
                for j in range(self.width[l + 1]):
                    self.unfix_symbolic(l, i, j)

    def lock(self, l, ids):
        self.act_fun[l].lock(ids)

    def unlock(self, l, ids):
        self.act_fun[l].unlock(ids)

    def get_range(self, l, i, j, verbose=True):
        x = self.spline_preacts[l][:, j, i]
        y = self.spline_postacts[l][:, j, i]
        x_min = torch.min(x)
        x_max = torch.max(x)
        y_min = torch.min(y)
        y_max = torch.max(y)
        if verbose:
            print(f'x range: [{x_min:.2f}, {x_max:.2f}]')
            print(f'y range: [{y_min:.2f}, {y_max:.2f}]')
        return x_min, x_max, y_min, y_max
    
    
    def loss(self):
        """Regularization loss term"""
        reg_loss = 0.0
        for l in range(self.depth):
            layer = self.act_fun[l]
            reg_loss += torch.sum(layer.coef ** 2)  # L2 regularization on coefficients
            reg_loss += torch.sum(layer.scale_base ** 2)  # L2 regularization on scale_base
            reg_loss += torch.sum(layer.scale_sp ** 2)  # L2 regularization on scale_sp
        return reg_loss

    def get_loss(self):
        """Get total loss for the network"""
        return self.loss()

    def plot(self, folder="./figures", beta=3, mask=False, mode="supervised", scale=0.5, tick=False, sample=False, in_vars=None, out_vars=None, title=None):
        if not os.path.exists(folder):
            os.makedirs(folder)
        # matplotlib.use('Agg')
        depth = len(self.width) - 1
        for l in range(depth):
            w_large = 2.0
            for i in range(self.width[l]):
                for j in range(self.width[l + 1]):
                    rank = torch.argsort(self.acts[l][:, i])
                    fig, ax = plt.subplots(figsize=(w_large, w_large))

                    num = rank.shape[0]

                    symbol_mask = self.symbolic_fun[l].mask[j][i]
                    numerical_mask = self.act_fun[l].mask.reshape(self.width[l + 1], self.width[l])[j][i]
                    if symbol_mask > 0. and numerical_mask > 0.:
                        color = 'purple'
                        alpha_mask = 1
                    if symbol_mask > 0. and numerical_mask == 0.:
                        color = "red"
                        alpha_mask = 1
                    if symbol_mask == 0. and numerical_mask > 0.:
                        color = "black"
                        alpha_mask = 1
                    if symbol_mask == 0. and numerical_mask == 0.:
                        color = "white"
                        alpha_mask = 0

                    if tick == True:
                        ax.tick_params(axis="y", direction="in", pad=-22, labelsize=50)
                        ax.tick_params(axis="x", direction="in", pad=-15, labelsize=50)
                        x_min, x_max, y_min, y_max = self.get_range(l, i, j, verbose=False)
                        plt.xticks([x_min, x_max], ['%2.f' % x_min, '%2.f' % x_max])
                        plt.yticks([y_min, y_max], ['%2.f' % y_min, '%2.f' % y_max])
                    else:
                        plt.xticks([])
                        plt.yticks([])
                    if alpha_mask == 1:
                        plt.gca().patch.set_edgecolor('black')
                    else:
                        plt.gca().patch.set_edgecolor('white')
                    plt.gca().patch.set_linewidth(1.5)
                    # plt.axis('off')

                    plt.plot(self.acts[l][:, i][rank].cpu().detach().numpy(), self.spline_postacts[l][:, j, i][rank].cpu().detach().numpy(), color=color, lw=5)
                    if sample == True:
                        plt.scatter(self.acts[l][:, i][rank].cpu().detach().numpy(), self.spline_postacts[l][:, j, i][rank].cpu().detach().numpy(), color=color, s=400 * scale ** 2)
                    plt.gca().spines[:].set_color(color)

                    lock_id = self.act_fun[l].lock_id[j * self.width[l] + i].long().item()
                    if lock_id > 0:
                        im = plt.imread(f'{folder}/lock.png')
                        newax = fig.add_axes([0.15, 0.7, 0.15, 0.15])
                        plt.text(500, 400, lock_id, fontsize=15)
                        newax.imshow(im)
                        newax.axis('off')

                    plt.savefig(f'{folder}/sp_{l}_{i}_{j}.png', bbox_inches="tight", dpi=400)
                    plt.close()

        def score2alpha(score):
            return np.tanh(beta * score)

        if mode == "supervised":
            alpha = [score2alpha(score.cpu().detach().numpy()) for score in self.acts_scale]
        elif mode == "unsupervised":
            alpha = [score2alpha(score.cpu().detach().numpy()) for score in self.acts_scale_std]

        width = np.array(self.width)
        A = 1
        y0 = 0.4

        neuron_depth = len(width)
        min_spacing = A / np.maximum(np.max(width), 5)

        max_neuron = np.max(width)
        max_num_weights = np.max(width[:-1] * width[1:])
        y1 = 0.4 / np.maximum(max_num_weights, 3)

        fig, ax = plt.subplots(figsize=(10 * scale, 10 * scale * (neuron_depth - 1) * y0))

        for l in range(neuron_depth):
            n = width[l]
            spacing = A / n
            for i in range(n):
                plt.scatter(1 / (2 * n) + i / n, l * y0, s=min_spacing ** 2 * 10000 * scale ** 2, color='black')

                if l < neuron_depth - 1:
                    n_next = width[l + 1]
                    N = n * n_next
                    for j in range(n_next):
                        id_ = i * n_next + j

                        symbol_mask = self.symbolic_fun[l].mask[j][i]
                        numerical_mask = self.act_fun[l].mask.reshape(self.width[l + 1], self.width[l])[j][i]
                        if symbol_mask == 1. and numerical_mask == 1.:
                            color = 'purple'
                            alpha_mask = 1.
                        if symbol_mask == 1. and numerical_mask == 0.:
                            color = "red"
                            alpha_mask = 1.
                        if symbol_mask == 0. and numerical_mask == 1.:
                            color = "black"
                            alpha_mask = 1.
                        if symbol_mask == 0. and numerical_mask == 0.:
                            color = "white"
                            alpha_mask = 0.
                        if mask == True:
                            plt.plot([1 / (2 * n) + i / n, 1 / (2 * N) + id_ / N], [l * y0, (l + 1 / 2) * y0 - y1], color=color, lw=2 * scale, alpha=alpha[l][j][i] * self.mask[l][i].item() * self.mask[l + 1][j].item())
                            plt.plot([1 / (2 * N) + id_ / N, 1 / (2 * n_next) + j / n_next], [(l + 1 / 2) * y0 + y1, (l + 1) * y0], color=color, lw=2 * scale, alpha=alpha[l][j][i] * self.mask[l][i].item() * self.mask[l + 1][j].item())
                        else:
                            plt.plot([1 / (2 * n) + i / n, 1 / (2 * N) + id_ / N], [l * y0, (l + 1 / 2) * y0 - y1], color=color, lw=2 * scale, alpha=alpha[l][j][i] * alpha_mask)
                            plt.plot([1 / (2 * N) + id_ / N, 1 / (2 * n_next) + j / n_next], [(l + 1 / 2) * y0 + y1, (l + 1) * y0], color=color, lw=2 * scale, alpha=alpha[l][j][i] * alpha_mask)

            plt.xlim(0, 1)
            plt.ylim(-0.1 * y0, (neuron_depth - 1 + 0.1) * y0)

        DC_to_FC = ax.transData.transform
        FC_to_NFC = fig.transFigure.inverted().transform
        DC_to_NFC = lambda x: FC_to_NFC(DC_to_FC(x))

        plt.axis('off')

        for l in range(neuron_depth - 1):
            n = width[l]
            for i in range(n):
                n_next = width[l + 1]
                N = n * n_next
                for j in range(n_next):
                    id_ = i * n_next + j
                    im = plt.imread(f'{folder}/sp_{l}_{i}_{j}.png')
                    left = DC_to_NFC([1 / (2 * N) + id_ / N - y1, 0])[0]
                    right = DC_to_NFC([1 / (2 * N) + id_ / N + y1, 0])[0]
                    bottom = DC_to_NFC([0, (l + 1 / 2) * y0 - y1])[1]
                    up = DC_to_NFC([0, (l + 1 / 2) * y0 + y1])[1]
                    newax = fig.add_axes([left, bottom, right - left, up - bottom])
                    if mask == False:
                        newax.imshow(im, alpha=alpha[l][j][i])
                    else:
                        newax.imshow(im, alpha=alpha[l][j][i] * self.mask[l][i].item() * self.mask[l + 1][j].item())
                    newax.axis('off')

        if in_vars != None:
            n = self.width[0]
            for i in range(n):
                plt.gcf().get_axes()[0].text(1 / (2 * (n)) + i / (n), -0.1, in_vars[i], fontsize=40 * scale, horizontalalignment='center', verticalalignment='center')

        if out_vars != None:
            n = self.width[-1]
            for i in range(n):
                plt.gcf().get_axes()[0].text(1 / (2 * (n)) + i / (n), y0 * (len(self.width) - 1) + 0.1, out_vars[i], fontsize=40 * scale, horizontalalignment='center', verticalalignment='center')

        if title != None:
            plt.gcf().get_axes()[0].text(0.5, y0 * (len(self.width) - 1) + 0.2, title, fontsize=40 * scale, horizontalalignment='center', verticalalignment='center')

    def train(self, dataset, opt="LBFGS", steps=100, log=1, lamb=0., lamb_l1=1., lamb_entropy=2., lamb_coef=0., lamb_coefdiff=0., update_grid=True, grid_update_num=10, loss_fn=None, lr=1e-1, stop_grid_update_step=50, batch=-1,
            small_mag_threshold=1e-16, small_reg_factor=1., metrics=None, sglr_avoid=False, save_fig=False, in_vars=None, out_vars=None, beta=3, save_fig_freq=1, img_folder='./video', device='cpu'):
        def reg(acts_scale):
            def nonlinear(x, th=small_mag_threshold, factor=small_reg_factor):
                return (x < th) * x * factor + (x > th) * (x + (factor - 1) * th)

            reg_ = 0.
            for i in range(len(acts_scale)):
                vec = acts_scale[i].reshape(-1, )

                p = vec / torch.sum(vec)
                l1 = torch.sum(nonlinear(vec))
                entropy = - torch.sum(p * torch.log2(p + 1e-4))
                reg_ += lamb_l1 * l1 + lamb_entropy * entropy

            for i in range(len(self.act_fun)):
                coeff_l1 = torch.sum(torch.mean(torch.abs(self.act_fun[i].coef), dim=1))
                coeff_diff_l1 = torch.sum(torch.mean(torch.abs(torch.diff(self.act_fun[i].coef)), dim=1))
                reg_ += lamb_coef * coeff_l1 + lamb_coefdiff * coeff_diff_l1

            return reg_
        
        steps = steps + 0
        pbar = tqdm(range(steps), desc='description', ncols=100)
        summary_step = 10

        if loss_fn is None:
            loss_fn = loss_fn_eval = lambda x, y: torch.mean((x - y) ** 2)
        else:
            loss_fn = loss_fn_eval = loss_fn

        grid_update_freq = int(stop_grid_update_step / grid_update_num)

        if opt == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=0.05)
        elif opt == "LBFGS":
            optimizer = LBFGS(self.parameters(), lr=lr, history_size=10, line_search_fn="strong_wolfe", tolerance_grad=1e-32, tolerance_change=1e-32, tolerance_ys=1e-32)

        results = {'train_loss': [], 'test_R^2': [], 'test_loss': [], 'reg': []}
        if metrics is not None:
            for metric in metrics:
                results[metric.__name__] = []

        if batch == -1 or batch > dataset['train_input'].shape[0]:
            batch_size = dataset['train_input'].shape[0]
            batch_size_test = dataset['test_input'].shape[0]
        else:
            batch_size = batch
            batch_size_test = batch

        global train_loss, reg_

        def closure():
            global train_loss, reg_
            optimizer.zero_grad()
            pred = self.forward(dataset['train_input'][train_id].to(device))
            if sglr_avoid:
                id_ = torch.where(torch.isnan(torch.sum(pred, dim=1)) == False)[0]
                train_loss = loss_fn(pred[id_], dataset['train_label'][train_id][id_].to(device))
            else:
                train_loss = loss_fn(pred, dataset['train_label'][train_id].to(device))
            reg_ = reg(self.acts_scale)
            objective = train_loss + lamb * reg_
            objective.backward()
            return objective

        if save_fig and not os.path.exists(img_folder):
            os.makedirs(img_folder)

        for _ in pbar:
            train_id = np.random.choice(dataset['train_input'].shape[0], batch_size, replace=False)
            test_id = np.random.choice(dataset['test_input'].shape[0], batch_size_test, replace=False)

            if _ % grid_update_freq == 0 and _ < stop_grid_update_step and update_grid:
                self.update_grid_from_samples(dataset['train_input'][train_id].to(device))

            if opt == "LBFGS":
                optimizer.step(closure)
            elif opt == "Adam":
                pred = self.forward(dataset['train_input'][train_id].to(device))
                if sglr_avoid:
                    id_ = torch.where(torch.isnan(torch.sum(pred, dim=1)) == False)[0]
                    train_loss = loss_fn(pred[id_], dataset['train_label'][train_id][id_].to(device))
                else:
                    train_loss = loss_fn(pred, dataset['train_label'][train_id].to(device))
                reg_ = reg(self.acts_scale)
                loss = train_loss + lamb * reg_
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # scheduler.step(loss)

            if _ % summary_step == 0:
                test_loss = loss_fn_eval(self.forward(dataset['test_input'][test_id].to(device)), dataset['test_label'][test_id].to(device))
                results['test_R^2'].append(r2_score(dataset['test_label'][test_id].cpu().detach().numpy(), self.forward(dataset['test_input'][test_id].to(device)).cpu().detach().numpy()))
                results['train_loss'].append(torch.sqrt(train_loss).cpu().detach().numpy())
                results['test_loss'].append(torch.sqrt(test_loss).cpu().detach().numpy())
                results['reg'].append(reg_.cpu().detach().numpy())

            if _ % log == 0:
                pbar.set_description("train loss: %.2e | test loss: %.2e | reg: %.2e | lr: %.2e " % (torch.sqrt(train_loss).cpu().detach().numpy(), torch.sqrt(test_loss).cpu().detach().numpy(), reg_.cpu().detach().numpy(), optimizer.param_groups[0]['lr']))

            if metrics is not None:
                for metric in metrics:
                    results[metric.__name__].append(metric().item())

            if save_fig and _ % save_fig_freq == 0:
                self.plot(folder=img_folder, in_vars=in_vars, out_vars=out_vars, title="Step {}".format(_), beta=beta)
                plt.savefig(img_folder + '/' + str(_) + '.jpg', bbox_inches='tight', dpi=200)
                plt.close()

        return results


    def prune(self, threshold=1e-2, mode="auto", active_neurons_id=None):
        mask = [torch.ones(self.width[0], )]
        active_neurons = [list(range(self.width[0]))]
        for i in range(len(self.acts_scale) - 1):
            if mode == "auto":
                in_important = torch.max(self.acts_scale[i], dim=1)[0] > threshold
                out_important = torch.max(self.acts_scale[i + 1], dim=0)[0] > threshold
                overall_important = in_important * out_important
            elif mode == "manual":
                overall_important = torch.zeros(self.width[i + 1], dtype=torch.bool)
                overall_important[active_neurons_id[i + 1]] = True
            mask.append(overall_important.float())
            active_neurons.append(torch.where(overall_important == True)[0])
        active_neurons.append(list(range(self.width[-1])))
        mask.append(torch.ones(self.width[-1], ))

        self.mask = mask

        for l in range(len(self.acts_scale) - 1):
            for i in range(self.width[l + 1]):
                if i not in active_neurons[l + 1]:
                    self.remove_node(l + 1, i)

        model2 = KAN(copy.deepcopy(self.width), self.grid, self.k, base_fun=self.base_fun, device=self.device)
        model2.load_state_dict(self.state_dict())
        for i in range(len(self.acts_scale)):
            if i < len(self.acts_scale) - 1:
                model2.biases[i].weight.data = model2.biases[i].weight.data[:, active_neurons[i + 1]]

            model2.act_fun[i] = model2.act_fun[i].get_subset(active_neurons[i], active_neurons[i + 1])
            model2.width[i] = len(active_neurons[i])
            model2.symbolic_fun[i] = self.symbolic_fun[i].get_subset(active_neurons[i], active_neurons[i + 1])

        return model2

    def remove_edge(self, l, i, j):
        self.act_fun[l].mask[j * self.width[l] + i] = 0.

    def remove_node(self, l, i):
        self.act_fun[l - 1].mask[i * self.width[l - 1] + torch.arange(self.width[l - 1])] = 0.
        self.act_fun[l].mask[torch.arange(self.width[l + 1]) * self.width[l] + i] = 0.
        self.symbolic_fun[l - 1].mask[i, :] *= 0.
        self.symbolic_fun[l].mask[:, i] *= 0.

    def suggest_symbolic(self, l, i, j, a_range=(-1, 1), b_range=(-1, 1), lib=None, topk=5, verbose=True):
        r2s = []

        if lib == None:
            symbolic_lib = SYMBOLIC_LIB
        else:
            symbolic_lib = {}
            for item in lib:
                symbolic_lib[item] = SYMBOLIC_LIB[item]

        for (name, fun) in symbolic_lib.items():
            r2 = self.fix_symbolic(l, i, j, name, a_range=a_range, b_range=b_range, verbose=False)
            r2s.append(r2.item())

        self.unfix_symbolic(l, i, j)

        sorted_ids = np.argsort(r2s)[::-1][:topk]
        r2s = np.array(r2s)[sorted_ids][:topk]
        topk = np.minimum(topk, len(symbolic_lib))
        if verbose == True:
            print('function', ',', 'r2')
            for i in range(topk):
                print(list(symbolic_lib.items())[sorted_ids[i]][0], ',', r2s[i])

        best_name = list(symbolic_lib.items())[sorted_ids[0]][0]
        best_fun = list(symbolic_lib.items())[sorted_ids[0]][1]
        best_r2 = r2s[0]
        return best_name, best_fun, best_r2

    def auto_symbolic(self, a_range=(-10, 10), b_range=(-10, 10), lib=None, verbose=1):
        for l in range(len(self.width) - 1):
            for i in range(self.width[l]):
                for j in range(self.width[l + 1]):
                    if self.symbolic_fun[l].mask[j, i] > 0.:
                        print(f'skipping ({l},{i},{j}) since already symbolic')
                    else:
                        name, fun, r2 = self.suggest_symbolic(l, i, j, a_range=a_range, b_range=b_range, lib=lib, verbose=False)
                        self.fix_symbolic(l, i, j, name, verbose=verbose > 1)
                        if verbose >= 1:
                            print(f'fixing ({l},{i},{j}) with {name}, r2={r2}')

    def symbolic_formula(self, floating_digit=2, var=None, normalizer=None, simplify=True, output_normalizer=None):
        symbolic_acts = []
        x = []

        def ex_round(ex1, floating_digit=floating_digit):
            ex2 = ex1
            for a in sympy.preorder_traversal(ex1):
                if isinstance(a, sympy.Float):
                    ex2 = ex2.subs(a, round(a, floating_digit))
            return ex2

        if var == None:
            for ii in range(1, self.width[0] + 1):
                exec(f"x{ii} = sympy.Symbol('x_{ii}')")
                exec(f"x.append(x{ii})")
        else:
            x = [sympy.symbols(var_) for var_ in var]

        x0 = x

        if normalizer != None:
            mean = normalizer[0]
            std = normalizer[1]
            x = [(x[i] - mean[i]) / std[i] for i in range(len(x))]

        symbolic_acts.append(x)

        for l in range(len(self.width) - 1):
            y = []
            for j in range(self.width[l + 1]):
                yj = 0.
                for i in range(self.width[l]):
                    a, b, c, d = self.symbolic_fun[l].affine[j, i]
                    sympy_fun = self.symbolic_fun[l].funs_sympy[j][i]
                    try:
                        yj += c * sympy_fun(a * x[i] + b) + d
                    except:
                        print('make sure all activations need to be converted to symbolic formulas first!')
                        return
                if simplify == True:
                    y.append(sympy.simplify(yj + self.biases[l].weight.data[0, j]))
                else:
                    y.append(yj + self.biases[l].weight.data[0, j])

            x = y
            symbolic_acts.append(x)

        if output_normalizer != None:
            output_layer = symbolic_acts[-1]
            means = output_normalizer[0]
            stds = output_normalizer[1]

            assert len(output_layer) == len(means), 'output_normalizer does not match the output layer'
            assert len(output_layer) == len(stds), 'output_normalizer does not match the output layer'
            
            output_layer = [(output_layer[i] * stds[i] + means[i]) for i in range(len(output_layer))]
            symbolic_acts[-1] = output_layer

        self.symbolic_acts = [[ex_round(symbolic_acts[l][i]) for i in range(len(symbolic_acts[l]))] for l in range(len(symbolic_acts))]

        out_dim = len(symbolic_acts[-1])
        return [ex_round(symbolic_acts[-1][i]) for i in range(len(symbolic_acts[-1]))], x0

    def clear_ckpts(self, folder='./model_ckpt'):
        if os.path.exists(folder):
            files = glob.glob(folder + '/*')
            for f in files:
                os.remove(f)
        else:
            os.makedirs(folder)

    def save_ckpt(self, name, folder='./model_ckpt'):
        if not os.path.exists(folder):
            os.makedirs(folder)

        torch.save(self.state_dict(), folder + '/' + name)
        print('save this model to', folder + '/' + name)

    def load_ckpt(self, name, folder='./model_ckpt'):
        self.load_state_dict(torch.load(folder + '/' + name))


