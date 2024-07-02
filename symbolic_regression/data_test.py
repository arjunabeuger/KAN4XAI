"""Trains the deep symbolic regression architecture on given functions to produce a simple equation that describes
the dataset. Uses L_0 regularization for the EQL network."""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from inspect import signature
from tqdm import trange
import argparse
import pickle
import json
from KAN.KAN import *

N_TRAIN = 100 #10_000  #1_000_000 # Size of training dataset
N_VAL = 10  #1000 #100_000    # Size of validation dataset
DOMAIN = (-1, 1)    # Domain of dataset - range from which we sample x
# DOMAIN = np.array([[0, -1, -1], [1, 1, 1]])   # Use this format if each input variable has a different domain
N_TEST = 10        # Size of test dataset
DOMAIN_TEST = (-2, 2)   # Domain of test dataset - should be larger than training domain to test extrapolation
NOISE_SD = 0        # Standard deviation of noise for training dataset
var_names = ["x", "y", "z", "a", "b", "c", "d", "e", "f"]

# Define functions module or import it if it's available
# import functions

# Define your functions here or import them if they're available
# import functions

def generate_data(func, N, range_min=-1, range_max=1):
    x_dim = len(signature(func).parameters)
    x = (range_max - range_min) * torch.rand([N, x_dim]) + range_min
    y = torch.tensor([[func(*x_i)] for x_i in x])
    return x, y

class Benchmark:
    def __init__(self, results_dir, n_layers=2, reg_weight=5e-3, learning_rate=1e-2,
                 n_epochs1=10001, n_epochs2=10001):
        self.activation_funcs = [
            # Define activation functions here
        ]
        self.n_layers = n_layers
        self.reg_weight = reg_weight
        self.learning_rate = learning_rate
        self.summary_step = 10
        self.n_epochs1 = n_epochs1
        self.n_epochs2 = n_epochs2

        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        self.results_dir = results_dir

        result = {
            "learning_rate": self.learning_rate,
            "summary_step": self.summary_step,
            "n_epochs1": self.n_epochs1,
            "n_epochs2": self.n_epochs2,
            "activation_funcs_name": [func.name for func in self.activation_funcs],
            "n_layers": self.n_layers,
            "reg_weight": self.reg_weight,
        }
        with open(os.path.join(self.results_dir, 'params.pickle'), "wb+") as f:
            pickle.dump(result, f)

    def benchmark(self, func, func_name, trials, N_sizes):
        print("Starting benchmark for function:\t%s" % func_name)
        print("==============================================")

        all_models = []
        for N in N_sizes:
            print(f"Running benchmark for N={N}")
            func_dir = os.path.join(self.results_dir, f"{func_name}_N_{N}")
            if not os.path.exists(func_dir):
                os.makedirs(func_dir)

            model = self.train(func, func_name, trials, func_dir, N)
            all_models.append(model)

        self.plot(all_models, folder=self.results_dir, title=f"Benchmark Results for {func_name}")

    def train(self, func, func_name, trials, func_dir, N):
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        print("Use cuda:", use_cuda, "Device:", device)

        x, y = generate_data(func, N)
        data, target = x.to(device), y.to(device)
        x_test, y_test = generate_data(func, N_TEST, range_min=DOMAIN_TEST[0], range_max=DOMAIN_TEST[1])
        test_data, test_target = x_test.to(device), y_test.to(device)

        x_dim = len(signature(func).parameters)
        
        pbar = trange(trials)
        print("Training on function " + func_name)

        best_model = None
        best_test_loss = float('inf')

        for trial in pbar:
            print(f"Training on function {func_name} Trial {trial + 1} out of {trials}")

            kan = KAN(width=[x_dim, 1, 1], grid=2, k=4)
            def count_parameters(model):
                return sum(p.numel() for p in model.parameters() if p.requires_grad)

            print(f"KAN parameters: {count_parameters(kan)}")

            dataset = {'train_input': data, 'train_label': target, 'test_input': test_data, 'test_label': test_target}

            zero_th = 100
            kan_results = kan.train(dataset, opt="Adam", steps=self.n_epochs1 + self.n_epochs2 + zero_th, lr=self.learning_rate)
            test_loss = np.array(kan_results['test_loss'])[-1].item()
            print(f"\nKAN MSE {test_loss}")
            print(f"KAN R^2: {kan_results['test_R^2'][-1]}")

            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_model = kan

        return best_model

    def plot(self, models, folder="./figures", title=None):
        if not os.path.exists(folder):
            os.makedirs(folder)

        combined_fig, combined_axes = plt.subplots(1, len(models), figsize=(10 * len(models), 10))
        if len(models) == 1:
            combined_axes = [combined_axes]

        for model_idx, model in enumerate(models):
            l = 0
            i = 0
            j = 0

            rank = torch.argsort(model.acts[l][:, i])
            ax = combined_axes[model_idx]

            symbol_mask = model.symbolic_fun[l].mask[j][i]
            numerical_mask = model.act_fun[l].mask.reshape(model.width[l + 1], model.width[l])[j][i]
            if symbol_mask > 0. and numerical_mask > 0.:
                color = 'purple'
            elif symbol_mask > 0. and numerical_mask == 0.:
                color = "red"
            elif symbol_mask == 0. and numerical_mask > 0.:
                color = "black"
            else:
                color = "white"

            ax.plot(model.acts[l][:, i][rank].cpu().detach().numpy(), model.spline_postacts[l][:, j, i][rank].cpu().detach().numpy(), color=color, lw=5)

            ax.set_title(f'Model {model_idx + 1}')

        if title:
            combined_fig.suptitle(title)

        combined_fig.savefig('combined_plot.png', bbox_inches="tight", dpi=400)
        plt.close(combined_fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the EQL network.")
    parser.add_argument("--results-dir", type=str, default='results/benchmark/test')
    parser.add_argument("--n-layers", type=int, default=3, help="Number of hidden layers, L")
    parser.add_argument("--reg-weight", type=float, default=5e-3, help='Regularization weight, lambda')
    parser.add_argument('--learning-rate', type=float, default=1e-2, help='Base learning rate for training')
    parser.add_argument("--n-epochs1", type=int, default=500, help="Number of epochs to train the first stage")
    parser.add_argument("--n-epochs2", type=int, default=500, help="Number of epochs to train the second stage, after freezing weights.")
    parser.add_argument("--N-sizes", type=int, nargs='+', default=[ 2, 3, 4, 5, 7, 10, 100, 1_000], help="Different sizes of N to benchmark")

    args = parser.parse_args()
    kwargs = vars(args)
    print(kwargs)

    bench = Benchmark(
        results_dir=kwargs['results_dir'],
        n_layers=kwargs['n_layers'],
        reg_weight=kwargs['reg_weight'],
        learning_rate=kwargs['learning_rate'],
        n_epochs1=kwargs['n_epochs1'],
        n_epochs2=kwargs['n_epochs2']
    )
    func = lambda x: torch.sigmoid(x)
    func_name = "sigmoid"
    trials = 1

    N_sizes = kwargs["N_sizes"]
    bench.benchmark(func, func_name, trials, N_sizes)
 
    # bench.benchmark(lambda x: x**2, func_name="x^2", trials=trial)
    # bench.benchmark(lambda x: x**3, func_name="x^3", trials=trial)
    # bench.benchmark(lambda x: np.sin(2*np.pi*x), func_name="sin(2pix)", trials=trial)
    # bench.benchmark(lambda x: np.exp(x), func_name="e^x", trials=trial)
    # bench.benchmark(lambda x, y: x*y, func_name="xy", trials=trial)
    # bench.benchmark(lambda x, y: np.sin(2 * np.pi * x) + np.sin(4*np.pi * y),
    #                 func_name="sin(2pix)+sin(4py)", trials=trial)
    # bench.benchmark(lambda x, y, z: 0.5*x*y + 0.5*z, func_name="0.5xy+0.5z", trials=trial)
    # bench.benchmark(lambda x, y, z: x**2 + y - 2*z, func_name="x^2+y-2z", trials=trial)
    # bench.benchmark(lambda x: np.exp(-x**2), func_name="e^-x^2", trials=trial)
    # bench.benchmark(lambda x: 1 / (1 + np.exp(-10*x)), func_name="sigmoid(10x)", trials=trial)
    # bench.benchmark(lambda x, y: x**2 + np.sin(2*np.pi*y), func_name="x^2+sin(2piy)", trials=trial)

    # # 3-layer functions
    # bench.benchmark(lambda x, y, z: (x+y*z)**3, func_name="(x+yz)^3", trials=trial)


    # NGUYEN equations
    # bench.benchmark(lambda x: x**3 + x**2 + x, func_name="x^3+x^2+x", trials=trial)
    # bench.benchmark(lambda x: x**5 + x**4 + x**3 + x**2 + x, func_name="x^5+x^4+x^3+x^2+x", trials=trial)
    # bench.benchmark(lambda x: np.sin(x**2) * np.cos(x) - 1, func_name="sin(x^2)*cos(x)-1", trials=trial)
    # bench.benchmark(lambda x: np.sin(x) + np.sin(x + x**2), func_name="sin(x)+sin(x+x^2)", trials=trial)
    # # bench.benchmark(lambda x: np.log(x + 1) + np.log(x**2 + 1), func_name="log(x+1)+log(x^2+1)", trials=trial)
    # # bench.benchmark(lambda x: np.sqrt(x), func_name="sqrt(x)", trials=trial) # Gives NaNs
    # bench.benchmark(lambda x0, x1: 2 * np.sin(x0) * np.cos(x1), func_name="2*sin(x0)*cos(x1)", trials=trial)
    # bench.benchmark(lambda x0, x1: x0**3 - 0.5 * x1**2, func_name="x0^3-0.5*x1^2", trials=trial)
    
    # # Feynman equations
    
    
    # bench.benchmark(lambda v1: np.exp(-v1**2/2)/np.sqrt(2*np.pi), func_name="I.6.2a", trials=trial)
    # bench.benchmark(lambda v1, v2: np.exp(-(v2/v1)**2/2)/(np.sqrt(2*np.pi)*v1), func_name="I.6.2", trials=trial)
    # bench.benchmark(lambda v1, v2, v3: np.exp(-((v2-v3)/v1)**2/2)/(np.sqrt(2*np.pi)*v1), func_name="I.6.2b", trials=trial)
    # bench.benchmark(lambda v1, v2, v3, v4: np.sqrt((v2-v1)**2+(v4-v3)**2), func_name="I.8.14", trials=trial)
    # bench.benchmark(lambda v1, v2, v3, v4, v5, v6, v7, v8, v9: v3*v1*v2/((v5-v4)**2+(v7-v6)**2+(v9-v8)**2), func_name="I.9.18", trials=trial)
    # # bench.benchmark(lambda v1, v2, v3: v1/np.sqrt(1-v2**2/v3**2), func_name="I.10.7", trials=trial) # Gives NaNs
    # bench.benchmark(lambda v1, v2, v3, v4, v5, v6: v1*v4+v2*v5+v3*v6, func_name="I.11.19", trials=trial)
    # bench.benchmark(lambda v1, v2: v1*v2, func_name="I.12.1", trials=trial)
    # bench.benchmark(lambda v1, v2, v3, v4: v1*v2*v4/(4*np.pi*v3*v4**3), func_name="I.12.2", trials=trial)
    # bench.benchmark(lambda v1, v2, v3: v1*v3/(4*np.pi*v2*v3**3), func_name="I.12.4", trials=trial)
    # bench.benchmark(lambda v1, v2: v1*v2, func_name="I.12.5", trials=trial)
    # bench.benchmark(lambda v1, v2, v3, v4, v5: v1*(v2+v3*v4*np.sin(v5)), func_name="I.12.11", trials=trial)
    # bench.benchmark(lambda v1, v2, v3, v4: 1/2*v1*(v2**2+v3**2+v4**2), func_name="I.13.4", trials=trial)
    # bench.benchmark(lambda v1, v2, v3, v4, v5: v5*v1*v2*(1/v4-1/v3), func_name="I.13.12", trials=trial)
    # bench.benchmark(lambda v1, v2, v3: v1*v2*v3, func_name="I.14.3", trials=trial)
    # bench.benchmark(lambda v1, v2: 1/2*v1*v2**2, func_name="I.14.4", trials=trial)
    # bench.benchmark(lambda v1, v2, v3, v4: (v1-v2*v4)/np.sqrt(1-v2**2/v3**2), func_name="I.15.3x", trials=trial) # Gives NaNs
    # bench.benchmark(lambda v1, v2, v3, v4: (v4-v3*v1/v2**2)/np.sqrt(1-v3**2/v2**2), func_name="I.15.3t", trials=trial) # Gives NaNs
    # bench.benchmark(lambda v1, v2, v3: v1*v2/np.sqrt(1-v2**2/v3**2), func_name="I.15.1", trials=trial) # Gives NaNs
    # bench.benchmark(lambda v1, v2, v3: (v3+v2)/(1+v3*v2/v1**2), func_name="I.16.6", trials=trial)
    # bench.benchmark(lambda v1, v2, v3, v4: (v1*v3+v2*v4)/(v1+v2), func_name="I.18.4", trials=trial)
    # bench.benchmark(lambda v1, v2: v1*v2*np.sin(theta), func_name="I.18.12", trials=trial)
    # bench.benchmark(lambda v1, v2, v3: v1*v2*v3*np.sin(theta), func_name="I.18.14", trials=trial)
    # bench.benchmark(lambda v1, v2, v3, v4: 1/2*v1*(v2**2+v3**2)*1/2*v4**2, func_name="I.24.6", trials=trial)
    # bench.benchmark(lambda v1, v2: v1/v2, func_name="I.25.13", trials=trial)
    # bench.benchmark(lambda v1, v2: np.arcsin(v1*np.sin(v2)), func_name="I.26.2", trials=trial)
    # bench.benchmark(lambda v1, v2, v3: 1/(1/v1+v3/v2), func_name="I.27.6", trials=trial)
    # bench.benchmark(lambda v1, v2: v1/v2, func_name="I.29.4", trials=trial)
    # bench.benchmark(lambda v1, v2, v3, v4: np.sqrt(v1**2+v2**2-2*v1*v2*np.cos(v3-v4)), func_name="I.29.16", trials=trial)
    # bench.benchmark(lambda v1, v2, v3: v1*np.sin(v3*v2/2)**2/np.sin(v2/2)**2, func_name="I.30.3", trials=trial)
    # bench.benchmark(lambda v1, v2, v3: np.arcsin(v1/(v3*v2)), func_name="I.30.5", trials=trial)
    # bench.benchmark(lambda v1, v2, v3, v4: v1**2*v2**2/(6*np.pi*v3*v4**3), func_name="I.32.5", trials=trial)
    # bench.benchmark(lambda v1, v2, v3, v4, v5, v6: (1/2*v1*v2*v3**2)*(8*np.pi*v4**2/3)*(v5**4/(v5**2-v6**2)**2), func_name="I.32.17", trials=trial)
    # bench.benchmark(lambda v1, v2, v3, v4: v1*v2*v3/v4, func_name="I.34.8", trials=trial)
    # bench.benchmark(lambda v1, v2, v3: v3/(1-v2/v1), func_name="I.34.1", trials=trial)
    # bench.benchmark(lambda v1, v2, v3: (1+v2/v1)/np.sqrt(1-v2**2/v1**2)*v3, func_name="I.34.14", trials=trial) # Gives NaNs
    # bench.benchmark(lambda v1, v2: (v2/(2*np.pi))*v1, func_name="I.34.27", trials=trial)
    # bench.benchmark(lambda v1, v2, v3: v1+v2+2*np.sqrt(v1*v2)*np.cos(v3), func_name="I.37.4", trials=trial) # Gives NaNs
    # bench.benchmark(lambda v1, v2, v3: 4*np.pi*epsilon*(v3/(2*np.pi))**2/(v1*v2**2), func_name="I.38.12", trials=trial)
    # bench.benchmark(lambda v1, v2: 3/2*v1*v2, func_name="I.39.1", trials=trial)
    # bench.benchmark(lambda v1, v2, v3: 1/(v1-1)*v2*v3, func_name="I.39.11", trials=trial)
    # bench.benchmark(lambda v1, v2, v3, v4: v1*v4*v2/v3, func_name="I.39.22", trials=trial)
    # bench.benchmark(lambda v1, v2, v3, v4, v5, v6: v1*np.exp(-v2*v5*v3/(v6*v4)), func_name="I.40.1", trials=trial) # Gives NaNs
    # bench.benchmark(lambda v1, v2, v3, v4, v5: v3/(2*np.pi)*v1**3/(np.pi**2*v5**2*(np.exp((v3/(2*np.pi))*v1/(v4*v2))-1)), func_name="I.41.16", trials=trial)
    # bench.benchmark(lambda v1, v2, v3, v4: v1*v2*v3/v4, func_name="I.43.16", trials=trial)
    # bench.benchmark(lambda v1, v2, v3: v1*v3*v2, func_name="I.43.31", trials=trial)
    # bench.benchmark(lambda v1, v2, v3, v4: 1/(v1-1)*v2*v4/v3, func_name="I.43.43", trials=trial)
    # bench.benchmark(lambda v1, v2, v3, v4, v5: v1*v2*v3*np.log(v5/v4), func_name="I.44.4", trials=trial) # Gives NaNs
    # bench.benchmark(lambda v1, v2, v3: np.sqrt(v1*v2/v3), func_name="I.47.23", trials=trial) # Gives NaNs
    # bench.benchmark(lambda v1, v2, v3: v1*v3**2/np.sqrt(1-v2**2/v3**2), func_name="I.48.2", trials=trial) # Gives NaNs
    # bench.benchmark(lambda v1, v2, v3, v4: v1*(np.cos(v2*v3)+v4*np.cos(v2*v3)**2), func_name="I.50.26", trials=trial)
    # bench.benchmark(lambda v1, v2, v3, v4, v5: v1*(v3-v2)*v4/v5, func_name="II.2.42", trials=trial)
    # bench.benchmark(lambda v1, v2: v1/(4*np.pi*v2**2), func_name="II.3.24", trials=trial)
    # bench.benchmark(lambda v1, v2, v3: v1/(4*np.pi*v2*v3), func_name="II.4.23", trials=trial)
    # bench.benchmark(lambda v1, v2, v3, v4: 1/(4*np.pi*v1)*v2*np.cos(v3)/v4**2, func_name="II.6.11", trials=trial)
    # bench.benchmark(lambda v1, v2, v3, v4, v5, v6: v2/(4*np.pi*v1)*3*v6/v3**5*np.sqrt(v4**2+v5**2), func_name="II.6.15a", trials=trial)
    # bench.benchmark(lambda v1, v2, v3, v4: v2/(4*np.pi*v1)*3*np.cos(v3)*np.sin(v3)/v4**3, func_name="II.6.15b", trials=trial)
    # bench.benchmark(lambda v1, v2, v3: 3/5*v1**2/(4*np.pi*v2*v3), func_name="II.8.7", trials=trial)
    # bench.benchmark(lambda v1, v2: v1*v2**2/2, func_name="II.8.31", trials=trial)
    # bench.benchmark(lambda v1, v2, v3: v1/v2*1/(1+v3), func_name="II.10.9", trials=trial)
    # bench.benchmark(lambda v1, v2, v3, v4, v5: v1*v2/(v3*(v4**2-v5**2)), func_name="II.11.3", trials=trial)
    # bench.benchmark(lambda v1, v2, v3, v4, v5, v6: v1*(1+v5*v6*np.cos(v4)/(v2*v3)), func_name="II.11.17", trials=trial)
    # bench.benchmark(lambda v1, v2, v3, v4, v5: v1*v2**2*v3/(3*v4*v5), func_name="II.11.20", trials=trial)
    # bench.benchmark(lambda v1, v2, v3, v4: v1*v2/(1-(v1*v2/3))*v3*v4, func_name="II.11.27", trials=trial)
    # bench.benchmark(lambda v1, v2: 1+v1*v2/(1-(v1*v2/3)), func_name="II.11.28", trials=trial)
    # bench.benchmark(lambda v1, v2, v3, v4: 1/(4*np.pi*v1*v2**2)*2*v3/v4, func_name="II.13.17", trials=trial)
    # bench.benchmark(lambda v1, v2, v3: v1/np.sqrt(1-v2**2/v3**2), func_name="II.13.23", trials=trial) # Gives NaNs
    # bench.benchmark(lambda v1, v2, v3: v1*v2/np.sqrt(1-v2**2/v3**2), func_name="II.13.34", trials=trial) # Gives NaNs
    # bench.benchmark(lambda v1, v2, v3: -v1*v2*np.cos(v3), func_name="II.15.4", trials=trial)
    # bench.benchmark(lambda v1, v2, v3: -v1*v2*np.cos(v3), func_name="II.15.5", trials=trial)
    # bench.benchmark(lambda v1, v2, v3, v4, v5: v1/(4*np.pi*v2*v3*(1-v4/v5)), func_name="II.21.32", trials=trial)
    # bench.benchmark(lambda v1, v2, v3: np.sqrt(v1**2/v2**2-np.pi**2/v3**2), func_name="II.24.17", trials=trial) # Gives NaNs
    # bench.benchmark(lambda v1, v2, v3: v1*v2*v3**2, func_name="II.27.16", trials=trial)
    # bench.benchmark(lambda v1, v2: v1*v2**2, func_name="II.27.18", trials=trial)
    # bench.benchmark(lambda v1, v2, v3: v1*v2/(2*np.pi*v3), func_name="II.34.2a", trials=trial)
    # bench.benchmark(lambda v1, v2, v3: v1*v2*v3/2, func_name="II.34.2", trials=trial)
    # bench.benchmark(lambda v1, v2, v3, v4: v1*v2*v3/(2*v4), func_name="II.34.11", trials=trial)
    # bench.benchmark(lambda v1, v2, v3: v1*v2/(4*np.pi*v3), func_name="II.34.29a", trials=trial)
    # bench.benchmark(lambda v1, v2, v3, v4, v5: v1*v4*v5*v3/(v2/(2*np.pi)), func_name="II.34.29b", trials=trial)
    # bench.benchmark(lambda v1, v2, v3, v4, v5: v1/(np.exp(v4*v5/(v2*v3))+np.exp(-v4*v5/(v2*v3))), func_name="II.35.18", trials=trial)
    # bench.benchmark(lambda v1, v2, v3, v4, v5: v1*v2*np.tanh(v2*v3/(v4*v5)), func_name="II.35.21", trials=trial)
    # bench.benchmark(lambda v1, v2, v3, v4, v5, v6, v7, v8: v1*v2/(v3*v4)+(v1*v5)/(v6*v7**2*v3*v4)*v8, func_name="II.36.38", trials=trial)
    # bench.benchmark(lambda v1, v2, v3: v1*(1+v3)*v2, func_name="II.37.1", trials=trial)
    # bench.benchmark(lambda v1, v2, v3, v4: v1*v2*v4/v3, func_name="II.38.3", trials=trial)
    # bench.benchmark(lambda v1, v2: v1/(2*(1+v2)), func_name="II.38.14", trials=trial)
    # bench.benchmark(lambda v1, v2, v3, v4: 1/(np.exp((v1/(2*np.pi))*v2/(v3*v4))-1), func_name="III.4.32", trials=trial)
    # bench.benchmark(lambda v1, v2, v3, v4: (v1/(2*np.pi))*v2/(np.exp((v1/(2*np.pi))*v2/(v3*v4))-1), func_name="III.4.33", trials=trial)
    # bench.benchmark(lambda v1, v2, v3: 2*v1*v2/(v3/(2*np.pi)), func_name="III.7.38", trials=trial)
    # bench.benchmark(lambda v1, v2, v3: np.sin(v1*v2/(v3/(2*np.pi)))**2, func_name="III.8.54", trials=trial)
    # bench.benchmark(lambda v1, v2, v3, v4, v5, v6: (v1*v2*v3/(v4/(2*np.pi)))*np.sin((v5-v6)*v3/2)**2/((v5-v6)*v3/2)**2, func_name="III.9.52", trials=trial)
    # bench.benchmark(lambda v1, v2, v3: v1*np.sqrt(v2**2+v3**2+Bz**2), func_name="III.10.19", trials=trial)
    # bench.benchmark(lambda v1, v2: v1*(v2/(2*np.pi)), func_name="III.12.43", trials=trial)
    # bench.benchmark(lambda v1, v2, v3, v4: 2*v1*v2**2*v3/(v4/(2*np.pi)), func_name="III.13.18", trials=trial)
    # bench.benchmark(lambda v1, v2, v3, v4, v5: v1*(np.exp(v2*v3/(v4*v5))-1), func_name="III.14.14", trials=trial) # Gives NaNs
    # bench.benchmark(lambda v1, v2, v3: 2*v1*(1-np.cos(v2*v3)), func_name="III.15.12", trials=trial)
    # bench.benchmark(lambda v1, v2, v3: (v1/(2*np.pi))**2/(2*v2*v3**2), func_name="III.15.14", trials=trial)
    # bench.benchmark(lambda v1, v2, v3: 2*np.pi*v1/(v2*v3), func_name="III.15.27", trials=trial)
    # bench.benchmark(lambda v1, v2, v3: v1*(1+v2*np.cos(v3)), func_name="III.17.37", trials=trial)
    # bench.benchmark(lambda v1, v2, v3, v4: -v1*v2**4/(2*(4*np.pi*epsilon)**2*(v3/(2*np.pi))**2)*(1/v4**2), func_name="III.19.51", trials=trial)
    # bench.benchmark(lambda v1, v2, v3, v4: -v1*v2*v3/v4, func_name="III.21.20", trials=trial)


    # import re
    # import pandas as pd
    
    # feynman_df = pd.read_csv('FeynmanEquations.csv')
    # # Function to extract variables and create lambda functions
    # def create_benchmark_lambda(formula, variables):
    #     # Extract variable names and replace in formula
    #     var_list = []
    #     for i, var in enumerate(variables):
    #         if pd.notna(var):
    #             var_name = f'v{i+1}'
    #             formula = re.sub(rf'\b{var}\b', var_name, formula)
    #             var_list.append(var_name)
        
    #     # Create lambda function string
    #     lambda_str = f"lambda {', '.join(var_list)}: {formula}"
    #     return lambda_str

    # # Extract and create benchmark functions
    # feynman_benchmarks = []
    # for index, row in feynman_df.iterrows():
    #     if pd.isna(row['# variables']):
    #         continue  # Skip rows where '# variables' is NaN

    #     formula = row['Formula']
    #     num_vars = int(row['# variables']) if not pd.isna(row['# variables']) else 0
    #     variables = [row[f'v{i+1}_name'] for i in range(num_vars)]
    #     func_name = row['Filename']
    #     lambda_str = create_benchmark_lambda(formula, variables)
    #     feynman_benchmarks.append((func_name, lambda_str))

    # # Print the benchmark functions
    # for func_name, lambda_str in feynman_benchmarks:
    #     print(f'bench.benchmark({lambda_str}, func_name="{func_name}", trials=trial)')