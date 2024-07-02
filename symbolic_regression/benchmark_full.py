import pickle
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import pretty_print, functions
from utils.symbolic_network import SymbolicNetL0, SymbolicNet
from inspect import signature
import time
import argparse
from tqdm import trange
from gplearn.genetic import SymbolicRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

from KAN.KAN import *
from KAN.utils import evaluate_complexity

from gflownet.gflownet import GFlowNet
from actions import Action, ExpressionTree
from policy import RNNForwardPolicy, CanonicalBackwardPolicy
from gflownet.utils import *
from gflownet.env.sr_env import SRTree
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, TransformerMixin
import re




torch.set_warn_always(False)
device = torch.device("mps")
N_TRAIN = 10_000
N_VAL = 1000
DOMAIN = (-1, 1)
N_TEST = 1000
DOMAIN_TEST = (-1, 1)
NOISE_SD = 0
var_names = ["x", "y", "z", "a", "b", "c", "d", "e", "f"]

init_sd_first = 0.1
init_sd_last = 1.0
init_sd_middle = 0.5

def generate_data(func, N, range_min=DOMAIN[0], range_max=DOMAIN[1]):
    """Generates datasets."""
    x_dim = len(signature(func).parameters)
    x = (range_max - range_min) * torch.rand([N, x_dim]) + range_min
    
    y = []
    valid_x = []
    for x_i in x:
        try:
            y_value = func(*x_i)
            if not np.isnan(y_value):
                y.append([y_value])
                valid_x.append(x_i)
            # else:
                # print(f"NaN produced by func for input: {x_i}")
        except Exception as e:
            print(f"Error in func for input {x_i}: {e}")

    y = torch.tensor(y)
    valid_x = torch.stack(valid_x)
    return valid_x, y

class Benchmark:
    def __init__(self, results_dir, n_layers=2, reg_weight=5e-3, learning_rate=1e-2,
                 n_epochs1=10001, n_epochs2=10001, num_epochs_gfn=1000):
        self.activation_funcs = [
            *[functions.Constant()] * 2,
            *[functions.Identity()] * 4,
            *[functions.Square()] * 4,
            *[functions.Sin()] * 2,
            *[functions.Exp()] * 2,
            *[functions.Sigmoid()] * 2,
            *[functions.Product(1.0)] * 2,
        ]
        self.n_layers = n_layers
        self.reg_weight = reg_weight
        self.learning_rate = learning_rate
        self.summary_step = 10
        self.n_epochs1 = n_epochs1
        self.n_epochs2 = n_epochs2
        self.num_epochs_gfn = num_epochs_gfn
        self.func_name = ''

        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        self.results_dir = results_dir

        result = {
            "learning_rate": self.learning_rate,
            "summary_step": self.summary_step,
            "n_epochs1": self.n_epochs1,
            "n_epochs2": self.n_epochs2,
            "num_epochs_gfn": self.num_epochs_gfn,
            "activation_funcs_name": [func.name for func in self.activation_funcs],
            "n_layers": self.n_layers,
            "reg_weight": self.reg_weight,
        }
        with open(os.path.join(self.results_dir, 'params.pickle'), "wb+") as f:
            pickle.dump(result, f)
            
    
    def benchmark(self, func, func_name, trials):
        self.func_name = func_name
        print("Starting benchmark for function:\t%s" % func_name)
        print("==============================================")
        print(f"Complexity of function: {evaluate_complexity(func_name)}")
        
        func_dir = os.path.join(self.results_dir, func_name)
        if not os.path.exists(func_dir):
            os.makedirs(func_dir)

        expr_list, error_test_list = self.train(func, func_name, trials, func_dir)

        error_expr_sorted = sorted(zip(error_test_list, expr_list))
        error_test_sorted = [x for x, _ in error_expr_sorted]
        expr_list_sorted = [x for _, x in error_expr_sorted]

        # with open(os.path.join(self.results_dir, 'eq_summary.txt'), 'a') as fi:
        #     fi.write("\n{}\n".format(func_name))
        #     for i in range(trials):
        #         fi.write("[%f]\t\t%s\n" % (error_test_sorted[i], str(expr_list_sorted[i])))

    def train(self, func, func_name='', trials=1, func_dir='results/test'):
        
        # Function to extract and normalize terms from the string
        def extract_terms(expression):
            # Replace variables with indices (e.g., x0, x1) with a common identifier (e.g., x)
            expression = re.sub(r'[a-zA-Z_]\d*', lambda m: m.group(0)[0], expression)
            # Find all occurrences of functions and variables with optional powers
            terms = re.findall(r'[a-zA-Z_]\w*\^?\d*', expression)
            # Remove duplicates and sort the list for consistency
            unique_terms = sorted(set(terms))
            return unique_terms

    
    
        # use_cuda = torch.cuda.is_available()
        # device = torch.device("cuda" if use_cuda else "cpu")
        # print("Use cuda:", use_cuda, "Device:", device)

        x, y = generate_data(func, N_TRAIN)
        data, target = x.to(device), y.to(device)
        x_test, y_test = generate_data(func, N_TEST, range_min=DOMAIN_TEST[0], range_max=DOMAIN_TEST[1])
        test_data, test_target = x_test.to(device), y_test.to(device)

        x_dim = len(signature(func).parameters)

        width = len(self.activation_funcs)
        n_double = functions.count_double(self.activation_funcs)

        eq_list = []
        error_test_final = []

        kan_mse_list, kan_rmse_list, kan_r2_list, kan_time_list = [], [], [], []
        gp_mse_list, gp_rmse_list, gp_r2_list, gp_time_list = [], [], [], []
        eql_mse_list, eql_rmse_list, eql_r2_list, eql_time_list = [], [], [], []
        gfn_results = {ptype: {'mse': [], 'rmse': [], 'r2': [], 'time': []} for ptype in ["rnn", "gru", "lstm", "kan"]}
        
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters())
        
        pbar = trange(trials)
        print()
        print("============ Symbolic Regression ======= Training on function " + func_name+ "======")
        lib = extract_terms(func_name)
        # print("Library: ", lib)
        for trial in pbar:
            net = SymbolicNetL0(self.n_layers,
                                funcs=self.activation_funcs,
                                initial_weights=[
                                    torch.fmod(torch.normal(0, init_sd_first, size=(x_dim, width + n_double)), 2),
                                    torch.fmod(torch.normal(0, init_sd_middle, size=(width, width + n_double)), 2),
                                    torch.fmod(torch.normal(0, init_sd_middle, size=(width, width + n_double)), 2),
                                    torch.fmod(torch.normal(0, init_sd_last, size=(width, 1)), 2)
                                ]).to(device)

            
            kan = KAN(width=[x_dim,1,1], grid=2, k=4, seed=trial)
            
            print("parameters before: ",count_parameters(kan))
            dataset = {'train_input': data, 'train_label': target, 'test_input': test_data, 'test_label': test_target}
            print()
            zero_th = 0
            kan_start = time.time()
            kan_results = kan.train(dataset, opt="LBFGS", steps=self.n_epochs1 + self.n_epochs2 + zero_th, lr=self.learning_rate)
            print("parameters after: ",count_parameters(kan))
            kan_end = time.time()
            kan_mse = np.array(kan_results['test_loss'])[-1].item()
            kan_rmse = np.sqrt(kan_mse)
            kan_r2 = kan_results['test_R^2'][-1]
            kan_time = kan_end - kan_start

            kan_mse_list.append(kan_mse)
            kan_rmse_list.append(kan_rmse)
            kan_r2_list.append(kan_r2)
            kan_time_list.append(kan_time)
            
            
            # lib = ['x','x^2', 'x^4', 'sin', 'sqrt', 'log']
            # kan.prune()
            # kan.auto_symbolic(lib=lib, verbose=0)
            # formula = kan.symbolic_formula()[0]
            # print(formula,'\n')
            # print(f"KAN equation complexity: {evaluate_complexity(formula[0])}")

            # print('\n')
            # print(f"KAN MSE {kan_mse}")
            print(f"KAN RMSE {kan_rmse}")
            # print(f"KAN R^2: {kan_r2}")
            # print(f"KAN time: {kan_time}")

            gp_start = time.time()
            est_gp = SymbolicRegressor(population_size=5000,
                                    generations=20, random_state=trial)
            est_gp.fit(data.cpu().numpy(), target.cpu().numpy().ravel())
            y_gp = est_gp.predict(test_data.cpu().numpy())

            # print(est_gp.get_params(deep=True))
            test_target_1d = test_target.cpu().numpy().ravel()
            y_gp_1d = y_gp.ravel()
            gp_end = time.time()
            gp_mse = np.mean((test_target_1d - y_gp_1d)**2)
            gp_rmse = np.sqrt(gp_mse)
            gp_r2 = r2_score(test_target_1d, y_gp_1d)
            gp_time = gp_end - gp_start

            gp_mse_list.append(gp_mse)
            gp_rmse_list.append(gp_rmse)
            gp_r2_list.append(gp_r2)
            gp_time_list.append(gp_time)
            print(est_gp._program)
            print(f"GPLearn equation complexity: {evaluate_complexity(est_gp._program)}")
            # print()
            # print(f"GPLearn MSE: {gp_mse}")
            print(f"GPLearn RMSE: {gp_rmse}")
            # print(f"GPLearn R^2: {gp_r2}")
            # print(f"GPLearn time: {gp_time}")

            loss_val = np.nan
            while np.isnan(loss_val):
                criterion = nn.MSELoss()
                optimizer = optim.Adam(net.parameters(), lr=self.learning_rate * 10, weight_decay=0.005)
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=3, verbose=True)

                t0 = time.time()

                for epoch in range(self.n_epochs1 + self.n_epochs2 + zero_th):
                    optimizer.zero_grad()
                    outputs = net(data.to(device))
                    mse_loss = criterion(outputs, target.to(device))
                    reg_loss = net.get_loss()
                    loss = mse_loss + self.reg_weight * reg_loss
                    loss.backward()
                    optimizer.step()

                    if epoch % self.summary_step == 0:
                        error_val = mse_loss.item()
                        reg_val = reg_loss.item()
                        loss_val = loss.item()

                        with torch.no_grad():
                            test_outputs = net(test_data)
                            test_loss = F.mse_loss(test_outputs, test_target)
                            test_r2 = r2_score(test_target.cpu().numpy(), test_outputs.cpu().numpy())
                            error_test_val = torch.sqrt(test_loss).item()

                        if np.isnan(loss_val):
                            break

                    if epoch == 2000:
                        scheduler.step(loss)
                    elif epoch == self.n_epochs1 + 2000:
                        scheduler.step(loss)

                eql_mse_list.append(error_test_val)
                eql_rmse_list.append(np.sqrt(error_test_val))
                eql_r2_list.append(test_r2)
                t1 = time.time()
                eql_time_list.append(t1 - t0)
                
                # # Print the expressions
                # with torch.no_grad():
                #     weights = net.get_weights()
                #     expr = pretty_print.network(weights, self.activation_funcs, var_names[:x_dim])
                #     print("EQL equation: ", expr)
                #     print("EQL equation complexity: ", evaluate_complexity(expr))

                # print()
                # print(f"EQL MSE {error_test_val}")  
                print(f"EQL RMSE {np.sqrt(error_test_val)}")
                # print(f"EQL R^2 {test_r2}")
                # print(f"EQL time: ", t1 - t0, "\n")

            print("===== GFN-SR POLICIES ========= Training on function " + func_name+ "======")
            print()
            
            def train_gfn_sr(batch_size, num_epochs, X, y, device, policy_type):
                torch.manual_seed(trial)
                action = Action(X.shape[1])
                env = SRTree(X, y.view(-1), action_space=action, max_depth=3, loss="dynamic")

                if policy_type == "rnn":
                    forward_policy = RNNForwardPolicy(batch_size, 128 , env.num_actions, 3,  model="rnn", device=device)
                elif policy_type == "gru":
                    forward_policy = RNNForwardPolicy(batch_size, 128, env.num_actions, 3, model="gru", device=device)
                elif policy_type == "lstm":
                    forward_policy = RNNForwardPolicy(batch_size, 128, env.num_actions, 3, model="lstm", device=device)
                elif policy_type == "kan":
                    forward_policy = KAN(width=[7, 1, 1, 1, 1, 1, 1,  1], grid=100, k=10, seed=trial)
                else:
                    raise ValueError("Unsupported policy type")

                backward_policy = CanonicalBackwardPolicy(env.num_actions)
                model = GFlowNet(forward_policy, backward_policy, env)
                params = [param for param in model.parameters() if param.requires_grad]
                # print(f"GFN-SR {policy_type} parameters: {sum(p.numel() for p in params)}")
                opt = torch.optim.Adam(params, lr=1e-3)

                for i in range(num_epochs):
                    s0 = env.get_initial_states(batch_size)
                    s, log = model.sample_states(s0)
                    loss = trajectory_balance_loss(log.total_flow, log.rewards, log.fwd_probs)
                    loss.backward()
                    opt.step()
                    opt.zero_grad()

                return model, env

            policy_types = [
                            # "rnn",
                            # "gru", 
                            "lstm",
                            "kan"
                            ]
            for policy_type in policy_types:
                policy_start = time.time()
                gfn_model, gfn_env = train_gfn_sr(batch_size=256, num_epochs=self.num_epochs_gfn, X=data, y=target, device=device, policy_type=policy_type)

                eval_s0 = gfn_env.get_initial_states(len(test_data))
                eval_s, log = gfn_model.sample_states(eval_s0)

                expressions = [ExpressionTree(encoding, gfn_env.action_space.action_fns, gfn_env.action_space.action_arities, gfn_env.action_space.action_names) for encoding in eval_s]

                eval_mse = []
                y_pred_list = []
                best_expression = [None, float('inf')]
                for expr_tree in expressions:
                    try:
                        y_pred = expr_tree(test_data)
                        y_pred_list.append(y_pred)
                        mse = F.mse_loss(y_pred, test_target.view(-1))
                        eval_mse.append(mse.item())
                        if mse < best_expression[1]:
                            best_expression = [expr_tree, mse]
                    except Exception as e:
                        print(f"Error in evaluating expression tree: {e}")
                        eval_mse.append(float('inf'))

                eval_mse = torch.tensor(eval_mse)
                eval_mse = eval_mse[torch.isfinite(eval_mse)]
                avg_mse = torch.median(eval_mse).item()
                y_pred_list = [yp for yp in y_pred_list if torch.isfinite(yp).all()]
                if y_pred_list:
                    y_pred = torch.stack(y_pred_list).median(dim=0).values
                    r2_score_gfn = r2_score(test_target.view(-1).detach().cpu().numpy(), y_pred.detach().cpu().numpy())
                else:
                    r2_score_gfn = float('-inf')
                policy_end = time.time()
                gfn_results[policy_type]['mse'].append(avg_mse)
                gfn_results[policy_type]['rmse'].append(np.sqrt(avg_mse))
                gfn_results[policy_type]['r2'].append(r2_score_gfn)
                gfn_results[policy_type]['time'].append(policy_end - policy_start)
                gfn_results[policy_type]['expression'] = best_expression[0].__str__()
                
               
                # print(f"GFN-SR {policy_type} MSE: {avg_mse}")
                print(f"GFN-SR {policy_type} RMSE: {np.sqrt(avg_mse)}")
                # print(f"GFN-SR {policy_type} R^2: {r2_score_gfn}")
                # print(f"GFN-SR Time {policy_type}: {policy_end - policy_start}")
                # print()

        # Calculate and print the averages and standard deviations
        print(f"\nSummary of results after all trials for {self.func_name}:")

        def print_summary(name, mse_list, rmse_list, r2_list, time_list):
            print(f"{name} MSE: {np.mean(mse_list):.4f} ± {np.std(mse_list):.4f}")
            print(f"{name} RMSE: {np.mean(rmse_list):.4f} ± {np.std(rmse_list):.4f}")
            print(f"{name} R^2: {np.mean(r2_list):.4f} ± {np.std(r2_list):.4f}")
            print(f"{name} Time: {np.mean(time_list):.4f} ± {np.std(time_list):.4f}")
            
        print_summary("KAN", kan_mse_list, kan_rmse_list, kan_r2_list, kan_time_list)
        print_summary("GPLearn", gp_mse_list, gp_rmse_list, gp_r2_list, gp_time_list)
        print_summary("EQL", eql_mse_list, eql_rmse_list, eql_r2_list, eql_time_list)

        for policy_type in policy_types:
            print_summary(f"GFN-SR {policy_type}", 
                        gfn_results[policy_type]['mse'], 
                        gfn_results[policy_type]['rmse'], 
                        gfn_results[policy_type]['r2'], 
                        gfn_results[policy_type]['time']),
            print(f"{policy_type} Expression: {gfn_results[policy_type]['expression']}\n")


        return eq_list, error_test_final


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the EQL network.")
    parser.add_argument("--results-dir", type=str, default='results/benchmark/test')
    parser.add_argument("--n-layers", type=int, default=3, help="Number of hidden layers, L")
    parser.add_argument("--reg-weight", type=float, default=5e-3, help='Regularization weight, lambda')
    parser.add_argument('--learning-rate', type=float, default=1e-2, help='Base learning rate for training')
    parser.add_argument("--n-epochs1", type=int, default=1000, help="Number of epochs to train the first stage")
    parser.add_argument("--n-epochs2", type=int, default=1000, help="Number of epochs to train the second stage, after freezing weights.")
    parser.add_argument("--num-epochs-gfn", type=int, default=2000, help="Number of epochs to train GFN-SR")

    args = parser.parse_args()
    kwargs = vars(args)
    print(kwargs)

    if not os.path.exists(kwargs['results_dir']):
        os.makedirs(kwargs['results_dir'])
    with open(os.path.join(kwargs['results_dir'], 'args.txt'), 'a') as meta:
        import json
        meta.write(json.dumps(kwargs))

    bench = Benchmark(**kwargs)
    trial = 3
    
    
    # Penguin equations
    # 1 
    # bench.benchmark(lambda x, y, z: ((x**4 / y **3) - (np.sin(z**4) + x**2))**z, func_name="((x^4 / y^3) - (sin(z^4) + x^2))^z", trials=trial)
    # 2
    # bench.benchmark(lambda x, y, z: (x**2 + np.log(y**3) + np.sin(z)**2)**(x/y), func_name="(x^2 + log(y^3) + sin(z)^2)^(x/y)", trials=trial)
    # 3
    # bench.benchmark(lambda x, y: (np.pi**x - np.pi**y)**np.pi , func_name="(pi^x - pi^y)^pi", trials=trial)
    # 4
    # bench.benchmark(lambda x, y: np.sqrt(x**4*y**4) - np.sqrt(y**3/x**4)**(np.sin(np.pi**2)) , func_name="sqrt(x^4*y^4) - sqrt(y^3/x^4)^(sin(pi^2)", trials=trial)
    # 5 
    # bench.benchmark(lambda x, y, z, a: np.tan(x + y*z) / np.log(a), func_name="tan(x + y*z) / log(a)", trials=trial)


    
    
    # bench.benchmark(lambda x, y, z: (x**2 + y**(1/3) - x**(1/4) + y**5)**(y/x), func_name="(x^2 + y^(1/3) - x^(1/4) + y^5)^(y/x)", trials=trial)
    # bench.benchmark(lambda x: x, func_name="x", trials=trial)
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
    # 1
    # bench.benchmark(lambda x: x**3 + x**2 + x, func_name="x^3+x^2+x", trials=trial)
    # 2
    # bench.benchmark(lambda x:  x**4 + x**3 + x**2 + x, func_name="x^4+x^3+x^2+x", trials=trial)
    # 3
    # bench.benchmark(lambda x: x**5 + x**4 + x**3 + x**2 + x, func_name="x^5+x^4+x^3+x^2+x", trials=trial)
    # 4
    # bench.benchmark(lambda x: x**6 + x**5 + x**4 + x**3 + x**2 + x, func_name="x^6+x^5+x^4+x^3+x^2+x", trials=trial)
    # 5
    # bench.benchmark(lambda x: np.sin(x**2) * np.cos(x) - 1, func_name="sin(x^2)*cos(x)-1", trials=trial)
    # 6
    # bench.benchmark(lambda x: np.sin(x) + np.sin(x + x**2), func_name="sin(x)+sin(x+x^2)", trials=trial)
    # # 7
    # bench.benchmark(lambda x: np.log(x + 1) + np.log(x**2 + 1), func_name="log(x+1)+log(x^2+1)", trials=trial)
    # # 8
    # bench.benchmark(lambda x: np.sqrt(x), func_name="sqrt(x)", trials=trial) 
    # # 9 
    # bench.benchmark(lambda x0, x1: np.sin(x0) * np.sin(x1**2), func_name="sin(x0)*cos(x1^2)", trials=trial)
    # # 10
    # bench.benchmark(lambda x0, x1: 2 * np.sin(x0) * np.cos(x1), func_name="2*sin(x0)*cos(x1)", trials=trial)
    # # 11
    # bench.benchmark(lambda x, y: x**y, func_name="x^y", trials=trial) # NEW!
    # # 12
    # bench.benchmark(lambda x, y: x**4 - x**3 + (y**2/2) - y, func_name="x^4 - x^3 + (y^2/2) - y", trials=trial) # NEW!
    
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
    # bench.benchmark(lambda v1, v2, v3: v1*v2*np.sin(v3), func_name="I.18.12", trials=trial)
    # bench.benchmark(lambda v1, v2, v3, v4: v1*v2*v3*np.sin(v4), func_name="I.18.14", trials=trial)
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
    # bench.benchmark(lambda v1, v2, v3, v4: 4*np.pi*v4*(v3/(2*np.pi))**2/(v1*v2**2), func_name="I.38.12", trials=trial)
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
    bench.benchmark(lambda v1, v2, v3, v4, v5, v6: (v1*v2*v3/(v4/(2*np.pi)))*np.sin((v5-v6)*v3/2)**2/((v5-v6)*v3/2)**2, func_name="III.9.52", trials=trial)
    # bench.benchmark(lambda v1, v2, v3: v1*np.sqrt(v2**2+v3**2+Bz**2), func_name="III.10.19", trials=trial)
    # bench.benchmark(lambda v1, v2: v1*(v2/(2*np.pi)), func_name="III.12.43", trials=trial)
    # bench.benchmark(lambda v1, v2, v3, v4: 2*v1*v2**2*v3/(v4/(2*np.pi)), func_name="III.13.18", trials=trial)
    # bench.benchmark(lambda v1, v2, v3, v4, v5: v1*(np.exp(v2*v3/(v4*v5))-1), func_name="III.14.14", trials=trial) # Gives NaNs
    # bench.benchmark(lambda v1, v2, v3: 2*v1*(1-np.cos(v2*v3)), func_name="III.15.12", trials=trial)
    # bench.benchmark(lambda v1, v2, v3: (v1/(2*np.pi))**2/(2*v2*v3**2), func_name="III.15.14", trials=trial)
    # bench.benchmark(lambda v1, v2, v3: 2*np.pi*v1/(v2*v3), func_name="III.15.27", trials=trial)
    # bench.benchmark(lambda v1, v2, v3: v1*(1+v2*np.cos(v3)), func_name="III.17.37", trials=trial)
    # bench.benchmark(lambda v1, v2, v3, v4, v5: -v1*v2**4/(2*(4*np.pi*v5)**2*(v3/(2*np.pi))**2)*(1/v4**2), func_name="III.19.51", trials=trial)
    # bench.benchmark(lambda v1, v2, v3, v4: -v1*v2*v3/v4, func_name="III.21.20", trials=trial)
