from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import pandas as pd
from WP32_utils import *
import os
import time
import warnings
from time import time

## Using fixed random seed = 100 to duplicate
# %%
np.random.seed(100)
random_state = 5
# %% md
# friction coefficients μ
# %%
from cmath import nan

μ = 1
Lx = 0.2775
Ly = 0.26
filename = os.path.join('Path Data', 'all_data.csv')
df = pd.read_csv(filename)
v, w, payload, unit_weight, m = df['v1'].values, df['w1'].values, df['payload'].values, df['unit weight'].values, df[
    'm'].values
M = payload * unit_weight + m  # df['payload']*df['unit weight'] + df['m']#all weight
r = np.array([vv / ww if ww != 0 else nan for vv, ww in zip(v, w)])
# P_ul = μ*M*g*v
P_ul = np.array([μ * MM * 9.8 * vv if ww == 0 else 0 for ww, vv, MM in zip(w, v, M)])
wheel_v1, wheel_v2 = np.array(
    [np.hypot((rr - Lx), Ly) * ww if ww != 0 else vv for ww, vv, rr in zip(w, v, r)]), np.array(
    [np.hypot((rr - Lx), Ly) * ww if ww != 0 else vv for ww, vv, rr in zip(w, v, r)])
wheel_v3, wheel_v4 = np.array(
    [np.hypot((rr + Lx), Ly) * ww if ww != 0 else vv for ww, vv, rr in zip(w, v, r)]), np.array(
    [np.hypot((rr + Lx), Ly) * ww if ww != 0 else vv for ww, vv, rr in zip(w, v, r)]),
df['M'] = M
df['r'] = r
df['P_ul'] = P_ul
df['wheel_v1'], df['wheel_v2'], df['wheel_v3'], df['wheel_v4'] = wheel_v1, wheel_v2, wheel_v3, wheel_v4
P_ut = np.array([1 / 4 * μ * MM * 9.8 * (np.abs(wv1) + np.abs(wv2) + np.abs(wv3) + np.abs(wv4)) if ww != 0 else 0 for
                 ww, MM, wv1, wv2, wv3, wv4, in zip(w, M, wheel_v1, wheel_v2, wheel_v3, wheel_v4)])
df['P_ut'] = P_ut
df['P_useful'] = P_ut + P_ul
# add knowledge part
df['w2'] = df['w1'] * df['w1']
df['v2'] = df['v1'] * df['v1']
df['w3'] = df['w1'] * df['w1'] * df['w1']
df['v3'] = df['v1'] * df['v1'] * df['v1']
df['w4'] = df['w1'] * df['w1'] * df['w1'] * df['w1']
df['v4'] = df['v1'] * df['v1'] * df['v1'] * df['v1']
df
# %%

df[['P_useful', 'Pstable']].plot()
df.to_csv('results_all.csv')
df.to_json('Results/results.json', orient='split')
# %%
test_size = 0.2
num_hidden_layers = (120)
# %% md
# Use Hybrid modelling - model *
# %%
data_size_factor = 1
max_iters = [i for i in range(1000, 15000, 1000)]
c = ['v1', 'w1', 'payload', 'P_ul', 'P_ut']

warnings.simplefilter('ignore', UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
results = []
mape_best = 1
for max_iter in max_iters:
    r = []
    # balanced training
    X_train, y_train, X_test, y_test = [], [], [], []
    for pl in range(3):
        df_xy = df[df['payload'] == pl]
        X_part, y_part = df_xy[c].to_numpy(), df_xy['Pstable'].to_numpy().reshape(-1)
        X_train_part, X_test_part, y_train_part, y_test_part = train_test_split(X_part, y_part,
                                                                                random_state=random_state,
                                                                                test_size=test_size)
        # print(X_train_part.shape, X_test_part.shape, y_train_part.shape, y_test_part.shape)
        X_train.append(X_train_part)
        y_train.append(y_train_part)
        X_test.append(X_test_part)
        y_test.append(y_test_part)

    X_train = np.vstack(X_train)
    y_train = np.block(y_train)
    X_test = np.vstack(X_test)
    y_test = np.block(y_test)

    # apply size factor
    X_train = X_train[:int(data_size_factor * X_train.shape[0])]
    y_train = y_train[:int(data_size_factor * y_train.shape[0])]
    X_test = X_test[:int(data_size_factor * X_test.shape[0])]
    y_test = y_test[:int(data_size_factor * y_test.shape[0])]
    # train

    t0 = time()
    regr = MLPRegressor(random_state=1, hidden_layer_sizes=num_hidden_layers, max_iter=max_iter).fit(X_train, y_train)
    if max_iter == 4000:
        model_filename = 'model_star.pickle'
        print("saving=================")
        pickle.dump(regr, open(os.path.join("Results", model_filename), 'wb'))

    y_pred_train = regr.predict(X_train)
    y_pred_test = regr.predict(X_test)
    r.append(r2_score(y_test, y_pred_test))
    r.append(MAPE(y_test, y_pred_test))
    r.append(time() - t0)
    print("R2 = %.2f, MAPE = %.3f, time = %.2fs" % (r[0], r[1], r[2]))
    results.append(r)
df_results = pd.DataFrame(results, columns=['R2', 'MAPE', 'Time'])
df_results['MAPE'].plot()
# %%
X_test
# %%
