import numpy as np
import yfinance as yf
import random
import matplotlib.pyplot as plt
import math

stock_set = yf.download("aapl msft goog tsm nvda asml adbe qcom amd intc",
                        start="2015-12-01", end="2022-12-01")
data = stock_set['Close']

ten_year_bond = yf.download("^TNX", start="2015-12-01", end="2022-12-01")
r_ten_year_bond = ten_year_bond['Close']
rf = np.mean(np.log(r_ten_year_bond / r_ten_year_bond.shift(1)).dropna()) * 252

log_returns = np.log(data / data.shift(1))
noa = 10  # Number of financial instruments defined.
rets = np.log(data / data.shift(1))


def port_ret(weights):
    return np.sum(rets.mean() * weights) * 252


def port_vol(weights):
    return np.sqrt(np.dot(weights.T, np.dot(rets.cov() * 252, weights)))


port_returns = []
port_variance = []

for p in range(5000):
    weights = np.random.random(noa)
    weights /= np.sum(weights)
    port_returns.append(port_ret(weights))
    port_variance.append(port_vol(weights))

port_returns = np.array(port_returns)
port_variance = np.array(port_variance)

plt.figure(1, figsize=(10, 6))
plt.scatter(port_variance, port_returns, c=(port_returns - rf) / port_variance,
            marker='o', cmap='coolwarm')
plt.grid(True)
plt.xlabel('excepted volatility')
plt.ylabel('expected return')
plt.colorbar(label='Sharpe ratio')
plt.show()


def statistics(weights):
    weights = np.array(weights)
    port_returns = np.sum(rets.mean() * weights) * 252
    port_variance = np.sqrt(np.dot(weights.T, np.dot(rets.cov() * 252, weights)))
    return np.array([port_returns, port_variance, port_returns / port_variance])


import scipy.optimize as sco


def min_sharpe(weights):
    return -statistics(weights)[2]


cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

bnds = tuple((0, 1) for x in range(noa))

opts = sco.minimize(min_sharpe, noa * [1. / noa, ], method='SLSQP', bounds=bnds, constraints=cons)

opts['x'].round(3)

statistics(opts['x']).round(3)


def min_variance(weights):
    return statistics(weights)[1]


target_returns = np.linspace(0.1, 0.36, 50)
target_variance = []

for tar in target_returns:
    cons = ({'type': 'eq', 'fun': lambda x: statistics(x)[0] - tar}, {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    res = sco.minimize(min_variance, noa * [1. / noa, ], method='SLSQP', bounds=bnds, constraints=cons)
    target_variance.append(res['fun'])

target_variance = np.array(target_variance)

plt.figure(2, figsize=(10, 6))
plt.scatter(port_variance, port_returns, c=(port_returns - rf) / port_variance,
            marker='o', cmap='coolwarm')
plt.plot(target_variance, target_returns, 'b', lw=2.0)
plt.plot(statistics(opts['x'])[1], statistics(opts['x'])[0], 'r*', markersize=15.0)
plt.grid(True)
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.colorbar(label='Sharpe ratio')
plt.show()

plt.figure(3, figsize=(10, 6))
name_list = ['aapl', 'msft', 'goog', 'tsm', 'nvda', 'asml', 'adbe', 'qcom', 'amd', 'intc']
plt.bar(range(len(opts['x'].round(3))), opts['x'].round(3), tick_label=name_list)
plt.ylabel(u"weight")
plt.title(u"Optimal asset allocation(Markowitz)")
plt.show()