
import numpy as np
import yfinance as yf
import random
import matplotlib.pyplot as plt
import math

# Import stock historical data, from 2020-12-01 to 2022-12-01
# The order is APPLE, MICROSOFT, GOOGLE, TSMC, NVIDIA, ASML, ADOBE, QUALCOMM, AMD, INTEL
stock_set = yf.download("aapl msft goog tsm nvda asml adbe qcom amd intc",
                        start="2015-12-01", end="2022-12-01")

data = stock_set['Close']

# Calculate annualized return
rets = np.log(data / data.shift(1)).dropna()
annual_rets = rets.mean() * 252

annual_cov = rets.cov() * 252

ten_year_bond = yf.download("^TNX", start="2015-12-01", end="222-12-01")

r_ten_year_bond = ten_year_bond['Close']

rf = np.mean(np.log(r_ten_year_bond / r_ten_year_bond.shift(1)).dropna())



def sharpe_ratio(weights):
    # Calculating the rate of return
    a_rets = np.sum(rets.mean() * weights) * 252
    # var
    var = np.dot(weights.T, np.dot(rets.cov() * 252, weights))
    # volatility
    vol = math.sqrt(var)
    # sharpe ratio
    sharpe = (a_rets - rf) / vol
    return sharpe


# APSO
class PSO:
    def __init__(self, dimension, time, size, low, up, v_low, v_high):
        # Initialisation
        self.dimension = dimension  # dimensions
        self.time = time  # Generations of iterations
        self.size = size  # Population size
        self.bound = []  # Boundary ranges of variables
        self.bound.append(low)
        self.bound.append(up)
        self.v_low = v_low
        self.v_high = v_high
        self.x = np.zeros((self.size, self.dimension))  # Position of all particles
        self.v = np.zeros((self.size, self.dimension))  # Velocity of all particles
        self.p_best = np.zeros((self.size, self.dimension))  # Optimal position of each particle
        self.g_best = np.zeros((1, self.dimension))[0]  # Global optimal position

        # Initialize the initial global optimal solution for generation 0
        temp = -1000000
        for i in range(self.size):
            for j in range(self.dimension):
                self.x[i][j] = random.uniform(self.bound[0][j], self.bound[1][j])
                self.v[i][j] = random.uniform(self.v_low, self.v_high)
            self.p_best[i] = self.x[i]  # Storage of optimal individuals
            fit = self.fitness(self.p_best[i])
            # Make changes
            if fit > temp:
                self.g_best = self.p_best[i]
                temp = fit

    def fitness(self, x):
        """
        Calculation of individual adaptation values
        """
        return sharpe_ratio(x)

    def update(self, size):
        c1 = 1.49618  # Learning Factor
        c2 = 1.49618
        w = 0.7298  # Self-weighting factor
        for i in range(size):
            # Update speed (core formula)
            self.v[i] = w * self.v[i] + c1 * random.uniform(0, 1) * (
                    self.p_best[i] - self.x[i]) + c2 * random.uniform(0, 1) * (self.g_best - self.x[i])
            # Speed limit
            for j in range(self.dimension):
                if self.v[i][j] < self.v_low:
                    self.v[i][j] = self.v_low
                if self.v[i][j] > self.v_high:
                    self.v[i][j] = self.v_high

            # Update Location
            self.x[i] = self.x[i] + self.v[i]

            # Location restrictions
            for j in range(self.dimension):
                if self.x[i][j] < self.bound[0][j]:
                    self.x[i][j] = self.bound[0][j]
                if self.x[i][j] > self.bound[1][j]:
                    self.x[i][j] = self.bound[1][j]
            # Update p_best and g_best
            if self.fitness(self.x[i]) > self.fitness(self.p_best[i]):
                self.p_best[i] = self.x[i]
            if self.fitness(self.x[i]) > self.fitness(self.g_best):
                self.g_best = self.x[i]
            # Forcing the sum of the weights to be 1
            if np.sum(self.x[i]) != 1:
                self.x[i] = self.x[i] / np.sum(self.x[i])



    def pso(self):
        best = []
        self.final_best = np.array([0.1] * 10)
        for gen in range(self.time):
            self.update(self.size)
            if self.fitness(self.g_best) > self.fitness(self.final_best):
                self.final_best = self.g_best.copy()
            print('current best position：{}'.format(self.final_best))
            temp = (self.fitness(self.final_best))
            print('current best sharpe ratio：{}'.format(temp))
            port_returns = np.sum(rets.mean() * self.final_best) * 252
            print('current best port_return:{}', port_returns)
            print(np.sum(self.final_best))
            best.append(temp)
        t = [i for i in range(self.time)]
        plt.figure(1)
        plt.plot(t, best, color='red', marker='.', ms=15)
        plt.rcParams['axes.unicode_minus'] = False
        plt.margins(0)
        plt.xlabel(u"times")  
        plt.ylabel(u"sharpe ratio") 
        plt.title(u"process") 
        plt.show()

        name_list = ['aapl', 'msft', 'goog', 'tsm', 'nvda', 'asml', 'adbe', 'qcom', 'amd', 'intc']
        plt.figure(2)
        plt.bar(range(len(self.final_best)), self.final_best, tick_label=name_list)
        plt.ylabel(u"weight")
        plt.title(u"Optimal asset allocation")
        plt.show()


if __name__ == '__main__':
    time = 100
    size = 50
    dimension = 10
    v_low = -0.1
    v_high = 0.1
    low = [0] * 10
    up = [0.5] * 10
    pso = PSO(dimension, time, size, low, up, v_low, v_high)
    pso.pso()

