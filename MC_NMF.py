import math
import sys
import numpy as np
import pandas as pd
import os
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.preprocessing import normalize


class MarkovChain:
    def __init__(self, dt, names, t=10000):
        self.dt = dt
        self.num = names.shape[0]
        self.names = names
        self.t = t
        self.m = np.zeros((self.num, self.num))
        self.w = np.zeros((t + 1, 1, self.num))
        self.w[0][:][:] = 1 / self.w[0].shape[1]
        self.w_inf = 0

    def update(self, i):
        index_a, score_a, index_b, score_b = self.dt[i]
        index_a -= 1
        index_b -= 1
        self.m[index_a][index_a] += (score_a / (score_a + score_b))
        self.m[index_b][index_b] += (score_b / (score_a + score_b))
        self.m[index_a][index_b] += (score_b / (score_a + score_b))
        self.m[index_b][index_a] += (score_a / (score_a + score_b))
        if score_a > score_b:
            self.m[index_a][index_a] += 1
            self.m[index_b][index_a] += 1
        elif score_a < score_b:
            self.m[index_a][index_b] += 1
            self.m[index_b][index_b] += 1

    def construct(self):
        for i in range(self.dt.shape[0]):
            self.update(i)
        #  self.m = normalize(self.m, norm='l1')
        self.m = (self.m.T / (np.sum(self.m.T, axis=0) + 1e-16)).T
        for i in range(1, self.t + 1):
            self.w[i] = np.matmul(self.w[i - 1], self.m)

    def rank(self):
        t_list = [10, 100, 1000, 10000]
        for i in t_list:
            index = np.argsort(-np.squeeze(self.w[i]))[:25]
            print('t =', i)
            for j in range(index.shape[0]):
                print(j + 1, self.names[index[j]], self.w[i][0][index[j]], sep='\t')

    def convergence(self):
        values, vectors = np.linalg.eig(self.m.T)
        location = (-1e-6 < np.real(values - 1)) & (np.real(values - 1) < 1e-6)
        w_inf = vectors[:, location]
        # w_inf = np.real(np.sum(w_inf, axis=1))
        # w_inf = np.expand_dims(w_inf, axis=0)
        # w_inf = normalize(w_inf, norm='l1', axis=1)
        w_inf = np.real(w_inf[:, 0])
        w_inf = np.expand_dims(w_inf, axis=0)
        # w_inf = normalize(w_inf, norm='l1', axis=1)
        w_inf = w_inf / np.sum(w_inf)
        self.w_inf = w_inf
        dif = np.linalg.norm((np.squeeze(self.w)[:] - w_inf), ord=1, axis=1)

        plt.figure(figsize=(9, 9))
        plt.plot(dif[1:], color="blue", linewidth=4)
        plt.xlabel("t - 1", fontsize=20)
        plt.ylabel("Loss", fontsize=20)
        plt.title("L1-Convergence", fontsize=20)
        plt.show()


class NMF:
    def __init__(self, frq, names):
        self.m = sum(1 for _ in open(frq))
        self.n = sum(1 for _ in open(names))
        self.x = np.zeros((self.n, self.m))
        self.w = {}
        self.h = {}
        self.w_norm = {}
        self.h_norm = {}
        self.obj = {}

        fl = open(frq)
        line = fl.readline()
        line = line.strip('\n\ufeff')
        j = 0
        while line:
            pointer = -1
            idx, cnt = 0, 0
            indexing, counting = True, False
            while pointer < len(line) - 1:
                pointer += 1
                if line[pointer] == ':':
                    counting, indexing = True, False
                    continue
                elif line[pointer] == ',' or pointer == (len(line) - 1):
                    indexing, counting = True, False
                    idx -= 1
                    self.x[idx][j] = cnt
                    cnt, idx = 0, 0
                    continue
                if counting:
                    cnt = 10 * cnt + int(line[pointer])
                elif indexing:
                    idx = 10 * idx + int(line[pointer])

            line = fl.readline()
            line = line.strip('\n')
            j += 1
        fl.close()
        words_df = pd.read_csv(names, header=None)
        self.words = np.array(words_df)

    def iteration(self, rank=25, it=100):
        self.w[rank] = np.random.rand(it + 1, self.n, rank)  # (3012, 25)
        self.h[rank] = np.random.rand(it + 1, rank, self.m)  # (25, 8447)
        self.obj[rank] = np.zeros(it)
        for i in range(it):
            purple = self.x / (self.w[rank][i].dot(self.h[rank][i]) + 1e-16)
            self.h[rank][i+1] = self.h[rank][i] * (self.w[rank][i] / (np.sum(self.w[rank][i], axis=0) + 1e-16)).T.dot(purple)
            #  normalize(self.w[rank][i].T, norm='l1').dot(purple))
            purple = self.x / (self.w[rank][i].dot(self.h[rank][i+1]) + 1e-16)
            self.w[rank][i+1] = self.w[rank][i] * (purple.dot(self.h[rank][i+1].T / (np.sum(self.h[rank][i + 1].T, axis=0) + 1e-16)))
            #  normalize(self.h[rank][i+1].T, norm='l1', axis=0)))
            approximate = self.w[rank][i + 1].dot(self.h[rank][i + 1])
            self.obj[rank][i] = np.sum(self.x * np.log(1 / (approximate + 1e-16)) + approximate)
        a = np.sum(self.w[rank][it], axis=0)
        self.w_norm[rank] = self.w[rank][it] / (a + 1e-16)
        self.h_norm[rank] = (a * self.h[rank][it].T).T

    def show(self, rank=25):
        plt.figure(figsize=(9, 9))
        plt.plot(self.obj[rank], color="blue", linewidth=4)
        plt.xlabel("iterations - 1", fontsize=20)
        plt.ylabel("Value", fontsize=20)
        plt.title("Objective", fontsize=20)
        plt.show()

        output = 10
        w_norm_sorted = np.argsort(-self.w_norm[rank], axis=0)
        for i in range(self.w_norm[rank].shape[1]):
            for j in range(output):
                print(i + 1, j + 1, self.words[w_norm_sorted[j][i]], self.w_norm[rank][w_norm_sorted[j][i]][i])


if __name__ == '__main__':
    dt_df = pd.read_csv("CFB2019_scores.csv", header=None).values
    names_df = pd.read_csv("TeamNames.txt", header=None).values
    dt_np = np.array(dt_df)  # (4279, 4)
    names_np = np.array(names_df)
    t = 10000
    problem_1 = MarkovChain(dt_np, names_np, t)
    problem_1.construct()
    problem_1.rank()
    problem_1.convergence()

    names = 'nyt_vocab.dat'
    frq = 'nyt_data.txt'
    rank, it = 25, 100
    problem_2 = NMF(frq, names)
    problem_2.iteration(rank, it)
    problem_2.show(rank)
