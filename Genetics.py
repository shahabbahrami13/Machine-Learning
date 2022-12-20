from math import sin
from math import pi
from math import sqrt
from math import pow
from math import exp
from math import fabs
from random import randrange
from random import uniform
from random import random
import numpy as np
import pandas as pd


class Genetic:
    @staticmethod
    def Generation(sample_num: int, x_num: int, *args):
        
        if x_num == len(args):
            g = np.arange(1, x_num + 1).reshape((1, x_num))

            for j in range(sample_num):
                one_row = []
                for i in args:
                    x = uniform(i[0], i[1])//0.01
                    x = x/100
                    one_row.append(x)
                one_row = np.array(one_row).reshape(1, x_num)
                g = np.r_['0', g, one_row]
            g = np.delete(g, 0, 0)
            return g
        else:
            print('x_num is smaller or larger than number of ranges!!(in Generation)')
            return None

    @staticmethod
    def Cross_over(g):
        
        if g is not None:
            sample_num = len(g)
            x_num = len(g[0])
            C = np.arange(1, x_num + 1).reshape((1, x_num))

            i = 0
            while i <= sample_num/8:
                i += 1

                r = randrange(0, sample_num-1)

                row_of_C = g[r].reshape((1, x_num))
                C = np.r_['0', C, row_of_C]
                C = np.r_['0', C, row_of_C]

                for c in range(len(g[r])):
                    a = random() // 0.1
                    a = a / 10
                    C[i, c] = g[r, c] * a + g[r+1, c] * (1 - a)
                    C[i+1, c] = g[r+1, c] * a + g[r, c] * (1 - a)

            C = np.delete(C, 0, 0)
            return np.r_['0', g, C]
        else:
            return None

    @staticmethod
    def Mutation(g, *args: list):
        
        if g is not None:
            sample_num = len(g)
            x_num = len(g[0])
            if x_num == len(args):
                M = np.arange(1, x_num+1).reshape((1, x_num))

                i = 0
                while i <= sample_num / 100:
                    i += 1

                    r = randrange(0, sample_num)
                    c = randrange(0, x_num)

                    row_of_M = g[r].reshape((1, x_num))
                    M = np.r_['0', M, row_of_M]
                    j1 = 0
                    for j2 in row_of_M[0]:
                        M[i, j1] = j2
                        j1 += 1

                    x = uniform(args[c][0], args[c][1])//0.01
                    x = x/100
                    M[i][c] = x

                M = np.delete(M, 0, 0)
                return np.r_['0', g, M]
            else:
                print('number of ranges is smaller or larger than x_num!!(in Mutation)')
        else:
            return None

    @staticmethod
    def Evaluation(sample_num, g, fit_ness=None, cost=None):
        
        if g is not None:
            if sample_num < len(g):
                if fit_ness:
                    answer = []
                    for row in g:
                        ans = fit_ness(row)
                        answer.append(ans)
                    answer = np.array(answer)

                    E = pd.DataFrame(g)
                    E['ans'] = answer
                    E = E.nlargest(sample_num, 'ans')
                    E = E.drop(columns='ans')

                    return E.values

                elif cost:
                    answer = []
                    for row in g:
                        ans = cost(row)
                        answer.append(ans)
                    answer = np.array(answer)

                    E = pd.DataFrame(g)
                    E['ans'] = answer
                    E = E.nsmallest(sample_num, 'ans')
                    E = E.drop(columns='ans')

                    return E.values

                else:
                    print('Enter fit_ness or cost function in Evaluation.')

            elif sample_num == len(g):
                return g
            else:
                print('your sample_num is smaller than your array(in Evaluation).')
                return None
        else:
            return None


def f(x):
    in_exp = fabs(100-(sqrt(pow(x[0], 2) + pow(x[1], 2))/pi))
    sins = fabs(sin(x[0])*sin(x[1]))
    ans = -0.0001*pow(sins*exp(in_exp)+1, 0.1)
    return ans


g = Genetic.Generation(400, 2, [-10, 10], [-10, 10])
iterable = 500

while iterable > 0 and g is not None:
    g = Genetic.Cross_over(g)
    g = Genetic.Mutation(g, [-10, 10], [-10, 10])
    g = Genetic.Evaluation(400, g, cost=f)
    iterable -= 1

if g is not None:
    print(g)
    print(f(g[0]))
    print('x1=', g[0, 0], '| x2=', g[0, 1])
