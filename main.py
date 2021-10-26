import random
import sys

import sqlite3
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.optimize import minimize, Bounds, LinearConstraint, leastsq


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def sum_list(lista):
    final = []
    for i in range(len(lista)):
        final += lista[i]
    return final


def expected_utility_value(alpha, p, z1, z2):
    alpha = np.array(alpha)
    return np.sign(z1) * np.power(p * np.abs(z1), alpha) + np.sign(z2) * np.power((1 - p) * np.abs(z2), alpha)


def expected_utility_error(alpha, p, z1, z2, ce):
    alpha = np.array(alpha)
    return np.power(
        np.sign(z1) * np.power(p * np.abs(z1), alpha) + np.sign(z2) * np.power((1 - p) * np.abs(z2), alpha) - ce, 2)


def CPT_utility_value(alpha, beta, gamma, delta, p, z1, z2):
    if z1 > 0 and z2 > 0:
        v1 = np.power(np.abs(z1), alpha)
        v2 = np.power(np.abs(z2), alpha)
    elif z1 <= 0 and z2 > 0:
        v1 = -np.power(np.abs(z1), beta)
        v2 = np.power(np.abs(z2), alpha)
    elif z1 > 0 and z2 <= 0:
        v1 = np.power(np.abs(z1), alpha)
        v2 = -np.power(np.abs(z2), beta)
    else:
        v1 = -np.power(np.abs(z1), beta)
        v2 = -np.power(np.abs(z2), beta)
    w = (delta * np.power(p, gamma)) / (delta * np.power(p, gamma) + np.power(1 - p, gamma))
    return w * v1 + (1 - w) * v2


def CPT_utility_error(alpha, beta, gamma, delta, p, z1, z2, ce):
    return np.power(CPT_utility_value(alpha, beta, gamma, delta, p, z1, z2) - ce, 2)


class CertainEquivalentData:

    def __init__(self, path, reproduce=False, quick=False):
        """
        loads data from sqlite3 file, creates 3 pandas dataframes:
        self.lottery => information about lotteries, id, prizes and probabilidades
        self.decision => experiment results; id_subject, id_loterry and certain equivalent (actual answer)
        self.subject => personal info about subject; id, female, semester, highincome

        :param path: path to data file
        :param reproduce: boolean: should the seed be fixed?
        """
        con = sqlite3.connect(path)
        self.lottery = pd.read_sql_query("SELECT * FROM lottery", con)
        self.decision = pd.read_sql_query("SELECT * FROM decision", con)
        self.subject = pd.read_sql_query("SELECT * FROM lottery", con)
        self.reproduce = reproduce
        self.quick = quick
        self.decision = pd.merge(
            self.decision,
            self.lottery,
            how="left",
            on=None,
            left_on="lottery",
            right_on="lottery",
            left_index=False,
            right_index=False,
            sort=False,
            suffixes=("_x", "_y"),
            copy=True,
            indicator=False,
            validate=None,
        )
        self.stored_results = {}
        self.stored_params = {}

    def test_expected_utility(self):
        print("Beggining process...")
        n_elem = len(self.decision) - 1
        if self.reproduce:
            random.seed(0)
        rand_sample = random.sample(range(n_elem), k=n_elem)
        tenfold_div = list(split(rand_sample, 10))
        stored_errors = {'errors_naive': [],
                         'errors_expected': [],
                         'errors_cpt': [],
                         'errors_ideal': []}
        stored_params = {'expected_param': [],
                         'CPT_params': []}
        for n, i in enumerate(tenfold_div):
            print(str((n + 1)) + " of 10")
            index = sum_list([part for part in tenfold_div if part != i])
            train_data = self.decision.iloc[index]
            test_data = self.decision.iloc[i]
            p, z1, z2, ce = np.array(train_data.p1), np.array(train_data.z1), np.array(train_data.z2), np.array(
                train_data.ce)

            def sum_sqrd_error(alpha):
                valores = [expected_utility_error(alpha, p[n], z1[n], z2[n], ce[n]) for n in range(len(p))]
                soma = sum(valores)
                return soma

            def sum_sqrd_error_CPT(x):
                alpha, beta, gamma, delta = x
                valores = [CPT_utility_error(alpha, beta, gamma, delta, p[n], z1[n], z2[n], ce[n]) for n in
                           range(len(p))]
                soma = sum(valores)
                return soma

            bound = Bounds(0, 2)
            res = minimize(sum_sqrd_error, x0=np.array([0.9]), bounds=bound)
            res2 = minimize(sum_sqrd_error_CPT,
                            x0=np.array([0.9, 1, .5, .5]),
                            bounds=([0, 2], [0, 2], [0, 2], [0, 2]))
            # print(res2)
            estimated_param, estimated_param_cpt = res.x, res2.x
            stored_params['expected_param'].append(estimated_param)
            stored_params['CPT_params'].append(estimated_param_cpt)
            error_naive, error_expected, error_cpt, error_ideal = 0, 0, 0, 0
            for n, row in test_data.iterrows():
                prediction_naive = self.expected_value(row['p1'], row['z1'], row['z2'])
                prediction_expected = expected_utility_value(estimated_param, row['p1'], row['z1'], row['z2'])
                prediction_cpt = CPT_utility_value(estimated_param_cpt[0],
                                                   estimated_param_cpt[1],
                                                   estimated_param_cpt[2],
                                                   estimated_param_cpt[3],
                                                   row['p1'],
                                                   row['z1'],
                                                   row['z2'])
                prediction_ideal = self.table_lookup(train_data, row['lottery'])
                error_naive += np.power(prediction_naive - row['ce'], 2)
                error_expected += np.power(prediction_expected - row['ce'], 2)
                error_cpt += np.power(prediction_cpt - row['ce'], 2)
                error_ideal += np.power(prediction_ideal - row['ce'], 2)
            stored_errors['errors_expected'].append(error_expected / len(test_data))
            stored_errors['errors_naive'].append(error_naive / len(test_data))
            stored_errors['errors_cpt'].append(error_cpt / len(test_data))
            stored_errors['errors_ideal'].append(error_ideal / len(test_data))
            if self.quick:
                break
        print(stored_errors)
        results = {'mean_error_naive': np.mean(stored_errors['errors_naive']),
                   'mean_error_expected': np.mean(stored_errors['errors_expected']),
                   'mean_error_cpt': np.mean(stored_errors['errors_cpt']),
                   'mean_error_ideal': np.mean(stored_errors['errors_ideal'])}
        self.stored_results = results
        self.stored_params = stored_params

        #return results

    def mean_estimated_params(self):
        """
        Run after the main test method

        :return: the mean estimated parameters of the two economic models (expected utility and CPT)
        """
        mean_expected = np.mean(self.stored_params['expected_param'])
        mean_alpha, mean_beta, mean_gamma, mean_delta = [0, 0, 0, 0]
        tam = len(self.stored_params['CPT_params'])
        for i in self.stored_params['CPT_params']:
            mean_alpha += i[0]
            mean_beta += i[1]
            mean_gamma += i[2]
            mean_delta += i[3]
        mean_alpha = mean_alpha / tam
        mean_beta = mean_beta / tam
        mean_gamma = mean_gamma / tam
        mean_delta = mean_delta / tam
        return [mean_expected, mean_alpha, mean_beta, mean_gamma, mean_delta]

    def print_results_completeness(self):
        error_naive, error_expected = self.stored_results['mean_error_naive'], self.stored_results[
            'mean_error_expected']
        error_cpt, error_ideal = self.stored_results['mean_error_cpt'], self.stored_results['mean_error_ideal']

        def completeness(erro):
            return 100 * (error_naive - erro) / (error_naive - error_ideal)

        bar = '------------------------------------------------------------'
        print(bar, 'Estimation results and errors:', bar, sep='\n', end="\n")
        print("{:<17} {:<15} {:<15}".format('Model', 'Squared Error', 'Completeness'))
        print("{:<17} {:<15} {:<15}".format('Naive', round(error_naive, 2), round(completeness(error_naive), 2)))
        print("{:<17} {:<15} {:<15}".format('Expected Utility', round(error_expected, 2),
                                            round(completeness(error_expected), 2)))
        print("{:<17} {:<15} {:<15}".format('CPT', round(error_cpt, 2), round(completeness(error_cpt), 2)))
        print("{:<17} {:<15} {:<15}".format('Ideal', round(error_ideal, 2), round(completeness(error_ideal), 2)))

    def print_results_params(self):
        params = self.mean_estimated_params()

        bar = '------------------------------------------------------------'
        print(bar, 'Estimated coeficients:', bar, sep='\n', end="\n")

        print("{:<20} {:<15}".format('Model', 'Coeficients'))
        print("{:<20} {:<15}".format('Expected Utility', params[0]))
        print("{:<20} {:<15}".format('CPT (alpha)', params[1]))

        print("{:<20} {:<15}".format('CPT (beta)', params[2]))
        print("{:<20} {:<15}".format('CPT (gamma)', params[3]))
        print("{:<20} {:<15}".format('CPT (delta)', params[4]))

    @staticmethod
    def expected_value(p, z1, z2):
        """
        :param p: probability of z1 happening
        :param z1: monetary prize 1
        :param z2: monetary prize 2
        :return: lottery expected value
        """
        return p * z1 + (1 - p) * z2

    @staticmethod
    def table_lookup(train_df, lottery):
        ansewrs = train_df.loc[train_df['lottery'] == lottery]
        mean_answer = np.mean(np.array(ansewrs['ce']))
        return mean_answer

    # -----Restrictiviness

    def mean_error_funs(self, fun1, fun2):
        p, z1, z2 = np.array(self.decision.p1), np.array(self.decision.z1), np.array(self.decision.z2)
        dist = np.mean(np.power(fun1(p, z1, z2) - fun2(p, z1, z2), 2))
        return dist

    def calculate_CPT_restrictiviness(self):
        pass


if __name__ == "__main__":
    data = CertainEquivalentData('Data/data.sqlite3',quick=True)
    erros = data.test_expected_utility()
    sys.stdout.flush()
    data.print_results_completeness()
    data.print_results_params()
