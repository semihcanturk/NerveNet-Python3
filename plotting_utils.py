import os
import re

import numpy as np
import pandas as pd

CENT_DICT = {'Six': 6, 'Eight': 8, 'Ten': 10, 'Twelve': 12, 'Twenty': 20, 'Forty': 40}


def collect_train_data(logfile):
    iters, avg_rewards, entropies, loss_kl, loss_surr, loss_vf, loss_l2 = list(), list(), list(), list(), list(), list(), list()

    pattern = r"(.*)@main.py:85]\D*?(\d+)\D*?"
    with open(logfile) as f:
        lines = f.readlines()
        for line in lines:
            num_iters = re.match(pattern, line)
            if num_iters is not None:
                num_iters = int(num_iters.group(0).rsplit(' ', 1)[-1])
                iters.append(int(num_iters))
            elif 'avg_reward:' in line:
                avg_rewards.append(float(line.rsplit(' ', 1)[-1]))
            elif 'entropy:' in line:
                entropies.append(float(line.rsplit(' ', 1)[-1]))
            elif ' kl:' in line:
                loss_kl.append(float(line.rsplit(' ', 1)[-1]))
            elif 'surr_loss:' in line:
                loss_surr.append(float(line.rsplit(' ', 1)[-1]))
            elif 'vf_loss:' in line:
                loss_vf.append(float(line.rsplit(' ', 1)[-1]))
            elif 'weight_l2_loss:' in line:
                loss_l2.append(float(line.rsplit(' ', 1)[-1]))

    iters = np.array(iters)
    avg_rewards = np.array(avg_rewards)
    entropies = np.array(entropies)
    loss_kl = np.array(loss_kl)
    loss_surr = np.array(loss_surr)
    loss_vf = np.array(loss_vf)
    loss_l2 = np.array(loss_l2)
    return iters, avg_rewards, entropies, loss_kl, loss_surr, loss_vf, loss_l2


def generate_train_df(dict):
    reward_dict = dict.copy()
    for model_name, logfile in iter(dict.items()):
        iters, avg_rewards, entropies, loss_kl, loss_surr, loss_vf, loss_l2 = collect_train_data(logfile)
        reward_dict[model_name] = avg_rewards
    df = pd.DataFrame.from_dict(reward_dict)
    df['iter'] = iters
    df.set_index('iter', inplace=True)
    return df


def collect_test_data(logfile):
    with open(logfile) as f:
        lines = f.readlines()
        for line in lines:
            if 'Test performance (100 rollouts):' in line:
                avg = float(line.rsplit(' ', 1)[-1])
            elif 'max:' in line:
                max = float(line.split(',', 1)[0].rsplit(' ', 1)[-1])
    return avg, max


def get_all_files(path):
    files = []
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            files.append(os.path.join(dirpath, filename))
    return files


def get_centipede_type(logfile):
    for k in CENT_DICT.keys():
        if k in logfile:
            return CENT_DICT[k]
    raise KeyError(f'Key not found')


def generate_transfer_df(path):
    rows = list()
    logfiles = get_all_files(path)
    for logfile in logfiles:
        path_comps = logfile.split('/')
        model = path_comps[2]
        type = get_centipede_type(path_comps[-1])
        avg, max = collect_test_data(logfile)
        rows.append([model, type, avg, max])
    df = pd.DataFrame(rows, columns=['Model', 'Centipede Type', 'Avg Reward', 'Max Reward'])
    return df
