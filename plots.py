import seaborn as sns
import matplotlib.pyplot as plt
sns.set(font_scale = 1.5)
sns.set_style("whitegrid")

import pandas as pd


def exploring_simulated_tau(config_sim):
    tau = pd.read_csv(config_sim['path_features'] + 'true_tau.csv')

    knob_ks = tau[tau['knob'] == 'ks']
    knob_kh = tau[tau['knob'] == 'kh']
    knob_ko = tau[tau['knob'] == 'ko']

    fig, ((ax0, ax1, ax2)) = plt.subplots(ncols=3, nrows=1, figsize=[20, 4])

    sns.set(font_scale=1.5)

    knob_ks.alpha = knob_ks.alpha.astype('str')
    ax0 = sns.lineplot(x='alpha', y='tau', data=knob_ks, ax=ax0)
    ax0.set_ylabel('τ - True Treatment Effect')
    ax0.set_xlabel('α - Treat. Effect Scale Knob')

    ax1 = sns.lineplot(x='gamma', y='tau', data=knob_kh, ax=ax1)
    ax1.set_ylabel('')
    ax1.set_xlabel('γ - Heterogeneity Knob')

    ax2 = sns.lineplot(x='beta', y='tau', data=knob_ko, ax=ax2)
    ax2.set_ylabel('')
    ax2.set_xlabel('β - Overlap Knob')

