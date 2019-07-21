import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

fx = pd.read_csv(r"C:\Users\Hiroshi\PycharmProjects\crowding\quote.csv",
                 parse_dates=[0]).\
    set_index("date").resample("W").last().pct_change().dropna(how="all")

correl_list = []
for date in fx.index:
    col_list = list(np.random.choice(fx.columns, 10, replace=False))
    g1_list = list(np.random.choice(col_list, 5, replace=False))
    g2_list = list(set(col_list) - set(g1_list))

    roll_correl = fx[g1_list + g2_list][: date].\
        rolling(window=52, min_periods=52).corr()

    # sign matrix
    g1_g1 = np.full((len(g1_list), len(g1_list)), 1)
    g2_g1 = np.full((len(g2_list), len(g1_list)), -1)
    g1_g2 = np.full((len(g1_list), len(g2_list)), -1)
    g2_g2 = np.full((len(g2_list), len(g2_list)), 1)
    g12_g1 = np.concatenate([g1_g1, g2_g1])
    g12_g2 = np.concatenate([g1_g2, g2_g2])
    sign_mat = np.concatenate([g12_g1, g12_g2], axis=1)

    correl_ = roll_correl.loc[date].values * sign_mat
    np.fill_diagonal(correl_, np.nan)
    correl_list.append(np.nanmean(correl_))

df = pd.DataFrame(correl_list, index=fx.index).\
    rename(columns={0: "correl"})
df.plot()
plt.show()