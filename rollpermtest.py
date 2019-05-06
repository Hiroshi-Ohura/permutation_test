import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pandas_datareader.data as web
import numpy as np
from dateutil.relativedelta import relativedelta
sns.set()

price = web.DataReader("spy", "yahoo", "1980/1/1").dropna()
price['return'] = price['Close'].pct_change()
ret = price.loc[:, ['return']].dropna()

month_begin = ret.resample('BMS').first()
sig_lng = pd.DataFrame(
    np.ones(len(month_begin)), index=month_begin.index, columns=['signal'])
sig = pd.DataFrame(sig_lng, index=ret.index).fillna(0)
test_ret = pd.concat([ret, sig], axis=1).\
    assign(test_ret=lambda d: d['return'] * d['signal'])


def permutation_test(df, n_sims):
    ref_dist = []
    for _ in range(n_sims):
        idx_perm = np.random.permutation(df.index)
        sig_perm = pd.DataFrame(df['signal'],
                                index=idx_perm).reset_index(drop=True)
        returns = df[['return']].reset_index(drop=True)
        ref_ret = sig_perm.values * returns.values
        ref_dist.append(ref_ret.cumsum()[-1])
    test_val = df['test_ret'].values.cumsum()[-1]
    pval = (ref_dist > test_val).sum() / n_sims
    return pval


pval_dict = {}
monthly_dates = ret.resample('BM').first().index
for date in monthly_dates:
    # 5Y rolling p value
    is_roll_rng = (test_ret.index > date) &\
                  (test_ret.index < date + relativedelta(years=5))
    df_roll_rng = test_ret[is_roll_rng]
    if test_ret.index[-1] < date + relativedelta(years=5):
        break
    else:
        pval_dict[df_roll_rng.index[-1]] = permutation_test(df_roll_rng, 100)

df_pval = pd.DataFrame.from_dict(pval_dict, orient='index', columns=['pvalue'])

sns.set_style('whitegrid')
fig, ax = plt.subplots()
sns.lineplot(x=df_pval.index, y=df_pval['pvalue'], ax=ax)
ax.set(title='5Y rolling permutation test')
ax.axhline(y=0.05, color='red', label='95% significant level',
           linestyle='--', linewidth=2)
ax.legend(loc='best')
plt.show()
