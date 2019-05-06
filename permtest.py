import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pandas_datareader.data as web
import numpy as np
sns.set()

price = web.DataReader("spy", "yahoo", "2000/1/1").dropna()
price['return'] = price['Close'].pct_change()
ret = price.loc[:, ['return']].dropna()

month_begin = ret.resample('BMS').first()
sig_lng = pd.DataFrame(
    np.ones(len(month_begin)), index=month_begin.index, columns=['signal'])
sig = pd.DataFrame(sig_lng, index=ret.index).fillna(0)
test_ret = pd.concat([ret, sig], axis=1).\
    assign(test_ret=lambda d: d['return'] * d['signal'])

ref_dist = []
for _ in range(1000):
    idx_perm = np.random.permutation(test_ret.index)
    sig_perm = pd.DataFrame(test_ret['signal'], index=idx_perm)\
        .reset_index(drop=True)
    returns = test_ret[['return']].reset_index(drop=True)
    ref_ret = sig_perm.values * returns.values
    ref_dist.append(ref_ret.cumsum()[-1])

test_val = test_ret['test_ret'].values.cumsum()[-1]
sig_level = np.percentile(ref_dist, 95)
print("P Value: {}".format((ref_dist > test_val).sum() / 1000))

sns.set_style('whitegrid')
fig, ax = plt.subplots()
sns.distplot(ref_dist, ax=ax, kde=False)
ax.set(xlabel='cumulative returns', title='permutation test')
ax.axvline(x=sig_level, color='red', label='95% significant level',
           linestyle='--', linewidth=2)
ax.axvline(x=test_val, color='green', label='test return',
           linestyle='--', linewidth=2)
ax.legend(loc='best')
plt.show()
