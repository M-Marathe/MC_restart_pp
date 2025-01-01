import matplotlib.pyplot as plt
##%matplotlib inline
import seaborn as sns
sns.set(rc={"figure.dpi":100, 'savefig.dpi':300})
sns.set_context('notebook')
sns.set_style("ticks")
#from IPython.display import set_matplotlib_formats
#set_matplotlib_formats('retina', 'svg')
import numpy as np
import scipy.stats as st

# The probabilities:
ps = [0.1, 0.3, 0.4, 0.2] # this has to sum to 1
# And here are the corresponding values:
xs = np.array([1, 2, 3, 4])
# Here is how you can define a categorical rv:
X = st.rv_discrete(name='Custom Categorical', values=(xs, ps))

print(X.pmf(1), X.pmf(10))
X.rvs(size=10)
fig, ax = plt.subplots(dpi=150)
ax.bar(xs, X.pmf(xs))
ax.set_xlabel('$x$')
ax.set_ylabel('$p(x)$')
plt.show()