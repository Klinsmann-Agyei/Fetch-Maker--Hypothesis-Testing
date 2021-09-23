# Import libraries
import numpy as np
import pandas as pd

dogs = pd.read_csv('dog_data.csv')

print(dogs.head())

whippet_rescue = dogs.is_rescue[dogs.breed == 'whippet']

num_whippet_rescues = np.sum(whippet_rescue == 1)
print(num_whippet_rescues)

num_whippets = len(whippet_rescue)
print(num_whippets)

from scipy.stats import binom_test
pval = binom_test(num_whippet_rescues, num_whippets, .08)
print(pval)

wt_whippets = dogs.weight[dogs.breed == 'whippet']
wt_terriers = dogs.weight[dogs.breed == 'terrier']
wt_pitbulls = dogs.weight[dogs.breed == 'pitbull']

from scipy.stats import f_oneway
Fstat, pval = f_oneway(wt_whippets, wt_terriers, wt_pitbulls)
print(pval)

dogs_wtp = dogs[dogs.breed.isin(['whippet', 'terrier', 'pitbull'])]

from statsmodels.stats.multicomp import pairwise_tukeyhsd
output = pairwise_tukeyhsd(dogs_wtp.weight, dogs_wtp.breed)
print(output)

dogs_ps = dogs[dogs.breed.isin(['poodle', 'shihtzu'])]

Xtab = pd.crosstab(dogs_ps.color, dogs_ps.breed)
print(Xtab)

from scipy.stats import chi2_contingency
chi2, pval, dof, exp = chi2_contingency(Xtab)
print(pval)