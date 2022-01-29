import numpy as np

'''
Cálcular a ecdf dos dados Experimentais
Receive an array and compute x,y to plot 
The ECDF is defined as ECDF(x) = (number of samples ≤ x) / (total number of samples)
'''

def ecdf(data):
    
    import numpy as np
    """Compute ECDF for a one-dimensional array of measurements."""
    
    # Number of data points: n
    n = len(data)

    # x-data for the ECDF: x
    x = np.sort(data)

    # y-data for the ECDF: y
    y = np.arange(1, n+1)/ n

    return x, y



''' Given an x,y  from ecdf plots ecdf graphically'''

def ecdf_plot(x, y, xlabel='name of xlabel'):
    
    import matplotlib.pyplot as plt
    
    _ = plt.plot(x,y,marker='.',linestyle='none')
    _ = plt.xlabel(xlabel)
    _ = plt.ylabel('ECDF')
    plt.margins(0.02)
    plt.show()
    

'''
ecdf_formal(x,data) returns value of the formal ecdf derived from dataset (data) for each value in array x (ecdf dos dados experimentais).

https://numpy.org/doc/stable/reference/generated/numpy.searchsorted.html

Sort data from theorical sample(data2), and using observation np.search sorted will create and array with best localization for theorical sample

x value (observations sorted) is used for ecdf and ecdf_formal(theorical in arbitrary point)

After create an array with values close to observation it's normalized /len(data2), now we have an theorical ecdf in specific point 
'''

def ecdf_formal(x, data2):  
    return np.searchsorted(np.sort(data2), x, side='right') / len(data2)


'''
Receives and data1 and data2(theorical) and calculates ecdf and cdf_formal returning K-S statistics

Now that you have ecdf from observation and theorical ecdf from theorical sample, let's compute the distances and use the max distance as K-S.

Write a function to compute the Kolmogorov-Smirnov statistic from two datasets, data1 and data2, 
in which data2 consists of samples from the theoretical distribution you are comparing your data to. 
Note that this means we are using hacker stats to compute the K-S statistic for a dataset and a theoretical distribution, 
not the K-S statistic for two empirical datasets.

'''


def ks_stat(data1, data2): # data2 consists of samples from the theoretical distribution you are comparing your data to
    # Compute ECDF from data: x, y
    x,y = ecdf(data1)
    
    # Calcula os valores correspondes da ecdf na cdf simulada
    cdf = ecdf_formal(x,data2)

    # Compute distances between concave corners and CDF
    D_top = cdf-y

    # Compute distance between convex corners and CDF
    D_bottom = cdf - y + 1/len(data1)

    return np.max((D_top, D_bottom))

'''

Here, n is the number of data points, and f is the function you will use to generate samples from the target CDF. For example, to test against an Exponential distribution, you would pass np.random.exponential as f. This function usually takes arguments, which must be passed as a tuple. So, if you wanted to take samples from an Exponential distribution with mean x_mean, you would use the args=(x_mean,) keyword. The keyword arguments size and n_reps respectively represent the number of samples to take from the target distribution and the number of replicates to draw.


n = size of samples that will be compared with theorical sample, use same size as yours observations 
f = function used to be compared (theorical). Ex: np.random.exponential()
args() = contain the parameters that i want to replicate for theorical function. Ex: mean
size = size of theorical sample from f
n_reps = how many times you want to replicate experiment that will calculate k-s comparing x_samp with theorical sample x_f

'''

def draw_ks_reps(n, f, args=(), size=10000, n_reps=10000):
    # Generate samples from target distribution
    x_f = f(*args,size=size)
    
    # Initialize K-S replicates
    reps = np.empty(n_reps)
    
    # Draw replicate, according with number of copies that i want
    for i in range(n_reps):
        # Draw samples for comparison
        x_samp = f(*args,size=n)
        
        # Compute K-S statistic
        reps[i] = dcst.ks_stat(x_samp,x_f)

    return reps

'''# Draw target distribution: x_f
data2 = np.random.exponential(np.mean(data),size=10000)
# Compute K-S stat: d
d = ks_stat(data,data2)

# Draw K-S replicates: reps - array containing simulated distances theorical x theorical n times
reps = draw_ks_reps(len(data), np.random.exponential, 
                         args=(np.mean(data),), size=10000, n_reps=10000)

# Compute and print p-value
p_val = sum(reps >= d) / 10000
print('p =', p_val)'''