import numpy as np

class Discretizer():

    def __init__(self, low, high, num_bins):
        self.BINS = num_bins+1
        self.low = low
        self.high = high
        self.bins = []
        self.init_bins()

    
    def init_bins(self):
        for i in range(len(self.low)):
            bin = np.linspace(self.low[i], self.high[i],self.BINS)
            self.bins.append(bin)

    def discretize(self, obs):
        discretized_obs = []
        for dim in range(len(self.low)):
            i=0
            while(obs[dim]>self.bins[dim][i]):
                i+=1
            discretized_obs.append(i-1)
        return discretized_obs
            

                
                    




