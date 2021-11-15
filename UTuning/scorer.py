import numpy as np
import scipy.stats as stats
import scipy.integrate as integrate

class scorer:
    '''
    Scorer class, a class to represent the accuracy, precision and goodness
    score from an uncertainty model.
    
    Attributes
    ----------
    Accuracy : float
    Precision : float
    Goodness : float
    Overall uncertainty : float
    Indicator Function : float array
    '''
    def __init__(self, Prediction, Truth, Sigma):
        
        n_quantiles = 11
        self.perc = np.linspace(0.0, 1.01, n_quantiles)
        # self.A_array=np.zeros(Prediction.shape[0])
        # self.P_array=np.zeros(Prediction.shape[0])
        # self.G_array=np.zeros(Prediction.shape[0])
        #self.U_array=np.zeros(Prediction.shape[0])

        self.IF_array=np.zeros((Prediction.shape[0],n_quantiles))
        
        if len(Prediction.shape)>1:
            for i in range(Prediction.shape[0]):
                IF = APG_calc(Truth[i], Prediction[i,:], Sigma[i],n_quantiles)
                #self.U_array[i]=U
                # self.A_array[i]=A
                # self.P_array[i]=P
                # self.G_array[i]=G
                self.IF_array[i,:] = IF
        else:
            Pred_array = np.zeros((Sigma.shape[0],100))
            for i in range(Prediction.shape[0]):
                Pred_array[i,:] = np.random.normal(loc=Prediction[i],scale=Sigma[i],size=100)
                IF = APG_calc(Truth[i], Pred_array[i,:], Sigma[i],n_quantiles)
                #self.U_array[i]=U
                # self.A_array[i]=A
                # self.P_array[i]=P
                # self.G_array[i]=G
                self.IF_array[i,:] = IF
        
        self.avgIndFunc = np.mean(self.IF_array, axis=0)
        
        self.a = np.zeros(len(self.avgIndFunc))
        for i in range(len(self.avgIndFunc)):
            if self.avgIndFunc[i] > self.perc[i] or self.avgIndFunc[i] == self.perc[i]:
                self.a[i] = 1
            else:
                self.a[i] = 0
        
    def Accuracy(self):
        Accuracy = integrate.simps(self.a, self.perc)
        return Accuracy
        #return print('Accuracy = {0:2.2f}'.format(np.mean(self.A_array)))
    
    def Precision(self):
        Prec = self.a*(self.avgIndFunc-self.perc)
        Precision = 1-2*integrate.simps(Prec, self.perc)
        return Precision
        #return print('Precision = {0:2.2f}'.format(np.mean(self.P_array)))
    
    def Goodness(self):
        Sum = (3*self.a-2)*(self.avgIndFunc-self.perc)
        Goodness = 1-integrate.simps(Sum, self.perc)
        return Goodness
        #return print('Goodness = {0:2.2f}'.format(np.mean(self.G_array)))

    def Overall_uncertainty(self,Sigma):
        return Sigma.mean()
        #return print('Overall uncertainty = {0:2.2f}'.format(np.mean(self.G_array)))
    
    def IndicatorFunction(self):
        return self.IF_array

def APG_calc(Truth, Pred, Sigma,n_quantiles):

    mask = np.random.choice([False, True],
                            len(Pred),
                            p=[0, 1]) # To display randomly less points [Remove , Keep] in fraction

    Pred=Pred[mask]
    
    perc = np.linspace(0.0, 1.01, n_quantiles)

    F = np.zeros(Pred.shape[0])
    Indicator_func = np.zeros((Pred.shape[0], perc.shape[0]))

    # range of symmetric p-probability intervals
    plow = (1 - perc) / 2
    pupp = (1 + perc) / 2
    
    for i in range(len(Pred)):
        F[i] = stats.norm.cdf(Truth,
                          loc=Pred[i],
                          scale=Sigma)
        for proba_low, proba_upp in zip(plow, pupp):
            for k in range(len(plow)):
                if plow[k] < F[i] <= pupp[k]:
                    Indicator_func[i, k] = 1
                else:
                    Indicator_func[i, k] = 0

    avgIndFunc = np.mean(Indicator_func, axis=0)

    return avgIndFunc

# class scorer:
#     '''
#     Scorer class, a class to represent the accuracy, precision and goodness
#     score from an uncertainty model.
    
#     Attributes
#     ----------
#     Accuracy : float
#     Precision : float
#     Goodness : float
#     Overall uncertainty : float
#     Indicator Function : float array
#     '''
#     def __init__(self, Prediction, Truth, Sigma):
        
#         n_quantiles = 11
        
#         self.A_array=np.zeros(Prediction.shape[0])
#         self.P_array=np.zeros(Prediction.shape[0])
#         self.G_array=np.zeros(Prediction.shape[0])
#         self.U_array=np.zeros(Prediction.shape[0])

#         self.IF_array=np.zeros((Prediction.shape[0],n_quantiles))
        
#         if len(Prediction.shape)>1:
#             for i in range(Prediction.shape[0]):
#                 A,P,G,IF,U= APG_calc(Truth[i], Prediction[i,:], Sigma[i],n_quantiles)
#                 self.U_array[i]=U
#                 self.A_array[i]=A
#                 self.P_array[i]=P
#                 self.G_array[i]=G
#                 self.IF_array[i,:] = IF
#         else:
#             Pred_array = np.zeros((Sigma.shape[0],100))
#             for i in range(Prediction.shape[0]):
#                 Pred_array[i,:] = np.random.normal(loc=Prediction[i],scale=Sigma[i],size=100)
#                 A,P,G,IF,U= APG_calc(Truth[i], Pred_array[i,:], Sigma[i],n_quantiles)
#                 self.U_array[i]=U
#                 self.A_array[i]=A
#                 self.P_array[i]=P
#                 self.G_array[i]=G
#                 self.IF_array[i,:] = IF
        
                
#     def Accuracy(self):
#         return np.mean(self.A_array)
#         #return print('Accuracy = {0:2.2f}'.format(np.mean(self.A_array)))
    
#     def Precision(self):
#         return np.mean(self.P_array)
#         #return print('Precision = {0:2.2f}'.format(np.mean(self.P_array)))
    
#     def Goodness(self):
#         return np.mean(self.G_array)
#         #return print('Goodness = {0:2.2f}'.format(np.mean(self.G_array)))

#     def Overall_uncertainty(self):
#         return np.mean(self.G_array)
#         #return print('Overall uncertainty = {0:2.2f}'.format(np.mean(self.G_array)))
    
#     def IndicatorFunction(self):
#         return self.IF_array

# def APG_calc(Truth, Pred, Sigma,n_quantiles):

#     mask = np.random.choice([False, True],
#                             len(Pred),
#                             p=[0, 1]) # To display randomly less points [Remove , Keep] in fraction

#     Pred=Pred[mask]
    
#     perc = np.linspace(0.0, 1.00, n_quantiles)

#     F = np.zeros(Pred.shape[0])
#     Indicator_func = np.zeros((Pred.shape[0], perc.shape[0]))

#     # range of symmetric p-probability intervals
#     plow = (1 - perc) / 2
#     pupp = (1 + perc) / 2
    
#     for i in range(len(Pred)):
#         F[i] = stats.norm.cdf(Truth,
#                           loc=Pred[i],
#                           scale=Sigma)
#         for proba_low, proba_upp in zip(plow, pupp):
#             for k in range(len(plow)):
#                 if plow[k] < F[i] <= pupp[k]:
#                     Indicator_func[i, k] = 1
#                 else:
#                     Indicator_func[i, k] = 0

#     avgIndFunc = np.mean(Indicator_func, axis=0)

#     a = np.zeros(len(avgIndFunc))
#     for i in range(len(avgIndFunc)):
#         if avgIndFunc[i] > perc[i] or avgIndFunc[i] == perc[i]:
#         #if avgIndFunc[i] > perc[i] or avgIndFunc[i] == perc[i] or avgIndFunc[i] > (perc[i]-perc[i]*0.1):
#             a[i] = 1
#         else:
#             a[i] = 0

#     U = Sigma.mean()

#     Accuracy = integrate.simps(a, perc)

#     Prec = a*(avgIndFunc-perc)
    
#     Precision = 1-2*integrate.simps(Prec, perc)

#     Sum = (3*a-2)*(avgIndFunc-perc)

#     Goodness = 1-integrate.simps(Sum, perc)

#     return Accuracy, Precision, Goodness, avgIndFunc, U