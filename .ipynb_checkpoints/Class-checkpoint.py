#%load_ext autoreload
#%autoreload 2
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
from scipy.stats import norm
from scipy.optimize import minimize

class GetAndFormatTheData:
    def GetDataFromYahoo(self,Ticker,start,end,interval):
        '''
          The function get the data from Yahoo finance
    
          Inputs: Ticker:string,date:string('YYYY-MM-DD'),
    
          period:string(1d,5d,1wk,1mo,3mo)
    
          output:return a data frame OHLS
        '''
        try:
            data = yf.download(Ticker,start=start,end=end,interval=interval)
            return data
        except:
            print('Error El Ticker no existe')
            
            
    def GetDataFromYahooSeveral(self,start,end,interval,columna,*args):
        '''
         The function get the data from Yahoo finance,join the columns and fill the values with the last value.
    
         Inputs: Ticker:string,date:string('YYYY-MM-DD'),period:string(1d,5d,1wk,1mo,3mo),columna:string.(Open,High,Low,Close,Adj Close,Volume),args: Several tickers
    
         output:return a data frame with the selected column and the tickers
        '''
        Base = pd.DataFrame()
        for i in args:
            assert isinstance(i,str),'The value is not string'
            try:
                data = self.GetDataFromYahoo(i,start,end,interval)[columna]
                Base = pd.merge(Base,data,how='outer',left_index=True,right_index=True)
                Base = Base.fillna(method='ffill')
            except:
                print("El Ticker no existe o no exsten datos en el Ticker %s"%(i))
        Base.columns = args
            
            
        return Base
            
class ReturnOps():
    def ComputeOnereturn(self,Base,columns):
        try:
            returns = Base[columns].pct_change()
            returns = returns.dropna()
            return returns
        except:
            print('Error no existe la columna o el archivo esta en el formato incorrecto')
    def ComputeSeveralReturns(self,Base):
        '''
        Funcion que toma un dataframe de precios y calcula los retornos
        '''
        returns = Base.pct_change()
        returns = returns.dropna()
        return returns
    
    def AnnualizingReturn(self,PerReturn,PerInAYear):
        '''
        Funcion que calcula el retorno anualizado
        '''
        ret = (1+PerReturn)**PerInAYear - 1
        return ret
    
    def MeanReturn(self,Base):
        Base = Base.mean()
        return Base
    
    def VolReturn(self,Base):
        Base = Base.std()
        return Base
    
    def Sesgo(self,Base):
        DesMean = Base-Base.mean()
        sigma_Base = Base.std(ddof=0)
        exp = (DesMean**3).mean()
        return exp/sigma_Base**3
    
    def kurtosis(self,Base):
        """
        Alternative to scipy.stats.kurtosis()
        Computes the kurtosis of the supplied Series or DataFrame
        Returns a float or a Series
        """
        demeaned_Base = Base - Base.mean()
        # use the population standard deviation, so set dof=0
        sigma_Base = Base.std(ddof=0)
        exp = (demeaned_Base**4).mean()
        return exp/sigma_Base**4
    
    def drawdown(self,return_series: pd.Series):
        """Takes a time series of asset returns.
           returns a DataFrame with columns for
           the wealth index, 
           the previous peaks, and 
           the percentage drawdown
        """
        wealth_index = 1000*(1+return_series).cumprod()
        previous_peaks = wealth_index.cummax()
        drawdowns = (wealth_index - previous_peaks)/previous_peaks
        return pd.DataFrame({"Wealth": wealth_index, 
                             "Previous Peak": previous_peaks, 
                             "Drawdown": drawdowns})
    def GetTheMinimunDateDraadown(self,Serie):
        Serie = self.drawdown(Serie)["Drawdown"].idxmin()
        return Serie
    
    def AnnualizedVol(self,Base,Per):
        '''
         Per the number of period in a year
        '''
        Base = self.VolReturn(Base)*np.sqrt(Per)
        return Base
    
    def AnnualizedMean(self,Base,DaysBase):
        '''
        The values should be divided by 100
        '''
        Days = Base.shape[0]
        Base = (Base+1).prod()**(DaysBase/Days) - 1
        return Base
    
    def is_normal(self,r,level=0.01):
        """
        Applies the Jarque-Bera test to determine if a Series is normal or not
        Test is applied at the 1% level by default
        Returns True if the hypothesis of normality is accepted, False otherwise
        """
        if isinstance(r, pd.DataFrame):
            return r.aggregate(self.is_normal)
        else:
            statistic, p_value = scipy.stats.jarque_bera(r)
            return p_value > level
    
    def semideviation(self,r):
        """
        Returns the semideviation aka negative semideviation of r
        r must be a Series or a DataFrame
        """
        is_negative = r < 0
        return r[is_negative].std(ddof=0)
    
    def varhistoric(self,r,level=5):
        """
        Returns the historic Value at Risk at a specified level
        i.e. returns the number such that "level" percent of the returns
        fall below that number, and the (100-level) percent are above
        """
        if isinstance(r,pd.DataFrame):
            return r.aggregate(self.varhistoric, level=level)

        elif isinstance(r, pd.Series):
            return -np.percentile(r, level)
        else:
            raise TypeError("Expected r to be a Series or DataFrame")
    
    def cvarhistoric(self,r,level=5):
        """
        Computes the Conditional VaR of Series or DataFrame
        """
        if isinstance(r,pd.Series):
            is_beyond = r <= -self.varhistoric(r, level=level)
            return -r[is_beyond].mean()
        elif isinstance(r,pd.DataFrame):
            return r.aggregate(self.cvarhistoric, level=level)
        else:
            raise TypeError("Expected r to be a Series or DataFrame")
    
    def vargaussian(self,r,level=5,modified=False):
        """
        Returns the Parametric Gauusian VaR of a Series or DataFrame
        If "modified" is True, then the modified VaR is returned,
        using the Cornish-Fisher modification
        """
        # compute the Z score assuming it was Gaussian
        z = norm.ppf(level/100)
        if modified:
            # modify the Z score based on observed skewness and kurtosis
            s = self.Sesgo(r)
            k = self.kurtosis(r)
            z = (z +
                    (z**2 - 1)*s/6 +
                    (z**3 -3*z)*(k-3)/24 -
                    (2*z**3 - 5*z)*(s**2)/36
                )
        return -(r.mean() + z*r.std(ddof=0))
    
    def annualize_rets(r, periods_per_year):
        """
        Annualizes a set of returns
        We should infer the periods per year
        but that is currently left as an exercise
        to the reader :-)
        """
        compounded_growth = (1+r).prod()
        n_periods = r.shape[0]
        return compounded_growth**(periods_per_year/n_periods)-1


    def annualize_vol(r, periods_per_year):
        """
        Annualizes the vol of a set of returns
        We should infer the periods per year
        but that is currently left as an exercise
        to the reader :-)
        """
        return r.std()*(periods_per_year**0.5)

class Portfolio(ReturnOps):
    def sharpe_ratioAnnual(self,Base,riskfree_rate,periods_per_year):
        """
        Computes the annualized sharpe ratio of a set of returns
        """
        # convert the annual riskfree rate to per period
        rf_per_period = (1+riskfree_rate)**(1/periods_per_year)-1
        excess_ret = Base - rf_per_period
        ann_ex_ret = self.AnnualizedMean(excess_ret,periods_per_year)
        ann_vol = self.AnnualizedVol(Base,periods_per_year)
        return ann_ex_ret/ann_vol
    
    
    def portfolio_return(self,weights,returns):
        """
        Computes the return on a portfolio from constituent returns and weights
        weights are a numpy array or Nx1 matrix and returns are a numpy array or Nx1 matrix
        """
        return weights.T @ returns
    
    def portfolio_vol(self,weights,covmat):
        """
        Computes the vol of a portfolio from a covariance matrix and constituent weights
        weights are a numpy array or N x 1 maxtrix and covmat is an N x N matrix
        """
        return (weights.T @ covmat @ weights)**0.5
    
    def plot_ef2(self,n_points,er,cov):
        """
        Plots the 2-asset efficient frontier
        """
        if er.shape[0] != 2 or er.shape[0] != 2:
            raise ValueError("plot_ef2 can only plot 2-asset frontiers")
        weights = [np.array([w, 1-w]) for w in np.linspace(0, 1, n_points)]
        rets = [self.portfolio_return(w, er) for w in weights]
        vols = [self.portfolio_vol(w, cov) for w in weights]
        ef = pd.DataFrame({
            "Returns": rets, 
            "Volatility": vols
        })
        return ef.plot.line(x="Volatility", y="Returns", style=".-")

    
    def minimize_vol(self,target_return,er,cov):
        """
        Returns the optimal weights that achieve the target return
        given a set of expected returns and a covariance matrix
        """
        n = er.shape[0]
        init_guess = np.repeat(1/n, n)
        bounds = ((0.0, 1.0),) * n # an N-tuple of 2-tuples!
        # construct the constraints
        weights_sum_to_1 = {'type': 'eq',
                            'fun': lambda weights: np.sum(weights) - 1
        }
        return_is_target = {'type': 'eq',
                            'args': (er,),
                            'fun': lambda weights, er: target_return - self.portfolio_return(weights,er)
        }
        weights = minimize(self.portfolio_vol, init_guess,
                           args=(cov,), method='SLSQP',
                           options={'disp': False},
                           constraints=(weights_sum_to_1,return_is_target),
                           bounds=bounds)
        return weights.x
    
    def optimal_weights(self,n_points,er,cov):
        """
        Returns a list of weights that represent a grid of n_points on the efficient frontier
        """
        target_rs = np.linspace(er.min(), er.max(),n_points)
        weights = [self.minimize_vol(target_return,er,cov) for target_return in target_rs]
        return weights
    
    
    def msr(self,riskfree_rate,er,cov):
        """
        Returns the weights of the portfolio that gives you the maximum sharpe ratio
        given the riskfree rate and expected returns and a covariance matrix
        """
        n = er.shape[0]
        init_guess = np.repeat(1/n, n)
        bounds = ((0.0, 1.0),) * n # an N-tuple of 2-tuples!
        # construct the constraints
        weights_sum_to_1 = {'type': 'eq',
                            'fun': lambda weights: np.sum(weights) - 1
        }
        def neg_sharpe(weights,riskfree_rate,er,cov):
            
            """
            Returns the negative of the sharpe ratio
            of the given portfolio
            """
            
            r = self.portfolio_return(weights,er)
            vol = self.portfolio_vol(weights,cov)
            return -(r - riskfree_rate)/vol

        weights = minimize(neg_sharpe,init_guess,
                           args=(riskfree_rate, er, cov), method='SLSQP',
                           options={'disp': False},
                           constraints=(weights_sum_to_1,),
                           bounds=bounds)
        return weights.x
    
    def gmv(self,cov):
        """
        Returns the weights of the Global Minimum Volatility portfolio
        given a covariance matrix
        """
        n = cov.shape[0]
        return self.msr(0,np.repeat(1,n),cov)
    
    def plot_ef(self,n_points,er,cov,style='.-',legend=False,show_cml=False,riskfree_rate=0,show_ew=False,show_gmv=False):
        """
        Plots the multi-asset efficient frontier
        """
        weights = self.optimal_weights(n_points,er,cov)
        rets = [self.portfolio_return(w, er) for w in weights]
        vols = [self.portfolio_vol(w, cov) for w in weights]
        ef = pd.DataFrame({
            "Returns": rets, 
            "Volatility": vols
        })
        ax = ef.plot.line(x="Volatility", y="Returns",style=style,legend=legend)
        if show_cml:
            ax.set_xlim(left = 0)
            # get MSR
            w_msr = self.msr(riskfree_rate,er,cov)
            r_msr = self.portfolio_return(w_msr,er)
            vol_msr = self.portfolio_vol(w_msr,cov)
            # add CML
            cml_x = [0, vol_msr]
            cml_y = [riskfree_rate,r_msr]
            ax.plot(cml_x, cml_y,color='green',marker='o',linestyle='dashed',linewidth=2,markersize=10)
        if show_ew:
            n = er.shape[0]
            w_ew = np.repeat(1/n, n)
            r_ew = self.portfolio_return(w_ew,er)
            vol_ew = self.portfolio_vol(w_ew,cov)
            # add EW
            ax.plot([vol_ew], [r_ew],color='goldenrod',marker='o',markersize=10)
        if show_gmv:
            w_gmv = self.gmv(cov)
            r_gmv = self.portfolio_return(w_gmv,er)
            vol_gmv = self.portfolio_vol(w_gmv,cov)
            # add EW
            ax.plot([vol_gmv],[r_gmv],color='midnightblue',marker='o',markersize=10)

            return ax


    
    




        
    




