import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np

class GetAndFormatTheData:
    #def __init__(self,Ticker):
    #    self.Ticker = Ticker
    
    def GetDataFromYahoo(self,Ticker,start,end,period):
        try:
            data = yf.download(Ticker,start=start,end=end,period=period)
            return data
        except:
            print('Error El Ticker no existe')
     
        
    




