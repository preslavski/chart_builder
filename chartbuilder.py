# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 09:44:48 2018

@author: bpreslavski
"""

import pandas as pd
import numpy as np

class Chartbuilder:
    def __init__(self, data):
        self.data = data
    
    def countall(self):
        '''
        Counts all rows in a dataframe
        '''
        return self.data.shape[0]
    
    def sumcolumn(self, dimention):
        return self.data[dimention].sum()
    
    def countcolumn(self, dimention):
        '''
        Counts non-blank values in a dimention
        '''
        return self.data[dimention].count()
    
    def countbylist(self, dimention, indexlist, colnames=[]):
        '''
        Count unique occurence of values in dimention and applies indexlist
        '''
        data = self.data
        data = data.groupby(dimention).size()
        data = data.reindex(index=indexlist)
        data = data.to_frame(name = dimention)
        if colnames:
            data.columns=colnames
        return data
    
    def countbysort(self, dimention, value, colnames=[]):
        '''
        Count unique occurence of values in dimention and applies sorting
        '''
        data = self.data
        data = data.loc[data['Sentiment'] == value]
        data = data.groupby(dimention).size()
        data = data.to_frame(name = dimention)
        data = data.sort_values(dimention, ascending=False).head(10)
        if colnames:
            data.columns=colnames
        return data

    def twoaxisbreakdown(self, vdimention, hdimention, colnames=[]):
        data = self.data
        data = data.groupby([vdimention, hdimention]).size().unstack(1)
        data = data.reindex(columns=colnames)
        data['Total'] = data.sum(axis=1)
        #data.fillna(0, inplace = True)
        return data

    def trendbyvolume(self, vdimention, hdimention, colnames=[]):
        data = self.data
        data['Day'] = data[vdimention].dt.day
        data = data.groupby(['Day', hdimention]).size().unstack(1)
        data.fillna(0, inplace = True)
        #data = data.to_frame(name = vdimention)
        if colnames:
            data.columns=colnames
        return data
    
    def trendbysum(self, dimention, sumdimention, colnames=[]):
        data = self.data
        data['Day'] = data[dimention].dt.day
        data = data.groupby(['Day'])[sumdimention].agg('sum')
        data = data.to_frame(name = sumdimention)
        if colnames:
            data.columns=colnames
        return data
    
    def getmonth(self):
        data = self.data
        months = {'01':'Jan','02':'Feb','03':'Mar','04':'Apr','05':'May','06':'Jun','07':'Jul','08':'Aug','09':'Sep','10':'Oct','11':'Nov','12':'Dec'}
        data['Delivered'] = pd.to_datetime(data.Delivered)
        data['Month'] = data['Delivered'].dt.strftime('%m')
        mean_month = data['Month'].agg('mode')
        mean_month = mean_month.astype(str)
        mean_month = mean_month.replace(months, regex=True)
        mean_month = mean_month.values[0]
        return mean_month
    
    def norm(self, field):
        data = self.data
        split_df = data[field].str.split('|').apply(pd.Series,1).stack()
        split_df.index = split_df.index.droplevel(-1)
        split_df.name = field + '_normalized'
        split_df = split_df.to_frame()
        return split_df
    
    def filterdata(self, dimention, filter_list):
        data = self.data
        data = data.replace(np.nan, 'nan', regex=True)
        data = data.loc[data[dimention].isin(filter_list)]
        return data