# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 15:41:06 2020

@author: JMMEINHARDT

This script is designed to assist with categorization of molecular performance properties by taking
an input Excel workbook "Descriptor Data.xlsx", applying a selection rule to the notebook, and then
writing it to a new Excel workbook "Filtered Descriptor Data.xlsx" with the rule applied.

"""

import pandas as pd

# initialize dataframe for input Excel workbook.
reduced_df = pd.read_excel('Dataset (Counting).xlsx', 0)
# ycolumn = 'Decomposition Temperature/ °C (5 K/ min, DSC)'

# apply selection rule.
lowerBound = 51.29195463
upperBound = 302.1465028
moiety = 'Nitrate'

print('Size of original dataframe:')
print(reduced_df.shape)

# iterate through each molecule in molecules_df dataframe and drop entries based on selection rule.
# reduced_df.drop(reduced_df[reduced_df['Impact Sensitivity / J without < and >'] > upperBound].index, axis = 0, inplace=True)
# reduced_df.drop(reduced_df[reduced_df['Impact Sensitivity / J without < and >'] < lowerBound].index, axis = 0, inplace=True)
reduced_df.drop(reduced_df[reduced_df[moiety] == 0].index, axis = 0, inplace=True)
print('Size of reduced dataframe:')
print(reduced_df.shape) 

#reduced_df.drop(reduced_df[reduced_df.ycolumn > upperBound].index, inplace=True)

# write to output Excel workbook.
# title = 'Filtered Descriptor Data ' + moiety
reduced_df.to_excel('Filtered Descriptor Data (' + moiety + ').xlsx')
