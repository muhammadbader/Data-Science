'''
    Pandas is build on top of Numpy
'''
import numpy as np
import pandas as pd

'''
    Seriess
'''

# labels = ['a','b','c']
# my_arr = [10,20,30]
# arr = np.array(my_arr)
# d = {x:y for x,y in zip(labels,my_arr)}


# print(pd.Series(data=my_arr)) # it prints with default indexes
# print(pd.Series(data=my_arr,index=labels)) # it prints with my indexes
# print(pd.Series(y_arr,labels)) # it prints with my indexes
    # panda also deals with numpy arrays
# print(pd.Series(data=arr)) # it prints with default indexes
# print(pd.Series(data=arr,index=labels)) # it prints with my indexes
    # DICTS
# print(pd.Series(d)) # automatically puts the data and indexes
    # and can hold variety of things
# pd.Series([sum,print,len])

        # Using Series
# ser1 = pd.Series([1,3,4,2,5],['Muh',"San","Mar","Moa","Mal"])
# ser2 = pd.Series(range(1,6),['Sol',"San","Mar","Moa","Mal"])
# print(ser1)
# print(ser2)
# print(ser1["Muh"])
# print(ser1+ser2) # it sums the intersections of both series and NaN for for non intersections


'''
    DataFrames
'''

# from numpy.random import randn
# np.random.seed(101)
# df = pd.DataFrame(randn(5,4),["A","B","C","D","E"],["W","X","Y","Z"]) # draws a 2D table
# print(df)
# print(df["W"]) # prints the whole columt as a Series
# print(df[["X","Z"]]) # prints the whole columt as a Series
# df["newCol"] = df["W"]+df["Y"]  # we can add a new column
# df["newCol2"] = range(0,5) # we can add a new column
# print(df)

# df2 = df.drop("Y",axis=1) # deletes a column-> default is axis = 0 deletes a row
# print(df) # not in place
# print(df2) # not in place
        # in place deletion
# df.drop("Y",axis=1,inplace=True) # deletes a column-> default is axis = 0 which is not a column
# print(df)


# selecting rows in DF

# print(df.loc["B"]) # return a series
# print(df.iloc[2]) # suing numeric indexing for rows,  return a series
# print(df.loc["B","Y"]) ## similar to dealing with matrixes
# print(df.loc[["A","B"],["Y","W"]]) ## similar to dealing with matrixes, returns a data frame


'''
     Conditional Selection
'''
# similarly to numpy using logical operators returns a boolean Data Frame

# print(df<1.2)
# bool_df = df>0
# print(df[bool_df])  # similar to df[df>0]

    # choosing based on specific column
# print(df[df["X"]>0])
# dfx = df[df["X"]>0]
# print(dfx["W"])
    # all in one line --> df[df["X"]>0]["W"]


        # Multiple conditions
# print(df[(df["W"]>0) & (df["X"]<1]))  # be carefull to use & not and,, using the () effects

'''
    Indexing Details
'''

# print(df.reset_index())  # adds a column of indexes not inplace unless entered inplace=True

# newindx = "KQ JL KB UAF IST".split()
# df["cities"]=newindx
# print(df.set_index("cities"))  ## overwrite the entered column name with the rows names ,, not inplace unless entered inplace=True


'''
    Multi-Index and Index Hierarchy
'''

# outside = ['G1','G1','G1','G2','G2','G2']
# inside = [1,2,3,1,2,3]
# hier_index = list(zip(outside,inside))
# hier_index = pd.MultiIndex.from_tuples(hier_index)
#
# df = pd.DataFrame(randn(6,2),hier_index,["A",'B']) # creating multiple levels of indexes

# df.loc["G1"].loc[2]  # how to take values from the multi-index

    # we can also name the columns
# df.index.names = ["Groups","Num"]


        # Cross sections

# print(df.xs(1,level="Num"))  # chooses all the rows with the value of 1 in column Num in all the G's
# print(df.xs("G1",1)) # cross-section between G1 and 1

'''
    Missing Data
'''

# df = pd.DataFrame({'A':[1,2,np.nan],
#                   'B':[5,np.nan,np.nan],
#                   'C':[1,2,3]})
# df.dropna()     # drops any row with a NaN value in it
# df.dropna(axis=1)     # drops any column with a NaN value in it
# df.dropna(thresh=2)   # enters a threshHold for the min NaN to drop the row/col

# df.fillna(value="New Value")    # opposite to dropna, fillna repalces all the NaN with input value
# df.fillna(value="New Value",limit=2)    # using limit=? limits the number of NaN to change in every row/col
# df['A'].fillna(value=df['A'].mean())      # fill the A col with the mean of the col


'''
    Groupby
'''


# data = {'Company':['GOOG','GOOG','MSFT','MSFT','FB','FB'],
#        'Person':['Sam','Charlie','Amy','Vanessa','Carl','Sarah'],
#        'Sales':[200,120,340,124,243,350]}
#
# df = pd.DataFrame(data)
#
# byComp = df.groupby("Company")
# print(byComp.mean())
# print(byComp.sum())
#
# print(byComp.count())

# print(byComp.describe()) ## gives you all the info of each group
# print(byComp.describe().transpose()['GOOG'])


'''
    Merging joining and Concatenating
'''

df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
                        'B': ['B0', 'B1', 'B2', 'B3'],
                        'C': ['C0', 'C1', 'C2', 'C3'],
                        'D': ['D0', 'D1', 'D2', 'D3']},
                        index=[0, 1, 2, 3])
# df2 = pd.DataFrame({'A': ['A4', 'A5', 'A6', 'A7'],
#                         'B': ['B4', 'B5', 'B6', 'B7'],
#                         'C': ['C4', 'C5', 'C6', 'C7'],
#                         'D': ['D4', 'D5', 'D6', 'D7']},
#                          index=[4, 5, 6, 7])
#
# df3 = pd.DataFrame({'A': ['A8', 'A9', 'A10', 'A11'],
#                         'B': ['B8', 'B9', 'B10', 'B11'],
#                         'C': ['C8', 'C9', 'C10', 'C11'],
#                         'D': ['D8', 'D9', 'D10', 'D11']},
#                         index=[8, 9, 10, 11])

# print(pd.concat([df1,df2,df3])) # concats all the dataFrames together,, the dimentions should match
# print(pd.concat([df1,df2,df3],axis=1))

'''
    Merging
'''


# left = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
#                      'A': ['A0', 'A1', 'A2', 'A3'],
#                      'B': ['B0', 'B1', 'B2', 'B3']})
#
# right = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
#                       'C': ['C0', 'C1', 'C2', 'C3'],
#                       'D': ['D0', 'D1', 'D2', 'D3']})

# print(pd.merge(left,right,how='inner',on='key'))    # the merge func merges DF based on the on= key

# left = pd.DataFrame({'key1': ['K0', 'K0', 'K1', 'K2'],
#                      'key2': ['K0', 'K1', 'K0', 'K1'],
#                      'A': ['A0', 'A1', 'A2', 'A3'],
#                      'B': ['B0', 'B1', 'B2', 'B3']})
#
# right = pd.DataFrame({'key1': ['K0', 'K1', 'K1', 'K2'],
#                       'key2': ['K0', 'K0', 'K0', 'K0'],
#                       'C': ['C0', 'C1', 'C2', 'C3'],
#                       'D': ['D0', 'D1', 'D2', 'D3']})
# print(pd.merge(left,right,on=['key1','key2']))



'''
    Joining
        it is simimlar to merging but with index param in DF
'''

# left = pd.DataFrame({'A': ['A0', 'A1', 'A2'],
#                      'B': ['B0', 'B1', 'B2']},
#                       index=['K0', 'K1', 'K2'])
#
# right = pd.DataFrame({'C': ['C0', 'C2', 'C3'],
#                     'D': ['D0', 'D2', 'D3']},
#                       index=['K0', 'K2', 'K3'])
# print(left)
# print(right)
# print(left.join(right))



'''
    Operations
'''

# df = pd.DataFrame({'col1':[1,2,3,4],'col2':[444,555,666,444],'col3':['abc','def','ghi','xyz']})
# df.head()

# df['col2'].unique() # finds the unique values in col2
# df['col2'].nuunique() # finds the numebr of unique values in col2   df['col2'].nuunique()=len(df['col2'].unique())
# df['col2'].value_counts()   # counts the unique values

        # Applying methos in DF

# def thrice(x):
#     return x**3

# df['col1'].apply(thrice)
# df['col3'].apply(len)
# df['col2'].apply(lambda x: x**2)


# df.sort_values('col2')  # sort out the values based on col2
# df.isnull()   # as the name suspects


# df = pd.DataFrame({'A':['foo','foo','foo','bar','bar','bar'],
#      'B':['one','one','two','two','one','one'],
#        'C':['x','y','x','y','x','y'],
#        'D':[1,3,2,5,4,1]})

# df2 = df.pivot_table(values='D',index=['A', 'B'],columns=['C'])     # creates a multi-index DF based on the axises in the input



'''
    Data Input and Output
'''

# df = pd.read_csv('example')     # reads a txt file as a DF

# df.to_csv('example',index=False)    # writes a csv file without indexes


    # Excels files

# pd.read_excel('Excel_Sample.xlsx',sheet_name='Sheet1')
# df.to_excel('Excel_Sample.xlsx',sheet_name='first try')


    # html

# data = pd.read_html("http://www.fdic.gov/bank/individual/failed/banklist.html")
# data[0]


    # SQL


# from sqlalchemy import create_engine



# engine = create_engine('sqlite:///:memory:') # a temporary sql memory
# df.to_sql('my_table', engine)
#
# sqldp = pd.read_sql('my_table',con=engine)
# print(sqldp)







