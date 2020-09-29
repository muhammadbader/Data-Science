'''
    Data Visualization
'''

'''
    MatPlot
'''
import matplotlib.pyplot as plt

import numpy as np

x = np.linspace(-2,5,20)
y = x**3
plt.plot(x,y)   # you need to use both in order to see the plot
plt.xlabel("from -2 to 5 label")
plt.ylabel("Y label")
plt.title("Muhammad")
plt.show()

plt.subplot(1,2,1)  #subplot(hight,width, serial number)
plt.plot(x,y,'r')
plt.show()
plt.subplot(1,2,2)
plt.plot(y,x,'b')
plt.show()


'''
    OOP method
'''
fig = plt.figure()  # basically the same result with OOP method
axis = fig.add_axes([0.1,0.1,0.8,0.8])
axis.plot(x,y)
axis.set_xlabel("X label") ## ... the same
plt.show()


fig = plt.figure()
axis1 = fig.add_axes([0.1,0.1,0.8,0.8]) #add_axes creates a new axis for the plot
axis2 = fig.add_axes([0.2,0.5,0.3,0.4])
axis1.plot(x,y)
axis2.plot(y,x)
plt.show()


fig,axis = plt.subplots()
fig,axis = plt.subplots(nrows=1,ncols=2)    # we can specifiy the no. of rows and cols in subplots
axis.plot(x,y)
plt.show()

fig,axis = plt.subplots(nrows=3,ncols=3)
plt.tight_layout()
# axes[0].plot(x,y)
# axes.set_title("Title 1")
# axes[1].plot(y,x)
plt.show()

'''
    Figure size and DPI (plot size)
'''

# fig = plt.figure(figsize=(3,2),dpi=100) #figsize plays with the size of the plot (3,2)
# ax = fig.add_axes([0,0,1,1])
# ax.plot(x,y)
# plt.show()
#
# fig,axes = plt.subplots(figsize=(5,5),dpi=100)
# axes.plot(x,y)
# plt.show()

'''
    Saving the plots as Images
'''
# fig,axes = plt.subplots(figsize=(5,5),dpi=100)
# axes.plot(x,y)
# fig.savefig('my_fpic.png',dpi=200)


# fig = plt.figure(figsize=(3,2),dpi=200) #figsize plays with the size of the plot (3,2)
# ax = fig.add_axes([0,0,1,1])
# ax.plot(x,x**2,label="X Square")
# ax.plot(x,y,label='X Cube')
# ax.legend() # adding labels to identify the plots
## ax.legend(loc=0) # adding labels to identify the plots, loc=0 puts thelegend in the best place for it or we can use loc = (tuple) to specify a place


'''
    Plot colors
'''

# fig = plt.figure()
# ax = fig.add_axes([-1,-1,1,1])
# ax.plot(x,y,color='orange',linewidth = 4, alpha = 0.5,linestyle = '--') # search for RGB hex code, linewidth or lw = 1 is the default, alpha measures the transperancy, linestyle or ls

# fig = plt.figure()
# ax = fig.add_axes([-1,-1,1,1])
# ax.plot(x,y,color='red',linewidth=1,alpha = 1,linestyle ='-',marker='+',markersize=10,markerfacecolor='blue') # marker to mark the x dots that I have entered


'''
    Zoom in/out
'''
# fig = plt.figure()
# ax = fig.add_axes([-1,-1,1,1])
# ax.plot(x,y)
# ax.set_xlim([0,1])
# ax.set_ylim([0,1])
# plt.show()


'''
    Seaborn
'''
# open source -> find on google -> github

# import seaborn as sns

'''
        Dsitribution plots
'''
# tips = sns.load_dataset('tips')
# print(tips.head())

# sns.distplot(tips['total_bill'],kde=False,bins=30)    # shows the distribution of the table based on 'total_bill' column, kde == stands for the smooth curve, bins == for more accurate columnt in the graph
# plt.show()

# sns.jointplot(x='total_bill',y='tip',data=tips,kind='hex')  # kind -> feature for the dots in the graph, kind  = 'reg' draws a regression line, kind = 'kde' draws a topological figure
# plt.show()

# sns.swarnplot(x='total_bill',y='tip',data=tips,kind='hex')
# plt.show()

# sns.pairplot(tips,hue="sex")  # a fully graphs of your data (numerical columns), hue-> distinguesh in the graphs the categorical columns


'''
    Categorical plots
'''

# sns.barplot(x='sex',y='total_bill',data=tips)   # y must be numerical -> shows the average
# plt.show()

# sns.countplot(x='sex',data=tips)    # counts occurences in the according to category
# plt.show()


# sns.boxplot(x='day',y='total_bill',data=tips,hue='smoker')   # box plots according to x, hue -> splits each box to inner categories according to the param'
# plt.show()

# sns.factorplot(x='day',y='total_bill',data=tips,kind='bar')       # use the kind= field to specify the graph kind-> bar, violin...
# plt.show()


'''
    Matrix Plot
'''
# flights  = sns.load_dataset('flights')
#
# tc = tips.corr()         # turns the table to matrix form (correlation)
# sns.heatmap(tc,annot=True)
#
#     # turns into a matrix
# fp = flights.pivot_table(index='month',columns='year',values='passengers')
# sns.heatmap(fp)
# plt.show()

'''
    Grids
'''
# iris = sns.load_dataset('iris')

# pg = sns.PairGrid(iris)         # pg holds all the plots but needs a method to show them
# pg.map(plt.scatter)     # use scatter for all plots
#
# pg.map_diag(sns.distplot)
# pg.map_upper(plt.scatter)
# pg.map_lower(sns.kdeplot)
# plt.show()

# g = sns.FacetGrid(data=tips,col='time',row='smoker')    # similar to PairGrid but I can specify the row and cols
# g.map(sns.kdeplot,'total_bill')
# plt.show()

# g.map(plt.scatter,'total_bill','tip')  # when passing a method that requires more that one arg
# plt.show()


'''
    Regression Plot
'''
# sns.lmplot(x='total_bill',y='tip',data=tips)
# sns.lmplot(x='total_bill',y='tip',data=tips,col='sex',row='time')  # separates the plot according to col= and row= param



'''
    Style and Color
'''
# sns.set_style('whitegrid')  # set the plots background as whitegrid -> ticks, darkgrid
# sns.despine(left=True,bottom=True)       # removes the top-right and top axises and keeps left and bottom by default
# sns.countplot(x='sex',data=tips)
# plt.figure(figsize=(12,2))


# titanic = sns.load_dataset('titanic')


'''
    pandas data visualization
'''
# import pandas as pd
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
#
#
# df1 = pd.read_csv('df1',index_col=0)
# df2 = pd.read_csv('df2')
#
#
# df1["A"].hist()
# plt.show()
#
# df2.plot.area()
# plt.show()
#
# df2.plot.bar()
# plt.show()
#
#
# df1.plot.scatter(x="A",y='B',c="C")
# plt.show()
