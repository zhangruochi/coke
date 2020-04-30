import matplotlib.pyplot as plt
import pandas as pd
import torch


def correlaition_matrix(numerical_matrix,selectd_num_features, labels )

    corr_matrix = numerical_matrix.loc[:, selectd_num_features]
    corr_matrix["target"] = labels
    corr_matrix.columns.tolist()
    names = corr_matrix.columns.tolist()

    correlations = corr_matrix.corr() 
    abs_corrections = abs(correlations)

    fig = plt.figure(figsize=(30,30)) 
    ax = fig.add_subplot() 
    ax = sns.heatmap(abs_corrections,cmap=plt.cm.Greys, linewidths=0.05,vmax=1, vmin=0 ,annot=True, annot_kws={'size':6,'weight':'bold'})

    plt.xticks(np.arange(len(names))+0.5,names, fontsize=20) 
    plt.yticks(np.arange(len(names))+0.5,names, fontsize=20) #
    ax.set_title('features correlation',fontsize=30)
    plt.show()

def plot_function(f, tx=None, ty=None, title=None, min=-2, max=2, figsize=(6,4)):
    x = np.linspace(min,max)
    fig,ax = plt.subplots(figsize=figsize)
    ax.plot(x,f(x))
    if tx is not None: ax.set_xlabel(tx)
    if ty is not None: ax.set_ylabel(ty)
    if title is not None: ax.set_title(title)



def pivot_plot(index,columns,values):

    fig = plt.figure(figsize=(14,6))

    table1 = pd.pivot_table(df,index=index,columns=columns,values=values)
    table1.plot(kind='bar',ax=ax1)
    # ax1.set_ylabel();
    plt.set_ylim((0,9.3))

