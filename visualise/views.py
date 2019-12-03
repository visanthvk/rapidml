from django.shortcuts import render
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt1
import seaborn as sns

def index(request):
    df = pd.read_csv(r'media/'+request.session['data_set'])
    desc = df.describe().to_html(classes="table table-bordered")
    head = df.head().to_html(classes="table table-bordered")
    d = df.iloc[:,:-1]
    for i in range(len(d.columns)):
        plt1.title(d.columns[i])
        plt1.boxplot(d[d.columns[i]])
        plt1.savefig(r'visualise/static/images/'+'box'+str(i)+'.png')
    box_plot_images = []
    plt1.close()
    plt.title("Class proportion")
    plt.ylabel("Class")
    plt.xlabel("Frequency")
    df.iloc[:,-1].value_counts().plot( kind='barh')
    plt.savefig(r'visualise/static/images/graph.png')
    for i in range(len(df.columns)):
        box_plot_images.append("images/box"+str(i)+".png")

    context = {
        'desc': desc, 
        'head': head,
        'box_plot_images': box_plot_images,
    }
    return render(request, 'visualise/index.html', context)