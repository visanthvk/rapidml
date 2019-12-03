from django.shortcuts import render
from sklearn.model_selection import train_test_split
from sklearn import svm
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix 
import seaborn as sn

def index(request):
    classification_models = ["SVM", "KNN", "Decision Tree", "Logistic Regression"]
    regression_models = ["Linear Regression", "Lasso Regression", "Ridge Regression", "Bayesian Ridge Regression"]
    models = regression_models
    if(request.session["type"]=="classification"):
        models = classification_models
    context = {
        'models': models,
    }
    return render(request, 'train/index.html', context)


def metrics(y_true,y_pred):
    from sklearn.metrics import f1_score
    f1 = f1_score(y_true, y_pred, average='macro')
    return f1
   
def train(request):
    df = pd.read_csv(r'media/'+request.session['data_set'])
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=int(request.POST['split'])/100)
    model = request.POST['model']

    classification_models = ["SVM", "KNN", "Decision Tree", "Logistic Regression"]
    regression_models = ["Linear Regression", "Lasso Regression", "Ridge Regression", "Bayesian Ridge Regression"]
    models = regression_models
    if(request.session['type']=="classification"):
        models = classification_models
    request.session['model'] = model

    if model =='SVM':
        clf = svm.SVC(C=1.0, kernel='linear')

    elif model== 'KNN':
        clf = KNeighborsClassifier(n_neighbors=5)

    elif model == 'Decision Tree':
        clf = DecisionTreeClassifier(random_state=0)

    elif model == 'Logistic Regression':
        clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial').fit(X_train, y_train)
        
    elif model == 'Linear Regression':
        reg = linear_model.LinearRegression().fit(X_train, y_train)
        
    elif model == 'Ridge Regression':
        reg = linear_model.Ridge(alpha=.5).fit(X_train, y_train)

    elif model == 'Lasso Regression':
        reg = linear_model.Lasso(alpha=0.1).fit(X_train, y_train) 

    elif models == 'Bayesian Ridge Regression':
        reg = linear_model.BayesianRidge().fit(X_train, y_train)  
    
    if(request.session['type'] == 'classification'):
        clf.fit(X_train, y_train)
        train_pred = clf.predict(X_train)
        train_f1 = str(metrics(y_train,train_pred))
        #plot_prediction(X_train,y_train,train_pred,'Training')
        test_pred = clf.predict(X_test)
        test_f1 = str(metrics(y_test,test_pred))
        #plot_prediction(X_test,y_test,test_pred,'Test')
        report = pd.DataFrame(classification_report(y_test, test_pred, output_dict=True)).to_html(classes="table table-bordered")
        cm = confusion_matrix(y_test, test_pred) 
        df_cm = pd.DataFrame(cm)
        sn.set(font_scale=1.4)
        sns_plot = sn.heatmap(df_cm, annot=True,annot_kws={"size": 16})
        fig = sns_plot.get_figure()
        fig.savefig(r'train/static/images/output.png')

    elif(request.session['type'] == 'regression'):
        train_pred = reg.predict(X_train)
        train_f1 = str(r2_score(y_train,train_pred))
        #plot_prediction(X_train,y_train,train_pred,'Training')
        test_pred = reg.predict(X_test)
        test_f1 = str(r2_score(y_test,test_pred))
        #plot_prediction(X_test,y_test,test_pred,'Test')

    context = {
            'train_f1': train_f1,
            'test_f1': test_f1,
            'models': models,
            'clf_report': report,
            'type': request.session['type'],
        }
    return render(request, 'train/index.html', context)