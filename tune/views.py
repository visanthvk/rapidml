from django.shortcuts import render
from sklearn import svm
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

def index(request):
    context = {
        'model': request.session['model']
    }
    return render(request, 'tune/index.html', context)

def metrics(y_true,y_pred):
    from sklearn.metrics import f1_score
    f1 = f1_score(y_true, y_pred, average='macro')
    return f1

def tune(request):
    model = request.session['model']
    if(model == 'SVM'):
        clf = svm.SVC(kernel = request.POST['kernel'], C = float(request.POST['cparam']), gamma=float(request.POST['gamma']))
    elif model == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=request.POST['k'])
    elif model == 'Decision Tree':
        clf = DecisionTreeClassifier(random_state=0, criterion = request.POST['criterion'])
    elif model == 'Logistic Regression':
        clf = LogisticRegression(C = request.POST['cparam'], max_iter = request.POST['iterations'])
    elif model == 'Linear Regression':
        reg = linear_model.LinearRegression()
    
    df = pd.read_csv(r'media/'+request.session['data_set'])
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.6)

    if(request.session['type'] == 'classification'):
        clf.fit(X_train, y_train)
        train_pred = clf.predict(X_train)
        train_f1 = str(metrics(y_train,train_pred))
        #plot_prediction(X_train,y_train,train_pred,'Training')
        test_pred = clf.predict(X_test)
        test_f1 = str(metrics(y_test,test_pred))
        #plot_prediction(X_test,y_test,test_pred,'Test')
        report = pd.DataFrame(classification_report(y_test, test_pred, output_dict=True)).to_html(classes="table table-bordered")
    else:
        reg.fit(X_train, y_train)
        train_pred = reg.predict(X_train)
        train_f1 = str(r2_score(y_train,train_pred))
        #plot_prediction(X_train,y_train,train_pred,'Training')
        test_pred = reg.predict(X_test)
        test_f1 = str(r2_score(y_test,test_pred))
        #plot_prediction(X_test,y_test,test_pred,'Test')

    context = {
        'model': model,
        'train_f1': train_f1,
        'test_f1': test_f1,
        'clf_report': report,
        'type': request.session['type'],
    }
    return render(request, 'tune/index.html', context)