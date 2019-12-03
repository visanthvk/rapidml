from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
import os
from automl import settings
import pandas as pd
import numpy as np
import statistics as st
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer

def index(request):
    myfile = request.FILES['myfile']
    file_data = myfile.read().decode("utf-8")
    lines = file_data.split("\n")
    field_names = lines[0].split(",")
    if(myfile):
        request.session['data_set'] = myfile.name
        fs = FileSystemStorage()
        path = os.path.join(settings.BASE_DIR,'media')
        fs = FileSystemStorage(location=f'{path}')
        filename = fs.save(myfile.name, myfile)
        PROJECT_ROOT = os.path.join(settings.BASE_DIR,'media')
        filename = os.path.join(PROJECT_ROOT,f'{filename}')
    context = {
        'headers':field_names,
        'file_name':myfile.name,
    }
    return render(request, 'preprocess/index.html', context)


def preprocess(request):
    if(request.method == 'POST'):
        feature = request.POST.getlist('feature')
        featuress = []
        for i in feature[:-1]:
            featuress.append(i)
        if(feature[-1][-1] == '\n'):
            featuress.append(feature[-1][:-2])
        else:
            featuress.append(feature[-1])
        missing = request.POST.getlist('missing')
        replace = request.POST.getlist('replace')
        is_drop = []
        is_categorical = []
        for key in request.POST:
            if('drop' in key):
                is_drop.append(request.POST.getlist(key)[0])
            if('categorical' in key):
                is_categorical.append(request.POST.getlist(key)[0])
        missing_custom = request.POST.getlist('missing_custom')
        replace_custom = request.POST.getlist('replace_custom')
        df = pd.read_csv(r'media/'+request.session['data_set'])
        features = {}
        m = 0 
        n = 0
        for i in range(len(featuress)):
            temp = [missing[i], replace[i], is_drop[i], is_categorical[i]]
            if(temp[0] == '3'):
                temp.append(missing_custom[m])
                m+=1
            if(temp[1] == '3'):
                temp.append(replace_custom[n])
                n+=1
            features[featuress[i]] =  temp
        '''
        missing

        1 - ?
        2 - Nan
        3 - Custom

        replace

        continous
        1 - Mean
        2- Median
        3 - Custom

        Cate
        1 - Most frequent
        2 - Custom


        encode
        cate

        target - label
        other - onehot

        feature: [missing, replace, , is_drop, is _categorical,(missing custom - opt),replace(custom - opt) ]
        '''
        p_type = "classification"
        if(df.iloc[:,-1].nunique()/df.iloc[:,-1].count() > 0.05):
            p_type="regression"
        request.session['type'] = p_type

        for i in features:
            todo = features[i]
            #drop
            if(todo[2]=='1'):
                del df[i]      
            else:
                #continuous
                if todo[3] == '2':
                    #missing = 'Nan'
                    if todo[0] == '2':
                        #replace with 'mean'
                        if todo[1] == '1':
                            imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
                            imputer = imputer.fit(np.array(df[i]).reshape(-1,1))
                            s = imputer.transform(np.array(df[i]).reshape(-1,1))
                            df[i] = s.ravel()
                        #replace with median    
                        elif todo[1] == '2':
                            imputer = Imputer(missing_values = 'NaN', strategy = 'median', axis = 0)
                            imputer = imputer.fit(np.array(df[i]).reshape(-1,1))
                            s = imputer.transform(np.array(df[i]).reshape(-1,1))
                            df[i] = s.ravel()
                        #replace with custom replace   
                        elif todo[1] == '3':
                            for n,j in enumerate(df[i]):
                                if(np.isnan(j)):
                                    df[i][n] = todo[-1]
                        else:
                            print("Nothing")
                    #missing = '?'
                    elif todo[0] == '1':
                        #replace with 'mean'
                        if todo[1] =='1':
                            repl = st.mean(df[i])
                            for n,j in enumerate(df[i]):
                                    if(j=="?"):
                                        df[i][n] = repl 
                        #replace with 'mediam'
                        elif todo[1] =='2':
                            repl = st.median(df[i])
                            for n,j in enumerate(df[i]):
                                if(j=="?"):
                                    df[i][n] = repl      
                        #replace with custom replace
                        elif todo[1] =='3':
                            for n,j in enumerate(df[i]):
                                if(j=='?'):
                                    df[i][n] = todo[-1]
                        else:
                            print("no replace")
                    #custom missing
                    elif todo[0] == '3':
                        #replace with 'mean'
                        if todo[1] =='1':
                            repl = st.mean(df[i])
                            for n,j in enumerate(df[i]):
                                    if(j==todo[-2]):
                                        df[i][n] = repl    
                        #replace with 'mediam'
                        elif todo[1] =='2':
                            repl = st.median(df[i])
                            for n,j in enumerate(df[i]):
                                if(j==todo[-2]):
                                    df[i][n] = repl       
                        #replace with custom replace
                        elif todo[1] =='3':
                            for n,j in enumerate(df[i]):
                                if(j==todo[-2]):
                                    df[i][n] = todo[-1]
                        else:
                            print("no replace") 
                else:
                    print('categorical')                
                    #missing = 'Nan'
                    if todo[0] == '2':                   
                        #replace with 'most frequent'
                        if todo[1] == '1':
                            imputer = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0)
                            imputer = imputer.fit(np.array(df[i]).reshape(-1,1))
                            s = imputer.transform(np.array(df[i]).reshape(-1,1))
                            df[i] = s.ravel()
                        #replace with custom replace   
                        elif todo[1] == '2':
                            for n,j in enumerate(df[i]):
                                if(np.isnan(j)):
                                    df[i][n] = todo[-1]
                        else:
                            print("Nothing")
                    #missing = '?'
                    elif todo[0] == '1':
                        #replace with 'mode'
                        if todo[1] =='1':
                            repl = st.mode(df[i])
                            for n,j in enumerate(df[i]):
                                    if(j=="?"):
                                        df[i][n] = repl     
                        #replace with custom replace
                        elif todo[1] =='2':
                            for n,j in enumerate(df[i]):
                                if(j=='?'):
                                    df[i][n] = todo[-1]
                        else:
                            print("no replace")
                    #custom missing
                    elif todo[0] == '3':
                        #replace with 'mode'
                        if todo[1] =='1':
                            repl = st.mode(df[i])
                            for n,j in enumerate(df[i]):
                                    if(j==todo[-2]):
                                        df[i][n] = repl         
                        #replace with custom replace
                        elif todo[1] =='2':
                            for n,j in enumerate(df[i]):
                                if(j==todo[-2]):
                                    df[i][n] = todo[-1]
                        else:
                            print("no replace")
                    else:
                        print("nothing") 
                    #encode
                    le = LabelEncoder()
                    le = le.fit(np.array(df[i]).reshape(-1,1))
                    s = le.transform(np.array(df[i]).reshape(-1,1))
                    df[i] = s.ravel()
    df.to_csv(r'media/'+request.session['data_set'], index=False, header=True)
    return redirect('/train')