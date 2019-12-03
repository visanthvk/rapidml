from django.shortcuts import render
from sklearn import svm
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def index(request):
    return render(request, 'home/index.html')

def preprocess(request):
    try:
        csv_file = request.FILES["myfile"]
        print("File Name")
        print(csv_file)
        file_data = csv_file.read().decode("utf-8")
        lines = file_data.split("\n")
        df = []
        field_names = lines[0].split(",")

        for line in lines[1:]:
            fields = line.split(",")
            if(fields[0] != ''):
                df.append(fields)

        head = df[0]
        encoder = []
        for i in range(0,len(head)):
            if(head[i][0].isalpha()):
                encoder.append(i)
    
        df = pd.DataFrame(df)
        label_encoder = LabelEncoder()
        for i in encoder:
            df.iloc[:,i]= label_encoder.fit_transform(df.iloc[:,i])
    
        clf = svm.SVC(gamma='scale')
        clf.fit(df.iloc[:,:-1], df.iloc[:,-1])
        return render(request, 'home/index.html')
    except Exception as ex:
        print(ex)
        context = {
        'error':ex,
        }
        return render(request,'error/index.html',context)