from flask import Flask,request
from flask_sqlalchemy import SQLAlchemy
import sqlite3
import sys
import os
from collections import Counter
import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import numpy as np 
from nltk.stem.porter import PorterStemmer
from sklearn import metrics
import pandas as pd
import re
from string import *
from sklearn.preprocessing import Normalizer
import subprocess
from graphviz import Source
import binascii


app =  Flask (__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///E:/SLIIT1/4th year/RESEARCH/Workspace/research_final/filestorage.db'
db = SQLAlchemy(app)

corpus = []
stemmer = PorterStemmer()

class filesSave(db.Model):
    name = db.Column(db.String(400),primary_key=True)
    data = db.Column(db.LargeBinary)

def getClosestTerm(term,transformer,model):
 
    term = stemmer.stem(term)
    index = transformer.vocabulary_[term]      
 
    model = np.dot(model,model.T)
    searchSpace =np.concatenate( (model[index][:index] , model[index][(index+1):]) )  
 
    out = np.argmax(searchSpace)
 
    if out<index:
        return transformer.get_feature_names()[out]
    else:
        return transformer.get_feature_names()[(out+1)]

#kcloset term
def kClosestTerms(k,term,tf,model):
 
    term = stemmer.stem(term)
    index = tf.vocabulary_[term]
 
    model = np.dot(model,model.T)
 
    closestTerms = {}
    for i in range(len(model)):
        closestTerms[tf.get_feature_names()[i]] = model[index][i]
 
    sortedList = sorted(closestTerms , key= lambda l : closestTerms[l])
 
    return sortedList[::-1][0:k]

@app.route('/')
def index():
    return 'hi'

@app.route('/storefile',methods = ['POST'])
def storeFile():
    file  = request.files['inputfile']
    newFile = filesSave(name=file.filename,data = file.read())
    db.session.add(newFile)
    db.session.commit() 
    return 'Saved : '+ file.filename

@app.route('/ontology',methods = ['POST'])
def createOntology():
    conn = sqlite3.connect('filestorage.db')
    cur = conn.cursor()
    cur.execute("SELECT * FROM files_save")
    rows = cur.fetchall()
    
    for row in rows:
        corpus.append((row[0], row[1]))
        #print(row[0], str(row[1]),'ascii')

    print(corpus)
    tf = TfidfVectorizer(analyzer='word', min_df = 0, stop_words = 'english')

    tfidf_matrix =  tf.fit_transform([content for file, content in corpus])

    print(tfidf_matrix)
    tfidf_matrix_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tf.get_feature_names())
    print("TFIDF WITH FEATURE NAMES\n",tfidf_matrix_df)
    print("= ="*20)

    svd =  TruncatedSVD(n_components=2,n_iter=5)
    lsa = svd.fit_transform(tfidf_matrix.T)

    a = getClosestTerm("asset",tf,lsa)

    concepts = ["fund","price","asset","global"]

    sim_matrix = np.dot(lsa,lsa.T)

    kterms_f = []
    for i in concepts:
        kterms = kClosestTerms(3,i,tf,lsa)
        for term in kterms:
            if term not in kterms_f:
                kterms_f.append(term)
        print(kterms)
    
    print(kterms_f)
    #append them in to ontology array
    for term1 in kterms_f:
        if term1 not in concepts:
            concepts.append(term1)

    print(concepts)
    #get the index
    index_array = []
    for y in concepts:
        index = tf.vocabulary_[y]
        index_array.append(index)

    print(index_array)

    rows=[]
    rows_f=[]

    for r in index_array:
        for c in index_array:
            rows.append(sim_matrix[r][c])
            print(r," - ",c," =rows\n",rows)
        rows_f.append(rows)
        print("aaray\n",rows_f)
        rows=[]
    
    rows_m = np.array(rows_f)
    print("rows_m\n",rows_m)

    final_sim_matrix = np.asmatrix(rows_m)
    print("matirx\n",final_sim_matrix)

    final_sim_matrix_df = pd.DataFrame(final_sim_matrix,columns=concepts)
    print("df\n",final_sim_matrix_df)

    #save data frame to csv
    final_sim_matrix_df.to_csv("dataframe.csv",index=False)

    #call R Script
    command = 'C:/Program Files/R/R-3.5.1/bin/Rscript'
    path2script = 'E:/SLIIT1/4th year/RESEARCH/Workspace/research_final/testr.R'
    #rgs = r_dataframe
    retcode = subprocess.call([command, path2script], shell=True)

    #geaph to pdf
    os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
    path = 'Igraph.dot'
    s = Source.from_file(path)
    #s.view()
    return a
    


if __name__ == '__main__':
    app.run(debug=True)



   