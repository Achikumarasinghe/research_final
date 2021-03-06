from flask import Flask,request,Response,render_template
from flask_sqlalchemy import SQLAlchemy
import requests
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
import graphviz
from flask import send_file
import zipfile
import io


app =  Flask (__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///E:/SLIIT1/4th year/RESEARCH/Workspace/research_final/filestorage.db'
db = SQLAlchemy(app)

corpus = []
concepts = []
id_list = []
oconcepts = []
stemmer = PorterStemmer()

class finance_text(db.Model):
    name = db.Column(db.String(400),primary_key=True)
    data = db.Column(db.LargeBinary)

class politics_text(db.Model):
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
    try:
        index = tf.vocabulary_[term]
    except Exception:
        return ["N/A"]

    model = np.dot(model,model.T)
 
    closestTerms = {}
    for i in range(len(model)):
        closestTerms[tf.get_feature_names()[i]] = model[index][i]
 
    sortedList = sorted(closestTerms , key= lambda l : closestTerms[l])
 
    return sortedList[::-1][0:k]

def stemming_tokenizer(str_input):
    words = re.sub(r"[^A-Za-z\-]", " ", str_input).split()
    words = [stemmer.stem(word) for word in words]
    #print('words',words)
    return words

@app.route('/suggesion')
def index():
    return render_template('home.html')


@app.route('/storefile',methods = ['POST'])
def storeFile():
    if(request.form['domain']=='finance'):
        file  = request.files['inputfile']
        newFile = finance_text(name=file.filename,data = file.read())
        db.session.add(newFile)
        db.session.commit() 
        return 'Saved : '+ file.filename
    elif(request.form['domain']=='politics'):
        file  = request.files['inputfile']
        newFile = politics_text(name=file.filename,data = file.read())
        db.session.add(newFile)
        db.session.commit() 
        return 'Saved : '+ file.filename

@app.route('/ontologyFinance',methods = ['POST'])
def createFinanceOntology():
    #request json and put nodes in to a lis
    r = requests.get('http://ontology-api-dev.jtzupwqmvj.us-east-1.elasticbeanstalk.com/ontology/get_domain_nodes?domain=Finance').json()
    id_list1 = []
    for n in r['nodes']:
        id_list1.append(n)
    for id in id_list1:
        name = r['nodes'][id]['propertyList'][0]['value']
        oconcepts.append(name)

    for o in oconcepts:
        concepts.append(o)
    

    #connect to DB and get all text file
    conn = sqlite3.connect('filestorage.db')
    cur = conn.cursor()
    cur.execute("SELECT * FROM finance_text")
    rows = cur.fetchall()

    #append them in to corpus
    for row in rows:
        corpus.append((row[0], row[1]))

    #print(corpus)
    tf = TfidfVectorizer(analyzer='word', min_df = 0.25, stop_words = 'english',tokenizer=stemming_tokenizer,token_pattern=r'\b[^\d\W]+\b',ngram_range=(1,2))

    tfidf_matrix =  tf.fit_transform([content for file, content in corpus])

    #print(tfidf_matrix)
    tfidf_matrix_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tf.get_feature_names())
    #print("TFIDF WITH FEATURE NAMES\n",tfidf_matrix_df)
    #print("= ="*20)

    #singular value dicomposition
    svd =  TruncatedSVD(n_components=2,n_iter=5)
    lsa = svd.fit_transform(tfidf_matrix.T)
        
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
        if term1 != "N/A":
            if term1 not in concepts:
                concepts.append(term1)

       
    print('concepts',concepts)
    #get the index
    index_array = []
    concepts_final = []
    new_concepts = []
    
    for y in concepts:
        try:
            index = tf.vocabulary_[y]
            concepts_final.append(y)
        except Exception:
            continue
        index_array.append(index)
    
    for cf in concepts_final:
        if cf not in oconcepts:
            new_concepts.append(cf)

    print('index array',index_array)
    print('concepts_final',concepts_final)
    print('concepts_new',new_concepts)

    rows=[]
    rows_f=[]

    for r in index_array:
        for c in index_array:
            rows.append(sim_matrix[r][c])
            #print(r," - ",c," =row\n",rows)
        rows_f.append(rows)
        #print("aray\n",rows_f)
        rows=[]
    
    rows_m = np.array(rows_f)
    #print("rows_m\n",rows_m)

    final_sim_matrix = np.asmatrix(rows_m)
    print("matrx\n",final_sim_matrix)

    final_sim_matrix_df = pd.DataFrame(final_sim_matrix,columns=concepts_final)
    print("df\n",final_sim_matrix_df)

    #save data frame to csv 
    final_sim_matrix_df.to_csv("dataframe.csv",index=False)

    #call R Script
    command = 'C:/Program Files/R/R-3.5.1/bin/Rscript'
    path2script = 'E:/SLIIT1/4th year/RESEARCH/Workspace/research_final/testr.R'
    #rgs = r_dataframe
    retcode = subprocess.call([command, path2script], shell=True)
    #print(retcode)

    file = open('Igraph.dot', 'r')
    lines = file.readlines()
    if os.path.exists("Igraph1.dot"):
        os.remove("Igraph1.dot")
        

    for line in lines:
        if line != '}\n':
            print(line)
            with open('Igraph1.dot','a') as out:
                out.write(line) 
        else:
            print("hiiii",c)
            with open('Igraph1.dot','a') as out:
                for c in new_concepts:
                    out.write(c+'[color="red"];\n')
                out.write('}') 
    
    
    #graph to pdf
    os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
    path_i = 'Igraph1.dot'
    path_h = 'Hgraph.dot'
    graphviz.render('dot', 'pdf', path_i, quiet=False)
    graphviz.render('dot', 'pdf', path_h, quiet=False)

    zipf = zipfile.ZipFile('Onto.zip','w', zipfile.ZIP_DEFLATED)
    zipf.write('Igraph1.dot.pdf')
    zipf.write('Hgraph.dot.pdf')
    zipf.close()
    return send_file('Onto.zip',mimetype = 'zip',as_attachment = True)
    
@app.route('/ontologyPolitics',methods = ['POST'])
def createPoliticOntology():
    #request json and put nodes in to a lis
    r = requests.get('http://ontology-api-dev.jtzupwqmvj.us-east-1.elasticbeanstalk.com/ontology/get_domain_nodes?domain=Politic').json()
    id_list1 = []
    for n in r['nodes']:
        id_list1.append(n)
    for id in id_list1:
        name = r['nodes'][id]['propertyList'][0]['value']
        oconcepts.append(name)

    for o in oconcepts:
        concepts.append(o)
    

    #connect to DB and get all text file
    conn = sqlite3.connect('filestorage.db')
    cur = conn.cursor()
    cur.execute("SELECT * FROM finance_text")
    rows = cur.fetchall()

    #append them in to corpus
    for row in rows:
        corpus.append((row[0], row[1]))

    #print(corpus)
    tf = TfidfVectorizer(analyzer='word', min_df = 0.25, stop_words = 'english',tokenizer=stemming_tokenizer,token_pattern=r'\b[^\d\W]+\b',ngram_range=(1,2))

    tfidf_matrix =  tf.fit_transform([content for file, content in corpus])

    #print(tfidf_matrix)
    tfidf_matrix_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tf.get_feature_names())
    

    #singular value dicomposition
    svd =  TruncatedSVD(n_components=2,n_iter=5)
    lsa = svd.fit_transform(tfidf_matrix.T)
        
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
        if term1 != "N/A":
            if term1 not in concepts:
                concepts.append(term1)

       
    print('concepts',concepts)
    #get the index
    index_array = []
    concepts_final = []
    new_concepts = []
    
    for y in concepts:
        try:
            index = tf.vocabulary_[y]
            concepts_final.append(y)
        except Exception:
            continue
        index_array.append(index)
    
    for cf in concepts_final:
        if cf not in oconcepts:
            new_concepts.append(cf)

    print('index array',index_array)
    print('concepts_final',concepts_final)
    print('concepts_new',new_concepts)

    rows=[]
    rows_f=[]

    for r in index_array:
        for c in index_array:
            rows.append(sim_matrix[r][c])
            #print(r," - ",c," =row\n",rows)
        rows_f.append(rows)
        #print("aray\n",rows_f)
        rows=[]
    
    rows_m = np.array(rows_f)
    #print("rows_m\n",rows_m)

    final_sim_matrix = np.asmatrix(rows_m)
    print("matrx\n",final_sim_matrix)

    final_sim_matrix_df = pd.DataFrame(final_sim_matrix,columns=concepts_final)
    print("df\n",final_sim_matrix_df)

    #save data frame to csv 
    final_sim_matrix_df.to_csv("dataframe.csv",index=False)

    #call R Script
    command = 'C:/Program Files/R/R-3.5.1/bin/Rscript'
    path2script = 'E:/SLIIT1/4th year/RESEARCH/Workspace/research_final/testr.R'
    #rgs = r_dataframe
    retcode = subprocess.call([command, path2script], shell=True)
    #print(retcode)

    file = open('Igraph.dot', 'r')
    lines = file.readlines()
    if os.path.exists("Igraph1.dot"):
        os.remove("Igraph1.dot")
        

    for line in lines:
        if line != '}\n':
            print(line)
            with open('Igraph1.dot','a') as out:
                out.write(line) 
        else:
            print("hiiii",c)
            with open('Igraph1.dot','a') as out:
                for c in new_concepts:
                    out.write(c+'[color="red"];\n')
                out.write('}') 
    
    
    #graph to pdf
    os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
    path_i = 'Igraph1.dot'
    path_h = 'Hgraph.dot'
    graphviz.render('dot', 'pdf', path_i, quiet=False)
    graphviz.render('dot', 'pdf', path_h, quiet=False)

    zipf = zipfile.ZipFile('Onto.zip','w', zipfile.ZIP_DEFLATED)
    zipf.write('Igraph1.dot.pdf')
    zipf.write('Hgraph.dot.pdf')
    zipf.close()
    return send_file('Onto.zip',mimetype = 'zip',as_attachment = True)
    


if __name__ == '__main__':
    app.run(host='localhost', port=8080,debug=True)



   