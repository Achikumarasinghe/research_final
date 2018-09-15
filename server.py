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

class filesSave(db.Model):
    name = db.Column(db.String(400),primary_key=True)
    data = db.Column(db.LargeBinary)

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

    
    return "oks"
    


if __name__ == '__main__':
    app.run(debug=True)



   