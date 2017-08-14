from __future__ import print_function
from __future__ import division
import sys
import os
import matplotlib.pyplot as plt
import argparse
import nltk, itertools
from nltk import ngrams
import pickle
from itertools import islice, chain, combinations
from nltk.stem import *
from nltk.stem.porter import *
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import brown
from nltk import NgramTagger
from nltk import treebank
from nltk.chunk import *
from nltk.chunk.util import *
from nltk.chunk.regexp import *
from nltk import Tree
from nltk.draw.tree import TreeView
from nltk.data import load

from nltk.parse.stanford import StanfordParser
from nltk.parse.stanford import StanfordDependencyParser
from nltk.parse.stanford import StanfordNeuralDependencyParser
from nltk.tag.stanford import StanfordPOSTagger, StanfordNERTagger
from nltk.tokenize.stanford import StanfordTokenizer
import enchant
from enchant.checker import SpellChecker
from nltk.metrics.distance import edit_distance
from nltk.tokenize import sent_tokenize
import re
from collections import OrderedDict
from operator import itemgetter
import gensim
import glob
import numpy as np

verbosity=1

def processOutputDirectory(directory_path):
    if os.path.exists( directory_path ):
        if verbosity == 2:
            print( "The requested output directory already exists. Contents will be overwritten." )
    else:
        try:
            os.system( "mkdir " + directory_path )
        except OSError:
            print( "Do not have permissions to create directory at the location requested. Exiting..." )
            sys.exit()

def processInputDirectory(dir):
    if os.path.exists(dir)==False:
        print("Please enter a directory that exists")
        sys.exit()
    else:
        all_data_files=glob.glob(dir+"/*.txt")
        if len(all_data_files)==0:
            print("Please make sure there is at least one data file in the input directory")
            sys.exit()
    #print(all_data_files)
    return all_data_files

def readAllDataFiles(all_data_files):
    texts=[]
    for filename in all_data_files:
        texts.append(open(filename,"r").read())
    return texts

def generateWordToVectorRepresentation(all_data_files,c_text,wtvmodel,cpu,win,min_cnt,f,iterations,method,out):
    model = gensim.models.Word2Vec(iter=iterations,size=f, window=win, min_count=min_cnt, workers=cpu)
    if wtvmodel==None:
        if c_text==None:
            text=" ".join(readAllDataFiles(all_data_files))
        else:
            text=open(c_text,"r").read()
        #print(text)
        sent_tokenize_list = sent_tokenize(text)
        sentences=[]
        for sentence in sent_tokenize_list:
            sentences.append(sentence.split())
        model.build_vocab(sentences)  
        model.train(sentences,total_examples=model.corpus_count,epochs=model.iter)
        model.save(out+"/wtv_model.txt")
    else:
        model=gensim.models.Word2Vec.load(wtvmodel)
    
    if c_text==None:
        all_texts=readAllDataFiles(all_data_files)
        for num,text in enumerate(all_texts):
            outfilename=out+"/"+all_data_files[num].split("/")[-1].split(".txt")[0]+"_word2vec.txt"
            vectors=[]
            for word in text.split():
                try:
                    vectors.append(np.array(model[word]))
                except KeyError:
                    continue
            vectors=np.array(vectors)
            if method=="sum":
                doc_feature_vector=np.sum(vectors,axis=0)
            else:
                doc_feature_vector=np.mean(vectors,axis=0)
            #print(doc_feature_vector)
            fhw=open(outfilename,"w")
            for val in doc_feature_vector:
                fhw.write(str(val)+" ")
            fhw.close()

def computeVectorFromFile(filename,method,f):
    fhr=open(filename,"r")
    vectors=[]
    for line in fhr:
        try:
            vectors.append(np.array([float(val) for val in line.split()[-f:]]))
        except ValueError:
            print(line)
            sys.exit()
    vectors=np.array(vectors)
    if method=="sum":
        return np.sum(vectors,axis=0)
    else:
        return np.mean(vectors,axis=0)
 
def generateWordToVectorRepresentationFastText(all_data_files,corpus_filename,input_dir,word2vec_fasttext_model,p,win,min_cnt,f,iter,skipgram_cbow,method,out):
    if skipgram_cbow!="cbow":
        skipgram_cbow="skipgram"
    if word2vec_fasttext_model==None:
        os.system("../fastText/fasttext "+skipgram_cbow+" -input "+corpus_filename+" -output "+out+"/fasttext_"+skipgram_cbow+"_model -minCount "+
              str(min_cnt)+" -dim "+str(f)+" -ws "+str(win)+" -epoch "+str(iter)+" -thread "+str(p)
              )
        modelfilename=out+"/fasttext_"+skipgram_cbow+"_model"
    else:
        modelfilename=word2vec_fasttext_model
    all_input_filenames=glob.glob(input_dir+"/*txt")
    for num,input_filename in enumerate(all_input_filenames):
        os.system("../fastText/fasttext print-word-vectors "+modelfilename+".bin < "+input_filename+" > "+out+"/temp.txt")
        feature_vector=computeVectorFromFile(out+"/temp.txt",method,f)
        fhw=open(out+"/"+all_data_files[num].split("/")[-1].split(".txt")[0]+"_word2vec_fasttext.txt","w")
        for val in feature_vector:
            fhw.write(str(val)+" ")
        fhw.close()

def parseCommandLineArguments():
    parser = argparse.ArgumentParser( prog="embeddingFeaturesExtraction.py" )
    requiredArguments = parser.add_argument_group( "Required arguments" )
    
    parser.add_argument("--processors","-p",help="Please enter the number of processors you wish to designate for this task",
                        default=1)
    
    
    # Arguments for word to vectors
    parser.add_argument("--word_to_vector","-wtv",help="Will perform word to vector conversion using gensim",default=0)
    parser.add_argument("--window","-w",help="Please enter the size of window for word2vec", default=3)
    parser.add_argument("--minimum_count","-min_cnt",help="Please enter the minimum count desired",default=5)
    parser.add_argument("--num_of_features","-f",help="Please enter the desired number of features",default=10)
    parser.add_argument("--num_iterations","-iter",help="Please enter the number of desired iterations for constructing the corpus. Also note that increasing the number of iterations will increase training time a lot.",default=5)
    parser.add_argument("--word2vec_to_feature_method","-m",help="Please enter sum if you wish to sum up the vectors. Any other argument entered will result in mean.",default="mean")
    parser.add_argument("--corpus_text","-c_text",help="Enter the name of the file which will be used to build the corpus. If no file is provided, then all the input data files will be used for constructing the corpus.")
    parser.add_argument("--word2vec_model","-wtvmodel",help="Provide the path to the trained word to vector model instead of the corpus file. If this option is provided the corpus training will be ignored.")
    
    parser.add_argument("--word_to_vector_fasttext","-wtv_fasttext",help="This will use fastText to form features",default=0)
    parser.add_argument("--skipgram_cbow","-skip_cbow",help="Chose either skipgram or cbow",default="skipgram")
    parser.add_argument("--word2vec_fasttext_model","-wtvmodel_fasttext",help="Provide the path to the trained word to vector model instead of the corpus file. If this option is provided the corpus training will be ignored.")
    parser.add_argument("--word2vec_fasttext_to_feature_method","-m_fasttext",help="Please enter sum if you wish to sum up the vectors. Any other argument entered will result in mean.",default="mean")
    
    # These are groups of arguments which are mandatory
    requiredArguments.add_argument( "--input_dir", "-inp_dir", help="Enter the full path of the directory which contains .txt files. All .txt files under this directory will be considered to be a potential data",
                                    required=True )
    requiredArguments.add_argument( "--output_directory", "-out_dir",
                                    help="Mention the name of the directory where you want all the output files to be put in. Please note that the program will attempt to create the folder in case it is not present. If the folder has some content then those will be overwritten. Make sure this folder is not the same as the input folder. The program can output some data with a .txt extension.",
                                    required=True )
    return parser.parse_args()
        
def main():
    global verbosity
    options = parseCommandLineArguments()
    print(options)
    processOutputDirectory( options.output_directory )
    all_data_files=processInputDirectory(options.input_dir)
    if options.word_to_vector=='1':
        generateWordToVectorRepresentation(all_data_files,options.corpus_text,options.word2vec_model,int(options.processors),int(options.window),int(options.minimum_count),int(options.num_of_features),int(options.num_iterations),options.word2vec_to_feature_method,options.output_directory)
    if options.word_to_vector_fasttext=="1":
        if options.corpus_text!=None:
            corpus_filename=options.corpus_text
        else:
            corpus_filename=options.output_directory+"/corpus.txt"
            open(corpus_filename,"w").write(" ".join(all_data_files))
        generateWordToVectorRepresentationFastText(all_data_files,corpus_filename,options.input_dir,options.word2vec_fasttext_model,int(options.processors),int(options.window),int(options.minimum_count),int(options.num_of_features),int(options.num_iterations),options.skipgram_cbow,options.word2vec_fasttext_to_feature_method,options.output_directory)
        

if __name__ == "__main__":
    main()