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
verbosity = 1


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


def parseCommandLineArguments():
    parser = argparse.ArgumentParser( prog="textFeaturesExtraction.py" )
    requiredArguments = parser.add_argument_group( "Required arguments" )

    parser.add_argument( "--verbose", "-b", help="Set this to a value from 1 to 5. (5 being most verbose)", default=1 )
    parser.add_argument( "--compute_word_n_grams", "-cwng",
                         help="Prompts the program to calculate word n-grams. Output will be provided both in the form of a pickled generator dump and in the form of a human readable text.",
                         nargs="+" )

    # The next two arguments occur together, they are mutually inclusive
    parser.add_argument( "--compute_word_n_skipgrams", "-cwnsg",
                         help="Prompts the program to calculate word n skip-grams.  Must provide the skips argument as well. Output will be provided both in the form of a pickled generator dump and in the form of a human readable text.",
                         nargs="+" )
    parser.add_argument( "--skipgrams", "-skpgrm", nargs="+",
                         help="Provide the skip values for calculating word skip gram. This argument is mandatory when compute_word_n_skipgrams argument is provided. A total of n*k computations will be done for all possible skipgrams and all possible ngrams " )

    parser.add_argument( "--compute_character_n_grams", "-ccng",
                         help="Prompts the program to calculate character n-grams. Output will be provided both in the form of a pickled generator dump and in the form of a human readable text.",
                         nargs="+" )
    parser.add_argument("--case_insensitive","-no_case",action='store_true',
                        help="Setting this option will perform case insensitive character n grams")
    
    parser.add_argument( "--lowercase", "-lower", action='store_true',
                         help="Converts the text to lowercase and then performs case-insensitive analysis" )
    
    parser.add_argument("--correct_spelling","-spellcheck",action="store_true",
                        help="Set this argument if you wish to perform spelling correction. Spelling correct will be done using enchant package")

    parser.add_argument( "--perform_stemming", "-stem", action='store_true',default=None,
                         help="Setting this argument will enforce stemming of words back to its root words" )
    
    mutex_parser_stemmer = parser.add_mutually_exclusive_group()
    mutex_parser_stemmer.add_argument( "--porter_stemmer", "-p_stem", action='store_true',
                                       help="This will make use of the Porter stemmer algorithm while performing stemming." )
    mutex_parser_stemmer.add_argument( "--lancaster_stemmer", "-l_stem", action='store_true',
                                       help="This will make use of the Lancaster stemmer algorithm while performing stemming." )
    mutex_parser_stemmer.add_argument( "--snowball_stemmer", "-s_stem", action='store_true',
                                       help="This will make use of the Snowball stemmer algorithm while performing stemming." )
    
    parser.add_argument("--lemmatize","-l",action='store_true',
                        help="This option will perform lemmatization on the input data.")
    parser.add_argument("--remove_stop_words","-rsw",action='store_true',help="Setting this option will remove stop words from the provided input data. The program looks up a set of words stored in nltk.")
    parser.add_argument("--tag_parts_of_speech","-tag_pos",action='store_true',
                        help="This option will tag each word in the text with their corresponding parts of speech")
    
    # The next two arguments occur together, they are mutually inclusive
    parser.add_argument("--tag_parts_of_speech_n_grams","-tag_pos_n_grams",nargs="+",
                        help="This option will tag each word and compute n grams based on the POS tag. This introduces a new tag called PCNT which stands for punctuations. This option can accomodate more than one value n at a time.")
    """parser.add_argument("--tagger","-tagger",
                        help="The name of the tagger to be used (brown, conll2000 or treebank ).")"""
    
    # The next two arguments occur together, they are mutually inclusive
    parser.add_argument("--tag_word_and_parts_of_speech_n_grams","-tag_word_pos_n_grams",nargs="+",
                        help="This option will perform n grams with a mixture of words and POS. Please provide a list of POS which will be preserved in the tag. All other POS will be replaced by their corresponding words in the provided text."
                        )
    parser.add_argument("--list_of_POS_preserved","-POS",nargs="+",
                        help="Please provide a comma separated list of POS which you wish to preserve.")
    
    parser.add_argument("--perform_chunking","-chunk",
                        help="Setting this option will perform chunking of the text and put the output in a pickled format")
    parser.add_argument("--perform_dependency_parsing","-parse_depend",action="store_true",
                        help="This argument will perform dependency parsing of the provided text. For this case the Stanford dependency parser has been used. Output will be written in a text file for each sentence which will be separated by the delimiter =====. Each record starts with the sentence followed by its dependency tree.")
    
    # These are groups of arguments which are mandatory
    requiredArguments.add_argument( "--input", "-i", help="Enter the full path of the file which contains the data",
                                    required=True )
    requiredArguments.add_argument( "--output_directory", "-out",
                                    help="Mention the name of the directory where you want all the output files to be put in. Please note that the program will attempt to create the folder in case it is not present. If the folder has some content then those will be overwritten.",
                                    required=True )

    args = parser.parse_args()
    return args


def skipgrams(sequence, n, k):
    # sequence:list of words, n is the ngram size, k is the gap.
    # Source: https://github.com/nltk/nltk/issues/1070
    for ngram in nltk.ngrams( sequence, n + k, pad_right=True ):
        head = ngram[:1]
        tail = ngram[1:]
        for skip_tail in itertools.combinations( tail, n - 1 ):
            if skip_tail[-1] is None:
                continue
            # print(head+skip_tail)
            yield head + skip_tail

def computeWordNSkipGrams(text, n, k_, output_dir):
    tokenizedText = nltk.word_tokenize( text )
    for ngram in n:
        for k in k_:
            if verbosity == 4:
                print( "Starting generation of Word N Skip Grams for ngram=", ngram, "and skip=", k )
            generator_obj = nltk.FreqDist( skipgrams( tokenizedText, int( ngram ), int( k ) ) )
            d = {c: count for c,count in generator_obj.items()}
            d=OrderedDict(sorted(d.items(), key=itemgetter(1),reverse=True))
            pickle.dump( d, open( output_dir + "/word_nskipgram_" + ngram + "_" + k + ".pickle", "wb" ) )
            fhw = open( output_dir + "/word_nskipgram_" + ngram + "_" + k + ".txt", "w" )
            for obj, count in d.items():
                fhw.write( str( obj ) + ":" + str( count ) + "\n" )

def computeWordNGrams(text, n, output_dir):
    tokenizedText = nltk.word_tokenize( text )
    for ngram in n:
        if verbosity == 4:
            print( "Starting generation of Word N grams for ngram=", ngram )
        generator_obj = nltk.FreqDist( ngrams( tokenizedText, int( ngram ) ) )
        
        """for c,count in generator_obj.items():
            print(c,count)"""
        d = {c: count for c,count in generator_obj.items()}
        d=OrderedDict(sorted(d.items(), key=itemgetter(1),reverse=True))
        pickle.dump( d, open( output_dir + "/word_ngram_" + ngram + ".pickle", "wb" ) )
        """for key in d:
            print(key,d[key])"""
        fhw = open( output_dir + "/word_ngram_" + ngram + ".txt", "w" )
        for obj, count in d.items():
            fhw.write( str( obj ) + ":" + str( count ) + "\n" )

def computeCharacterNGrams(text, n, output_dir):
    chars = [char for char in text]
    # tokenizedText = nltk.word_tokenize(text)
    for ngram in n:
        if verbosity == 4:
            print( "Starting generation of character N grams for ngram=", ngram )
        generator_obj = nltk.FreqDist( ngrams( chars, int( ngram ) ) )
        d = {c: count for c,count in generator_obj.items()}
        d=OrderedDict(sorted(d.items(), key=itemgetter(1),reverse=True))
        pickle.dump( d, open( output_dir + "/char_ngram_" + ngram + ".pickle", "wb" ) )
        fhw = open( output_dir + "/char_ngram_" + ngram + ".txt", "w" )
        for obj, count in d.items():
            fhw.write( str( obj ) + ":" + str( count ) + "\n" )

def checkInputFile(filename):
    if os.path.exists( filename ) == False:
        print( "The input file does not exist. Exiting..." )
        sys.exit()
    try:
        fhr = open( filename, "r" )
    except OSError:
        print( "Cannot read input file. Do not have necessary permissions. Exiting..." )
        sys.exit()

class MySpellChecker():

    def __init__(self, dict_name='en_US', max_dist=2):
        self.spell_dict = enchant.Dict(dict_name)
        self.max_dist = max_dist

    def replace(self, word):
        suggestions = self.spell_dict.suggest(word)

        if suggestions:
            for suggestion in suggestions:
                if edit_distance(word, suggestion) <= self.max_dist:
                    return suggestions[0]

        return word

def processInputFile(filename, output_dir, lowercase, stem, lemma, rsw, spellcheck):
    """
    :rtype: object
    """
    fhr = open( filename, "r" )
    text = fhr.read()
    #print(text.split())
    if lowercase:
        text = text.lower()
    if stem:
        print("Performing stemming")
        if stem=="lancaster":
            stemmer = LancasterStemmer()
        elif stem=="porter":
            stemmer = PorterStemmer()
        elif stem=="snowball":
            stemmer=SnowballStemmer("english")
        new_text = " ".join( [stemmer.stem( word ) for word in text.split(" ")] )
        text = new_text
    if lemma:
        print("Performing Lemmatization")
        wordnet_lemmatizer = WordNetLemmatizer()
        new_text = " ".join( [wordnet_lemmatizer.lemmatize( word ) for word in text.split(" ")] )
        text=new_text
    if rsw:
        print("Performing stop word removal")
        stopWords = set( stopwords.words( 'english' ) )
        words = word_tokenize( text )
        wordsFiltered = []
        new_text=" ".join([w for w in words if w not in stopWords])
        text=new_text
    if spellcheck:
        print("Spell checking")
        my_spell_checker = MySpellChecker(max_dist=3)
        chkr = SpellChecker("en_US", text)
        for err in chkr:
            print("Error",err.word,my_spell_checker.replace(err.word))
            err.replace(my_spell_checker.replace(err.word))
        new_text = chkr.get_text()
        text=new_text
        
    name=filename
    if "/" in filename:
        name=filename.split("/")[-1]
    open(output_dir+"/"+name.split(".")[0]+"_modified."+name.split(".")[-1],"w").write(text)
    return text

def tagPartsOfSpeech(text,filename,output_dir=""):
    """
    This function will tag each word in the text with its corresponding Parts of Speech.
    """
    fhr=open(filename,"r")
    tokens=word_tokenize(fhr.read())
    tagged_doc=nltk.pos_tag(tokens)
    #print(tagged_doc[:10])
    """tagged_doc=set(tagged_doc)
    print(len(tagged_doc))"""
    name=filename
    if "/" in filename:
        name=filename.split("/")[-1]
    if output_dir!="":
        fhw=open(output_dir+"/"+name.split(".")[0]+"_POS."+name.split(".")[-1],"w")
        for ele in tagged_doc:
            fhw.write(str(ele))
            if str(ele)=="('.', '.')":
                fhw.write("\n")
        #fhw.write()
        fhw.close()
    return tagged_doc

def tagPOSNGram(text,filename,output_dir,n):
    
    tagged_doc=tagPartsOfSpeech(text,filename,output_dir="")
    #print(tagged_doc)
    tags=[]
    for pair in tagged_doc:
        tags.append(pair[1])
    #print(tags)
    tag_text_punctuation_tagged=tags
    """for word in tags:
        if word not in ["(",")",",",".",":","?","<",">","-","*","&","^","%","$","#","@","!",";","'","\"","...","**","***"]:
            tag_text_punctuation_tagged.append(word)
        else:
            tag_text_punctuation_tagged.append("PCNT")"""
    #print(tag_text)
    tag_text=" ".join(tag_text_punctuation_tagged)
    tokenizedText = nltk.word_tokenize( tag_text )
    for ngram in n:
        generator_obj = nltk.FreqDist( ngrams( tokenizedText, int( ngram ) ) )
        d = {c: count for c,count in generator_obj.items()}
        d=OrderedDict(sorted(d.items(), key=itemgetter(1),reverse=True))
        pickle.dump( d, open( output_dir + "/POS_ngram_" + ngram + ".pickle", "wb" ) )
        fhw = open( output_dir + "/POS_ngram_" + ngram + ".txt", "w" )
        for obj, count in d.items():
            fhw.write( str( obj ) + ":" + str( count ) + "\n" )
        fhw.close()

def performChunking(text,filename,output_dir):
    """r1 = RegexpChunkRule('<a|b>'+ChunkString.IN_CHINK_PATTERN,'{<a|b>}', 'chunk <a> and <b>')
    r2 = RegexpChunkRule(re.compile('<a|b>'+ChunkString.IN_CHINK_PATTERN),'{<a|b>}', 'chunk <a> and <b>')
    r3 = ChunkRule('<a|b>', 'chunk <a> and <b>')
    r4 = ChinkRule('<a|b>', 'chink <a> and <b>')
    r5 = UnChunkRule('<a|b>', 'unchunk <a> and <b>')
    r6 = MergeRule('<a>', '<b>', 'merge <a> w/ <b>')
    r7 = SplitRule('<a>', '<b>', 'split <a> from <b>')
    r8 = ExpandLeftRule('<a>', '<b>', 'expand left <a> <b>')
    r9 = ExpandRightRule('<a>', '<b>', 'expand right <a> <b>')
    for rule in r1, r2, r3, r4, r5, r6, r7, r8, r9:
        print(rule)"""
    grammar = "NP: {<DT>?<JJ>*<NN>}"
    #grammar = "<DT>?<JJ>*<NN>"
    cp = nltk.RegexpParser(grammar)
    
    text=open(filename,"r").read()
    tokens=word_tokenize(text)
    tagged_doc=nltk.pos_tag(tokens)
    result = cp.parse(tagged_doc)
    #print(result)
    #result.draw()
    #print(type(result))
    #print(result.pformat(parens="[]"))
    pickle.dump( result, open( output_dir + "/chunks.pickle", "wb" ) )

def printUPennTagInfo():
    tagdict = load('help/tagsets/upenn_tagset.pickle')
    for tag in tagdict:
        print(tag,tagdict[tag][0],tagdict[tag][1])


def performMixedWordPOSnGram(filename,output_dir,n,POS_preserved):
    text=open(filename,"r").read()
    tagged_doc=tagPartsOfSpeech(text, filename)
    mixed_word_pos=[]
    for pair in tagged_doc:
        if pair[1] in POS_preserved:
            mixed_word_pos.append(pair[1])
        else:
            mixed_word_pos.append(pair[0])
    tokenizedText=nltk.word_tokenize( " ".join(mixed_word_pos) )
    for ngram in n:
        if verbosity == 4:
            print( "Starting generation of Word N grams for ngram=", ngram )
        generator_obj = nltk.FreqDist( ngrams( tokenizedText, int( ngram ) ) )
        d = {c: count for c,count in generator_obj.items()}
        d=OrderedDict(sorted(d.items(), key=itemgetter(1),reverse=True))
        pickle.dump( d, open( output_dir + "/mixed_word_ngram_pos_" + ngram + ".pickle", "wb" ) )
        fhw = open( output_dir + "/mixed_word_ngram_pos_" + ngram + ".txt", "w" )
        for obj, count in d.items():
            fhw.write( str( obj ) + ":" + str( count ) + "\n" )

def performDependencyParsing(filename,output_dir):
    path_to_jar = '/Users/sagnik/Documents/stanford-corenlp-full-2017-06-09/stanford-corenlp-3.8.0.jar'
    path_to_models_jar = '/Users/sagnik/Documents/stanford-corenlp-full-2017-06-09/stanford-corenlp-3.8.0-models.jar'
    path_to_visual_jar = "/Users/sagnik/Documents/stanford-corenlp-full-2017-06-09/dependensee-3.7.0.jar"
    path_to_another_jar = "/Users/sagnik/Documents/stanford-corenlp-full-2017-06-09/slf4j-api.jar"
    dependency_parser = StanfordDependencyParser(path_to_jar=path_to_jar, path_to_models_jar=path_to_models_jar)
    
    text=open(filename,"r").read()
    sent_tokenize_list = sent_tokenize(text)
    fhw=open(output_dir+"/dependency_parsed.txt","w")
    for sentence in sent_tokenize_list:
        fhw.write(sentence)
        fhw.write("\n")
        #print(nltk.word_tokenize( sentence ))
        regex = re.compile(".*?\((.*?)\)")
        if "[" in sentence:
            result = re.findall(regex, sentence)
            sentence=re.sub("[\(\[].*?[\)\]]", "", sentence)
            #print("Removed []",sentence)
        result = dependency_parser.raw_parse(sentence)
        dep = result.__next__()
        result=list(dep.triples())
        for row in result:
            fhw.write(str(row))
            fhw.write("\n")
        #print("="*200)
        fhw.write("=====")
        fhw.write("\n")
    """result = dependency_parser.raw_parse(text)
    dep = result.__next__()
    result=list(dep.triples())
    for row in result:
        print(row)"""
    """cmd="java -cp "+path_to_visual_jar+":"+path_to_jar+":"+path_to_models_jar+":"+path_to_another_jar+" com.chaoticity.dependensee.Main "
    cmd+=text.replace("\n",".").replace("(","").replace(")","")+" "
    cmd+=output_dir+"/test.png"
    os.system(cmd)"""
    
def main():
    global verbosity
    options = parseCommandLineArguments()
    print(options)
    verbosity = options.verbose
    processOutputDirectory( options.output_directory )
    checkInputFile( options.input )
    if options.perform_stemming!=None:
        stemmer="porter"
        if options.porter_stemmer!=None:
            stemmer="porter"
        if options.lancaster_stemmer!=None:
            stemmer="lancaster"
        if options.snowball_stemmer!=None:
            stemmer="snowball"
    else:
        stemmer=None
    file_text = processInputFile( options.input,options.output_directory, options.lowercase, stemmer, options.lemmatize, options.remove_stop_words, options.correct_spelling )

    if options.lowercase == None:
        options.lowercase = False
    if options.compute_word_n_grams != None:
        computeWordNGrams( file_text, options.compute_word_n_grams, options.output_directory )
    if options.compute_character_n_grams != None:
        computeCharacterNGrams( file_text, options.compute_word_n_grams, options.output_directory )
    if options.compute_word_n_skipgrams != None and options.skipgrams == None:
        print( 'You must provide the skip argument alongwith the skipgrams' )
        sys.exit()
    if options.compute_word_n_skipgrams == None and options.skipgrams != None:
        print( 'You must not provide the skip argument without specifying the skipgrams argument' )
        sys.exit()
    if options.compute_word_n_skipgrams != None and options.skipgrams != None:
        computeWordNSkipGrams( file_text, options.compute_word_n_skipgrams, options.skipgrams,
                               options.output_directory )
    if options.tag_parts_of_speech != None:
        tagPartsOfSpeech(file_text,options.input,options.output_directory)
    if options.tag_parts_of_speech_n_grams!=None:
        """if options.tagger=="None":
            print("You must provide the name of a tagger")
            sys.exit()
        if options.tagger!="brown" and options.tagger!="conll2000" and options.tagger!="treebank":
            print("You must provide one of the acceptable taggers")"""
        tagPOSNGram(file_text,options.input,options.output_directory,options.tag_parts_of_speech_n_grams)
    if options.perform_chunking != None:
        performChunking(file_text,options.input,options.output_directory)
    
    if options.tag_word_and_parts_of_speech_n_grams !=None :
        if options.list_of_POS_preserved==None:
            print("You must provide a list of POS to be preserved. Please choose one or more from the list below.")
            printUPennTagInfo()
            sys.exit()
        performMixedWordPOSnGram(options.input,options.output_directory,options.tag_word_and_parts_of_speech_n_grams,options.list_of_POS_preserved)
    if options.perform_dependency_parsing!=False:
        performDependencyParsing(options.input,options.output_directory)

if __name__ == "__main__":
    main()
