import os
# please set the following system environment variable before running fact_extractor.py
# 1) 'CLASSPATH' - jar file location to MinIE system
# 2) 'JAVA_HOME' - java-8-openjdk directory path
os.environ['CLASSPATH'] = "/home/tushar/research/source_codes/miniepy/target/minie-0.0.1-SNAPSHOT.jar"
os.environ['JAVA_HOME'] = '/usr/lib/jvm/java-8-openjdk-amd64'
from jnius import autoclass


#minIE relation extraction
class MinIEFactExtractor:
    def __init__(self):
        if 'CLASSPATH' not in os.environ or 'JAVA_HOME' not in os.environ:
            raise Exception("'CLASSPATH' and 'JAVA_HOME' system environment variable not set. Please refer: https://github.com/mmxgn/miniepy")
        self._initialize_core_services()
    
    def _initialize_core_services(self): 
        self.CoreNLPUtils = autoclass('de.uni_mannheim.utils.coreNLP.CoreNLPUtils')
        self.AnnotatedProposition = autoclass('de.uni_mannheim.minie.annotation.AnnotatedProposition')
        self.MinIE = autoclass('de.uni_mannheim.minie.MinIE')
        self.StanfordCoreNLP = autoclass('edu.stanford.nlp.pipeline.StanfordCoreNLP')
        self.String = autoclass('java.lang.String')
        self.Dictionary = autoclass('de.uni_mannheim.utils.Dictionary')
        self.JavaArray = autoclass('java.lang.reflect.Array')

        # Dependency parsing pipeline initialization
        self.parser = self.CoreNLPUtils.StanfordDepNNParser()
        
    def get_triples(self, sentence):
        minie = self.MinIE(self.String(sentence), self.parser, 2)
        triples = []
        for ap in minie.getPropositions().elements():
            if ap is not None:
                triples.append([x[1:-1] for x in ap.getTripleAsString().strip().split('\t')])
        return triples