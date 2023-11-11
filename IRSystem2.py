import json
import math
import os
import re
import sys
import heapq
from PorterStemmer import PorterStemmer

class IRSystem:

    def __init__(self):
        self.titles = []
        self.docs = []
        self.vocab = []
        self.alphanum = re.compile('[^a-zA-Z0-9]')
        self.p = PorterStemmer()

    def get_uniq_words(self):
        uniq = set()
        for doc in self.docs:
            for word in doc:
                uniq.add(word)
        return uniq

    def __read_raw_data(self, dirname):
        print ("Stemming Documents...")
        titles = []
        docs = []
        os.mkdir('%s/stemmed' % dirname)
        title_pattern = re.compile('(.*) \d+\.txt')
        filenames = []
        for filename in os.listdir('%s/raw' % dirname):
            if filename.endswith(".txt") and not filename.startswith("."):
                filenames.append(filename)

        for i, filename in enumerate(filenames):
            title = title_pattern.search(filename).group(1)
            print ("    Doc %d of %d: %s" % (i+1, len(filenames), title))
            titles.append(title)
            contents = []
            f = open('%s/raw/%s' % (dirname, filename), 'r')
            of = open('%s/stemmed/%s.txt' % (dirname, title), 'w')
            for line in f:
                line = line.lower()
                line = [xx.strip() for xx in line.split()]
                line = [self.alphanum.sub('', xx) for xx in line]
                line = [xx for xx in line if xx != '']
                line = [self.p.stem(xx) for xx in line]
                contents.extend(line)
                if len(line) > 0:
                    of.write(" ".join(line))
                    of.write('\n')
            f.close()
            of.close()
            docs.append(contents)
        return titles, docs

    def __read_stemmed_data(self, dirname):
        print ("Already stemmed!")
        titles = []
        docs = []
        filenames = []
        for filename in os.listdir('%s/stemmed' % dirname):
            if filename.endswith(".txt") and not filename.startswith("."):
                filenames.append(filename)
        if len(filenames) != 60:
            msg = "There are not 60 documents in ../data/RiderHaggard/stemmed/\n"
            msg += "Remove ../data/RiderHaggard/stemmed/ directory and re-run."
            raise Exception(msg)

        for i, filename in enumerate(filenames):
            title = filename.split('.')[0]
            titles.append(title)
            contents = []
            f = open('%s/stemmed/%s' % (dirname, filename), 'r')
            for line in f:
                line = [xx.strip() for xx in line.split()]
                contents.extend(line)
            f.close()
            docs.append(contents)
        return titles, docs

    def read_data(self, dirname):
        print ("Reading in documents...")
        filenames = os.listdir(dirname)
        subdirs = os.listdir(dirname)
        if 'stemmed' in subdirs:
            titles, docs = self.__read_stemmed_data(dirname)
        else:
            titles, docs = self.__read_raw_data(dirname)
        ordering = [idx for idx, title in sorted(enumerate(titles),
            key = lambda xx : xx[1])]
        self.titles = []
        self.docs = []
        numdocs = len(docs)
        for d in range(numdocs):
            self.titles.append(titles[ordering[d]])
            self.docs.append(docs[ordering[d]])
        self.vocab = [xx for xx in self.get_uniq_words()]

    def compute_tfidf(self):
        print ("Calculating tf-idf...")
        self.tfidf = {}
        N = len(self.docs)
        for word in self.vocab: 
            if word not in self.tfidf: 
                self.tfidf[word] = {} 
            idf = math.log10(N*1./len(self.inv_index[word]))
            for index,d in enumerate(self.inv_index[word]):
                tf = math.log10(1.*len(self.inv_index[word][d]))
                self.tfidf[word][d] = (1+tf)*idf
        tfidf_l2norm2 = {}
        for word, d_dict in self.tfidf.items():
            for d,val in d_dict.items():
                tfidf_l2norm2[d] = tfidf_l2norm2.get(d, 0.0) + val ** 2
        self.tfidf_l2norm = dict((k,math.sqrt(v)) for k,v in tfidf_l2norm2.items())               

    def get_tfidf(self, word, document):
        if self.tfidf[word][document] is not None: 
            return self.tfidf[word][document]
        else:
            return 0

    def get_tfidf_unstemmed(self, word, document):
        word = self.p.stem(word)
        return self.get_tfidf(word, document)

    def index(self):
        print ("Indexing...")
        inv_index = {}
        for i,title in enumerate(self.titles):
            for j,word in enumerate(self.docs[i]):
                if not word in inv_index:
                    inv_index[word] = {}
                if not i in inv_index[word]:
                    inv_index[word][i] = []
                inv_index[word][i].append(j)
        self.inv_index = inv_index

    def get_posting(self, word):
        posting = self.inv_index[word].keys()
        return posting
        
    def get_posting_unstemmed(self, word):
        word = self.p.stem(word)
        return self.get_posting(word)

    def boolean_retrieve(self, query):
        docs = []
        for d in range(len(self.docs)):
            docs.append(d)
        docsets = set(docs)
        for q in query:
            docsets &= set(self.inv_index[q].keys())
        docs = list(docsets)
        return docs   

    def rank_retrieve(self, query):
        scores = [0.0 for xx in range(len(self.docs))]
        wordvec = {}
        for word in query:
            wordvec[word] = wordvec.get(word,0) + 1
        wordvec = dict((word, math.log10(wordvec[word])+1.) for word in wordvec)
        def get_score(d):
            d_vec = dict((word, self.tfidf[word].get(d,0.0)) for word in wordvec)    
            return sum(wordvec[word] * d_vec[word] for word in d_vec)/self.tfidf_l2norm[d]
            scores = []
        for d in range(len(self.docs)):
            heapq.heappush(scores, (get_score(d), d))
        return [(k,v) for v,k in heapq.nlargest(10,scores)]

    def process_query(self, query_str):
        query = query_str.lower()
        query = query.split()
        query = [self.alphanum.sub('', xx) for xx in query]
        query = [self.p.stem(xx) for xx in query]
        return query

    def query_retrieve(self, query_str):
        query = self.process_query(query_str)
        return self.boolean_retrieve(query)

    def query_rank(self, query_str):
        query = self.process_query(query_str)
        return self.rank_retrieve(query)

def run_tests(irsys):
    print ("===== Running tests =====")
    ff = open('C:/Users/hp 850/Desktop/queries.txt')
    questions = [xx.strip() for xx in ff.readlines()]
    ff.close()
    ff = open('C:/Users/hp 850/Desktop/solutions.txt')
    solutions = [xx.strip() for xx in ff.readlines()]
    ff.close()
    epsilon = 1e-4
    for part in range(4):
        points = 0
        num_correct = 0
        num_total = 0
        prob = questions[part]
        soln = json.loads(solutions[part])
        if part == 0:     
            print ("Inverted Index Test")
            words = prob.split(", ")
            for i, word in enumerate(words):
                num_total += 1
                posting = irsys.get_posting_unstemmed(word)
                if set(posting) == set(soln[i]):
                    num_correct += 1
        elif part == 1:   
            print ("Boolean Retrieval Test")
            queries = prob.split(", ")
            for i, query in enumerate(queries):
                num_total += 1
                guess = irsys.query_retrieve(query)
                if set(guess) == set(soln[i]):
                    num_correct += 1
        elif part == 2:   
            print ("TF-IDF Test")
            queries = prob.split("; ")
            queries = [xx.split(", ") for xx in queries]
            queries = [(xx[0], int(xx[1])) for xx in queries]
            for i, (word, doc) in enumerate(queries):
                num_total += 1
                guess = irsys.get_tfidf_unstemmed(word, doc)
                if guess >= float(soln[i]) - epsilon and \
                        guess <= float(soln[i]) + epsilon:
                    num_correct += 1
        elif part == 3:   
            print ("Cosine Similarity Test")
            queries = prob.split(", ")
            for i, query in enumerate(queries):
                num_total += 1
                ranked = irsys.query_rank(query)
                top_rank = ranked[0]
                if top_rank[0] == soln[i][0]:
                    if top_rank[1] >= float(soln[i][1]) - epsilon and \
                            top_rank[1] <= float(soln[i][1]) + epsilon:
                        num_correct += 1

        feedback = "%d/%d Correct. Accuracy: %f" % \
                (num_correct, num_total, float(num_correct)/num_total)
        if num_correct == num_total:
            points = 3
        elif num_correct > 0.75 * num_total:
            points = 2
        elif num_correct > 0:
            points = 1
        else:
            points = 0
        print ("    Score: %d Feedback: %s" % (points, feedback))

def main(args):
    irsys = IRSystem()
    irsys.read_data('C:/Users/hp 850/Desktop/Data/RiderHaggard')
    irsys.index()
    irsys.compute_tfidf()
    if len(args) == 0:
        run_tests(irsys)
    else:
        query = " ".join(args)
        print ("Best matching documents to '%s':" % query)
        results = irsys.query_rank(query)
        for docId, score in results:
            print ("%s: %e" % (irsys.titles[docId], score))
if __name__ == '__main__':
    args = sys.argv[1:]
    main(args)