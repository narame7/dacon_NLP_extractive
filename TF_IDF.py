from utils import *
from math import log10
from kogpt2.pytorch_kogpt2 import get_pytorch_kogpt2_model
from gluonnlp.data import SentencepieceTokenizer
from kogpt2.utils import get_tokenizer


class TF(object):

    def __init__(self):
        self.tok_path = get_tokenizer()
        self.tok = SentencepieceTokenizer( self.tok_path, num_best=0, alpha=0)

    def getlist(self,data_path):

        data= load_jsonl(data_path)
        d=[]
        mem=[]
        for i in range(len(data)):
            d.append(data[i]['article_original'])
        for i in d:
            mem.append(self.tfidfScorer(i))

        result=self.score(mem)
        result=self.get_index(result)

        return result


    def tfidf(self,t, d, D):
        return self.tf(t,d)*self.idf(t, D)

    def tf(self,t, d):
        return 0.5 + 0.5*self.f(t,d)/max([self.f(w,d) for w in d])

    def idf(self,t, D):
        # D is documents == document list
        numerator = len(D)
        denominator = 1 + len([ True for d in D if t in d])
        return log10(numerator/denominator)

    def f(self,t, d):

        return d.count(t)


    def tokenizer(self,d):
        print(d)
        return self.tok(d)

    def tfidfScorer(self,D):
        tokenized_D = [self.tokenizer(d) for d in D]

        result = []
        for d in tokenized_D:
            result.append([(self.tfidf(t, d, tokenized_D)) for t in d])
        return result

    def score(self,D):

        result=[]

        for i in D:
            mem=[]
            for j in i:
                mem.append(np.array(j).sum())

            result.append(mem)

        return np.array(result)

    def get_index(self,D):
        result=[]

        for i in D:
            target=np.argsort(np.array(i))[-3:]
            target=np.sort(target)
            strings=str(target.item(0))+'\n'+str(target.item(1))+'\n'+str(target.item(2))
            result.append(strings)


        return result