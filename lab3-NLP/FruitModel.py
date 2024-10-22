import math
import random
from SST_2.dataset import traindataset, minitraindataset
from fruit import get_document, tokenize
import pickle
import numpy as np
from importlib.machinery import SourcelessFileLoader
from autograd.BaseGraph import Graph
from autograd.BaseNode import *
np.random.seed(0)
random.seed(0)

class NullModel:
    def __init__(self):
        pass

    def __call__(self, text):
        return 0


class NaiveBayesModel:
    def __init__(self):
        self.dataset = traindataset(shuffle=False) # 完整训练集，需较长加载时间
        #self.dataset = minitraindataset(shuffle=False) # 用来调试的小训练集，仅用于检查代码语法正确性

        # 以下内容可根据需要自行修改，不修改也可以完成本题
        self.token_num = [{}, {}] # token在正负样本中出现次数
        self.V = 0 #语料库token数量
        self.pos_neg_num = [0, 0] # 正负样本数量
        self.count()

    def count(self):
        # TODO: YOUR CODE HERE
        # 提示：统计token分布不需要返回值
        for text, label in self.dataset:
            self.pos_neg_num[label] += 1
            for token in text:
                if token not in self.token_num[label]:
                    self.token_num[label][token] = 1
                else:
                    self.token_num[label][token] += 1
                self.V += 1

    def __call__(self, text):
        # TODO: YOUR CODE HERE
        # 返回1或0代表当前句子分类为正/负样本
        alpha = 0.01
        prob = [math.log(self.pos_neg_num[0]/(self.pos_neg_num[0]+self.pos_neg_num[1])), math.log(self.pos_neg_num[1]/(self.pos_neg_num[0]+self.pos_neg_num[1]))]
        for token in text:
            for i in range(2):
                if token in self.token_num[i]:
                    prob[i] += math.log((self.token_num[i][token]+alpha)/(alpha*self.V+len(self.token_num[i])))
                else:
                    prob[i] += math.log(alpha/(alpha*self.V+len(self.token_num[i])))
        return 1 if prob[1]>prob[0] else 0


def buildGraph(dim, num_classes, L): #dim: 输入一维向量长度, num_classes:分类数
    # 以下类均需要在BaseNode.py中实现
    # 也可自行修改模型结构
    nodes = [Attention(dim), relu(), LayerNorm((L, dim)), ResLinear(dim), relu(), LayerNorm((L, dim)), Mean(1), Linear(dim, num_classes), LogSoftmax(), NLLLoss(num_classes)]
    
    graph = Graph(nodes)
    return graph


save_path = "model/attention.npy"

class Embedding():
    def __init__(self):
        self.emb = dict() 
        with open("words.txt", encoding='utf-8') as f: #word.txt存储了每个token对应的feature向量，self.emb是一个存储了token-feature键值对的Dict()，可直接调用使用
            for i in range(50000):
                row = next(f).split()
                word = row[0]
                vector = np.array([float(x) for x in row[1:]])
                self.emb[word] = vector
        
    def __call__(self, text, max_len=50):
        # TODO: YOUR CODE HERE
        # 利用self.emb将句子映射为一个二维向量（LxD），注意，同时需要修改训练代码中的网络维度部分
        # 默认长度L为50，特征维度D为100
        # 提示: 考虑句子如何对齐长度，且可能存在空句子情况（即所有单词均不在emd表内） 
        emb_text = np.zeros((max_len, 100))
        for i, word in enumerate(text):
            if i >= max_len:
                break
            if word in self.emb:
                emb_text[i] = self.emb[word]
        return emb_text


class AttentionModel():
    def __init__(self):
        self.embedding = Embedding()
        with open(save_path, "rb") as f:
            self.network = pickle.load(f)
        self.network.eval()
        self.network.flush()

    def __call__(self, text, max_len=50):
        X = self.embedding(text, max_len)
        X = np.expand_dims(X, 0)
        pred = self.network.forward(X, removelossnode=1)[-1]
        haty = np.argmax(pred, axis=-1)
        return haty[0]


class QAModel():
    def __init__(self):
        self.document_list = get_document()

    def tf(self, word, document):
        # TODO: YOUR CODE HERE
        # 返回单词在文档中的频度
        # document变量结构请参考fruit.py中get_document()函数
        word_count = document['document'].count(word)
        total_words = len(document['document'])
        return word_count/total_words if total_words != 0 else 0
    
    def idf(self, word):
        # TODO: YOUR CODE HERE
        # 返回单词IDF值，提示：你需要利用self.document_list来遍历所有文档
        # 注意python整除与整数除法的区别
        num_documents = len(self.document_list)
        num_documents_with_word = sum(1 for document in self.document_list if word in document['document'])
        return math.log10(num_documents/(num_documents_with_word+1))
    
    def tfidf(self, word, document):
        # TODO: YOUR CODE HERE
        # 返回TF-IDF值
        return self.tf(word, document)*self.idf(word)

    def __call__(self, query):
        query = tokenize(query) # 将问题token化
        # TODO: YOUR CODE HERE
        # 利用上述函数来实现QA
        # 提示：你需要根据TF-IDF值来选择一个最合适的文档，再根据IDF值选择最合适的句子
        # 返回时请返回原本句子，而不是token化后的句子，数据结构请参考README中数据结构部分以及fruit.py中用于数据处理的get_document()函数
        
        # Calculate the TF-IDF value for each document
        document_tfidf = []
        for document in self.document_list:
            tfidf_sum = sum([self.tfidf(word, document) for word in query])
            document_tfidf.append(tfidf_sum)
        # Choose the document with the highest TF-IDF value
        best_doc = self.document_list[np.argmax(document_tfidf)]
        # Calculate the IDF value for each sentence
        sentence_scores = []
        for singleton in best_doc['sentences']:
            words = singleton[0]
            sentence = singleton[1]
            matching_words_idf = sum([self.idf(word) for word in query if word in words])
            query_term_density = sum(1 for word in words if word in query)/len(words) if len(words) != 0 else 0 
            sentence_scores.append((sentence, matching_words_idf, query_term_density))
        # Choose the sentence with the highest IDF value, for the case of a tie, choose the one with the highest query term density
        best_sentence = max(sentence_scores, key=lambda x: (x[1], x[2]))[0]
        return best_sentence

modeldict = {
    "Null": NullModel,
    "Naive": NaiveBayesModel,
    "Attn": AttentionModel,
    "QA": QAModel,
}


if __name__ == '__main__':
    embedding = Embedding()
    lr = 1e-2   # 学习率
    wd1 = 1e-5  # L1正则化
    wd2 = 2e-4   # L2正则化
    batchsize = 64
    max_epoch = 20
    
    max_L = 50
    num_classes = 2
    feature_D = 100
    
    graph = buildGraph(feature_D, num_classes, max_L) # 维度可以自行修改

    # 训练
    # 完整训练集训练有点慢
    best_train_acc = 0
    dataloader = traindataset(shuffle=True) # 完整训练集
    #dataloader = minitraindataset(shuffle=True) # 用来调试的小训练集
    for i in range(1, max_epoch+1):
        hatys = []
        ys = []
        losss = []
        graph.train()
        X = []
        Y = []
        cnt = 0
        for text, label in dataloader:
            x = embedding(text, max_L)
            label = np.zeros((1)).astype(np.int32) + label
            X.append(x)
            Y.append(label)
            cnt += 1
            if cnt == batchsize:
                X = np.stack(X, 0)
                Y = np.concatenate(Y, 0)
                graph[-1].y = Y
                graph.flush()
                pred, loss = graph.forward(X)[-2:]
                hatys.append(np.argmax(pred, axis=-1))
                ys.append(Y)
                graph.backward()
                graph.optimstep(lr, wd1, wd2)
                losss.append(loss)
                cnt = 0
                X = []
                Y = []

        loss = np.average(losss)
        acc = np.average(np.concatenate(hatys)==np.concatenate(ys))
        print(f"epoch {i} loss {loss:.3e} acc {acc:.4f}")
        if acc > best_train_acc: 
            best_train_acc = acc
            with open(save_path, "wb") as f:
                pickle.dump(graph, f)