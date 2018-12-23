
#coding: utf-8
import os
import time
import random
import jieba
import nltk
import sklearn
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import PCA
from sklearn import metrics
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

def MakeWordsSet(words_file):
    words_set = set()
    with open(words_file, 'r') as fp:
        for line in fp.readlines():
            word = line.strip().decode("utf-8")
            if len(word)>0 and word not in words_set: # 去重
                words_set.add(word)
    return words_set

def TextProcessing(folder_path, test_size=0.2):
    folder_list = os.listdir(folder_path)
    data_list = []
    class_list = []

    # 类间循环
    for folder in folder_list:
        new_folder_path = os.path.join(folder_path, folder)
        files = os.listdir(new_folder_path)
        # 类内循环
        for file in files:
            with open(os.path.join(new_folder_path, file), 'r') as fp:
               raw = fp.read()
            # print raw
            ## --------------------------------------------------------------------------------
            ## jieba分词
            # jieba.enable_parallel(4) # 开启并行分词模式，参数为并行进程数，不支持windows
            word_cut = jieba.cut(raw, cut_all=False) # 精确模式，返回的结构是一个可迭代的genertor
            word_list = list(word_cut) # genertor转化为list，每个词unicode格式
            # jieba.disable_parallel() # 关闭并行分词模式
            # print word_list
            ## --------------------------------------------------------------------------------
            data_list.append(word_list)
            class_list.append(folder.decode('gbk'))

    ## 划分训练集和测试集
    # train_data_list, test_data_list, train_class_list, test_class_list = sklearn.cross_validation.train_test_split(data_list, class_list, test_size=test_size)
    data_class_list = zip(data_list, class_list)
    random.shuffle(data_class_list)
    data_list, class_list = zip(*data_class_list)

    # 统计词频放入all_words_dict
    all_words_dict = {}
    for word_list in data_list:
        for word in word_list:
            if all_words_dict.has_key(word):
                all_words_dict[word] += 1
            else:
                all_words_dict[word] = 1
    # key函数利用词频进行降序排序
    all_words_tuple_list = sorted(all_words_dict.items(), key=lambda f:f[1], reverse=True) # 内建函数sorted参数需为list
    all_words_list = list(zip(*all_words_tuple_list)[0])

    return all_words_list,all_words_tuple_list,data_list, class_list




def words_dict(all_words_list, stopwords_set=set()):
    # 选取特征词
    feature_words = []
    n = 1
    for t in all_words_list:
        if not t.isdigit() and t not in stopwords_set and 1<len(t)<5:
            feature_words.append(t)
    return feature_words


def TextFeatures(train_data_list, feature_words, flag='nltk'):
    def text_features(text, feature_words):
        text_words = set(text)
        features = np.zeros(len(feature_words))
        ## -----------------------------------------------------------------------------------
        if flag == 'nltk':
            ## nltk特征 dict
            features = {word:1 if word in text_words else 0 for word in feature_words}
        elif flag == 'sklearn':
            ## sklearn特征 list
            features = [1 if word in text_words else 0 for word in feature_words]
            for word in feature_words:
                if word in text_words:
                    features[feature_words.index(word)] += 1
        else:
            features = []
        ## -----------------------------------------------------------------------------------
        return features
    train_feature_list = [text_features(text, feature_words) for text in train_data_list]
    return train_feature_list


def TextClassifier(train_feature_list, test_feature_list, train_class_list, test_class_list, flag='nltk'):
    ## -----------------------------------------------------------------------------------
    if flag == 'nltk':
        ## nltk分类器
        train_flist = zip(train_feature_list, train_class_list)
        test_flist = zip(test_feature_list, test_class_list)
        classifier = nltk.classify.NaiveBayesClassifier.train(train_flist)
        # print classifier.classify_many(test_feature_list)
        # for test_feature in test_feature_list:
        #     print classifier.classify(test_feature),
        # print ''
        test_accuracy = nltk.classify.accuracy(classifier, test_flist)
    elif flag == 'sklearn':
        ## sklearn分类器
        classifier = MultinomialNB().fit(train_feature_list, train_class_list)
        # print classifier.predict(test_feature_list)
        # for test_feature in test_feature_list:
        #     print classifier.predict(test_feature)[0],
        # print ''
        test_pre = classifier.predict(test_feature_list)
        f1_accuracy = metrics.f1_score(test_class_list,test_pre,average='weighted')
        test_accuracy = classifier.score(test_feature_list, test_class_list)

    else:
        test_accuracy = []
    return test_accuracy,f1_accuracy


if __name__ == '__main__':

    print "start"

    ## 文本预处理
    folder_path = './Data'  
    all_words_list,all_words_tuple_list,data_list, class_list = TextProcessing(folder_path, test_size=0.2)

    # 生成stopwords_set
    stopwords_file = './stopwords_cn.txt'
    stopwords_set = MakeWordsSet(stopwords_file)
    
    ## 文本特征提取和分类
    flag = 'sklearn'
    # flag = 'sklearn'
    # deleteNs = range(0, 1000, 10)
    test_accuracy_list = []
    test_f1_list = []
    tfidf_data = [" ".join(t) for t in data_list]
    tfidf_model2 = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b",max_df=0.4).fit(tfidf_data)
    all_words_list = tfidf_model2.get_feature_names()
    # for deleteN in deleteNs:
        # feature_words = words_dict(all_words_list, deleteN)
    feature_words = words_dict(all_words_list, stopwords_set)
    data_feature_list = TextFeatures(data_list, feature_words, flag)
    print len(feature_words)
    # pca = PCA(n_components=1000)
    # data_pca_list = pca.fit_transform(data_feature_list)
    test_pca_list = data_feature_list[-500:]
    print np.shape(test_pca_list)

    test_class_list = class_list[-500:]
    train_n = range(1000, 2500, 50)
    for n in train_n:
        train_pca_list = data_feature_list[0:n]
        train_class_list = class_list[0:n]
        (test_accuracy,f1_accuracy) = TextClassifier(train_pca_list, test_pca_list, train_class_list, test_class_list, flag)
        test_accuracy_list.append(test_accuracy)
        test_f1_list.append(f1_accuracy) 
    print test_accuracy_list.index(max(test_accuracy_list)),'accuracy:',max(test_accuracy_list)
    # print all_words_tuple_list
    # print len(all_words_tuple_list)
    # 结果评价
    plt.figure()
    plt.plot(train_n,test_f1_list,color='red')
    plt.title('Relationship of train_n and f1_accuracy')
    plt.xlabel('train_n')
    plt.ylabel('f1_accuracy')
    plt.show()
    plt.savefig('result.png')

    print "finished"