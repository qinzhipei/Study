# -*- coding: utf-8 -*-
'文本挖掘'
'''
#1.去除文本框两边的空格：text.strip()
#2.去除字符串中句号
#3.大写转换自定义函数
#4.字符替换（正则表达式）
#5.移除标点
#6.文本分词
#7.删除停用词
#8.提取词干
#9.词性标注(Part of speech tag)
#10.将句子转换为词性标注的特征:word_tokenize
#11.使用Brown corpus来测试词性标注器的效果
#12.TFIDF(term frequency-inverse document frequency)
#13.Bag of word 词袋模型
'''



# 1.去除文本框两边的空格：text.strip()
text_data = ["      Interrobang. By Aishwarya Henriette    ",
             "Parking And Going. By Karl Gautier",
             "  Today Is The night. By Jarek Praksh   "]

strip_space = [text.strip() for text in text_data]
strip_space

#2.去除字符串中句号
strip_space = [text.replace(".","") for text in strip_space]

#3.大写转换自定义函数
def capitalizer(string:str) -> str:
    return string.upper()
[capitalizer(text) for text in strip_space]

#4.字符替换（正则表达式）
import re
def substitude(string):
    return re.sub("[a-zA-Z]","6",string)
[substitude(text) for text in strip_space]

#5.移除标点
import unicodedata
import sys

text_data = ['Hi!!! I. Love. This. Song....',
             '10000% Agree!!! #LoveIT',
             'Right?!?!']

# 使用unicodedata载入标点符号的包
punctuation = dict.fromkeys(i for i in range(sys.maxunicode) 
                            if unicodedata.category(chr(i)).startswith('P'))
#将Unicode中的标点字符作为key，None作为value，然后将字符串中所有在punctuation字典中出现过的字符(标点)转换成None，高效地移除它们。
[string.translate(punctuation) for string in text_data]

#6.文本分词
import nltk
from nltk.tokenize import word_tokenize
string = 'The science of today is the technology of tomorrow'
word_tokenize(string)
# 导入分句模块,与分词不同,分句将把字符串分为不同的句子（分隔符为.）
from nltk.tokenize import sent_tokenize
string = 'The science of today is the technology of tomorrow. Tomorrow is today'
sent_tokenize(string)


#7.删除停用词
nltk.download('stopwords')
from nltk.corpus import stopwords # 在nltk包中加载停用词
tokenized_words = ['i',
                   'am',
                   'going',
                   'to',
                   'go',
                   'to',
                   'the',
                   'store',
                   'and',
                   'park']

stop_words = stopwords.words('english') #引用英文停用词列表

# 选择在文本字符串列表中的单词且不能出现在停用词列表中
[word for word in tokenized_words if word not in stop_words]
#输出['going', 'go', 'store', 'park']


#8.提取词干
#词干提取的目的是将一个单词的主干成分进行提取，删除其后缀(动名词的ing等)，保留词根的意思，例如“tradition”和“traditional”的词干都是“tradit”
# 在nltk.stem.porter中导入PorterStemmer
from nltk.stem.porter import PorterStemmer
tokenized_words = ['i','am','humbled','by','this','traditional','meeting','fucking']

# 实例化提取词干器
porter = PorterStemmer()
[porter.stem(word) for word in tokenized_words]


#9.词性标注(Part of speech tag)
from nltk import pos_tag
from nltk.tokenize import word_tokenize
nltk.download('averaged_perceptron_tagger')
text_data = 'Chris loved outdoor running'
text_tagged = pos_tag(word_tokenize(text_data))
text_tagged
# 利用特定词性对单词进行过滤
[text for text,tag in pos_tag(word_tokenize(text_data)) if tag in ['NN','NNS','NNP','NNPS']]


#10.将句子转换为词性标注的特征
from nltk.tokenize import word_tokenize

tweets = ["I am eating a burrito for breakfast",
          " Political science is an amazing field",
          "San Francisco is an awesome city"]

tagged_tweets = []

for tweet in tweets:
    tweet_tag = nltk.pos_tag(word_tokenize(tweet))
    # 只保留词性标注的词性
    tagged_tweets.append([tag for word,tag in tweet_tag])

print(tagged_tweets) #输出每个词语的词性


#11.使用Brown corpus来测试词性标注器的效果
from nltk.corpus import brown
from nltk.tag import UnigramTagger #词性标注器模块
from nltk.tag import BigramTagger
from nltk.tag import TrigramTagger
nltk.download('brown')
# 在布朗语料库中获取文本数据,切分成句子
sentences = brown.tagged_sents(categories = 'news')

# 将4000个句子作为训练,将623个作为测试
train = sentences[:4000]
test = sentences[4000:]

# 创建回退标注器
unigram = UnigramTagger(train)
bigram = BigramTagger(train,backoff = unigram)
trigram = TrigramTagger(train,backoff = bigram)

# 查看准确率
trigram.evaluate(test)
bigram.evaluate(test)
unigram.evaluate(test)


#12.TFIDF(term frequency-inverse document frequency)
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
text_data = np.array(['I love Brazil. Brazil!',
                      'Sweden is best',
                      'Germany beats both'])
# 实例化tfidf转换器
tdidf = TfidfVectorizer()
feature_matrix = tdidf.fit_transform(text_data)

feature_matrix.toarray()
'''array([[0.        , 0.        , 0.        , 0.89442719, 0.        ,
        0.        , 0.4472136 , 0.        ],
       [0.        , 0.57735027, 0.        , 0.        , 0.        ,
        0.57735027, 0.        , 0.57735027],
       [0.57735027, 0.        , 0.57735027, 0.        , 0.57735027,
        0.        , 0.        , 0.        ]])'''

feature_matrix.toarray().shape  # 三行句子,共八个词(3, 8)

# 使用tfidf查看单词
tdidf.vocabulary_

pd.DataFrame(feature_matrix.toarray(),columns=list(tdidf.vocabulary_.keys()))


#13.Bag of word 词袋模型
from sklearn.feature_extraction.text import CountVectorizer
# 实例化向量器
count = CountVectorizer()
bag_of_words = count.fit_transform(text_data)
bag_of_words.toarray()
# 查看词的名称
count.get_feature_names()

#查看特定词在句子中的出现次数
count_2gram = CountVectorizer(ngram_range = (1,2),stop_words='english',vocabulary=['brazil'])
bag = count_2gram.fit_transform(text_data)
bag.toarray() 
'''array([[2],
       [0],
       [0]], dtype=int64) 在三个句子中，brazil分别出现2，0，0次'''
    



from nltk.corpus import wordnet as wn
nltk.download('wordnet')
wn.synsets('dog')