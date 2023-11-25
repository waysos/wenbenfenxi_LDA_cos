# 本文件为数据预处理函数
import re
import jieba
import numpy as np
import matplotlib.pyplot as plt
from gensim import corpora, models
from gensim.matutils import sparse2full


# 全角转半角
def full_to_half(text:str):      # 输入为一个句子
    _text = ""
    for char in text:
        inside_code = ord(char)     # 以一个字符（长度为1的字符串）作为参数，返回对应的 ASCII 数值
        if inside_code == 12288:    # 全角空格直接转换
            inside_code = 32
        elif 65281 <= inside_code <= 65374:  # 全角字符（除空格）根据关系转化
            inside_code -= 65248
        _text += chr(inside_code)
    return _text

# 文本清洗，过滤非文本内容
def clear_character(sentence):
    #pattern = re.compile("[^\u4e00-\u9fa5^,^.^!^a-z^A-Z^0-9]")  # 只保留中英文、数字和符号，去掉其他东西
    pattern = re.compile("[^\u4e00-\u9fa5^a-z^A-Z^0-9]")         # 只保留中英文和数字
    line = re.sub(pattern, '', sentence)  # 把文本中匹配到的字符替换成空字符
    new_sentence = ''.join(line.split())  # 去除空白
    return new_sentence

# 分词函数
def cut_word(text):
    text = jieba.cut(text)
    # for i in text:
    #     print(i)
    return text

# 停用词函数
def drop_stopwords(contents, stopwords):
    contents_clean = []
    for line in contents:
        line_clean = []
        for word in line:
            if word in stopwords:
                continue
            line_clean.append(word)
        contents_clean.append(line_clean)
    return contents_clean

def choose_topic(dictionary_wai, corpus_wai, finall_data_wai):
        '''
        @description: 生成模型
        @param
        @return: 生成主题数分别为1-15的LDA主题模型，并保存起来。
        '''
        dictionary = dictionary_wai
        corpus = corpus_wai
        texts = finall_data_wai
        for i in range(1, 16):
            print('目前的topic个数:{}'.format(i))
            print('目前的数据量:{}'.format(len(texts)))
            temp = 'lda_{}_{}'.format(i, len(texts))
            tmp = models.ldamodel.LdaModel(corpus, num_topics=i, id2word=dictionary, passes=20)
            file_path = './{}.model'.format(temp)
            tmp.save(file_path)
            print('------------------')


def perplexity_visible_model(topic_num, data_num, corpus_wai):
    '''
    @description: 绘制困惑度-主题数目曲线
    @param {type}
    @return:
    '''
    # texts = self.fenci_data()
    corpus = corpus_wai
    x_list = []
    y_list = []
    for i in range(1, topic_num):
        model_name = './lda_{}_{}.model'.format(i, data_num)
        try:
           lda = models.ldamodel.LdaModel.load(model_name)
           perplexity = lda.log_perplexity(corpus)
           print(perplexity)
           x_list.append(i)
           y_list.append(perplexity)
        except Exception as e:
           print(e)
    plt.xlim(0, 16)
    plt.ylim(-11, -8)
    plt.xlabel('num topics')
    plt.ylabel('perplexity score')
    plt.legend(('perplexity_values'), loc='best')
    plt.plot(x_list, y_list)
    plt.show()

def visible_model(topic_num, data_num, dictionary_wai, finall_data_wai):
        '''
        @description: 可视化模型
        @param :topic_num:主题的数量
        @param :data_num:数据的量
        @return: 可视化lda模型
        '''
        dictionary = dictionary_wai
        texts = finall_data_wai
        x_list = []
        y_list = []
        for i in range(1, topic_num):
            model_name = './lda_{}_{}.model'.format(i, data_num)
            try:
                lda = models.ldamodel.LdaModel.load(model_name)
                cv_tmp = models.CoherenceModel(model=lda, texts=texts, dictionary=dictionary, coherence='c_v')
                x_list.append(i)
                y_list.append(cv_tmp.get_coherence())
                print("进行中")
            except:
                print('没有这个模型:{}'.format(model_name))
        plt.plot(x_list, y_list)
        plt.xlabel('num topics')
        plt.ylabel('coherence score')
        plt.legend(('coherence_values'), loc='best')
        plt.show()

# 与下面两个为计算文本余弦相似度的函数
def get_lda_vector(lda_model, doc):
    # 将文档转换为词袋向量
    vec = lda_model.id2word.doc2bow(doc)
    # 将词袋向量转换为LDA主题向量
    topics = lda_model[vec]
    # 将稀疏向量转换为密集向量，并返回
    return sparse2full(topics, lda_model.num_topics)
# 输入为生成的lda模型和要计算余弦相似的语句
def cosine_similarity(lda_model, doc1, doc2):
    # 将文档转换为LDA主题向量
    vec1 = get_lda_vector(lda_model, doc1)
    vec2 = get_lda_vector(lda_model, doc2)
    # 计算余弦相似度
    sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return sim


def main():
    # 获取要处理的数据
    with open('info.txt', 'r', encoding='utf-8') as f:
            data = f.readlines()
            # print(data[5736])

    # 全角转半角
    for i in range(0, len(data)):
        data[i] = full_to_half(data[i])
    # print(data)

    # 过滤非文本
    for i in range(0, len(data)):
        data[i] = clear_character(data[i])
    # print(data)

    # 创建数组
    fc_data = []

    # 分词
    for i in range(0, len(data)):
        zjfc_data = []
        text = cut_word(data[i])
        for j in text:
            zjfc_data.append(j)
        fc_data.append(zjfc_data)
    # print(fc_data)

    # 停用词表
    with open('cn_stopwords.txt', 'r', encoding='utf-8') as f:
            stopwords = f.readlines()
    for i in range(0, len(stopwords)):
        stopwords[i] = full_to_half(stopwords[i])
    for i in range(0, len(stopwords)):
        stopwords[i] = clear_character(stopwords[i])

    # 去掉文本中的停用词
    qu_stopword_data = drop_stopwords(fc_data, stopwords)
    #print(qu_stopword_data)

    # 去除空数组
    finall_data = list(filter(None, qu_stopword_data))
    #print(finall_data)

    # 构建词袋模型
    dictionary = corpora.Dictionary(finall_data)
    corpus = [dictionary.doc2bow(text) for text in finall_data]
    # print(dictionary.token2id)
    # print(corpus)

#### 以下为LDA模型选取文本特征
    # # 生成模型,第一次运行的时候需要去注释
    # choose_topic(dictionary, corpus, finall_data)

    # # 求取较为合适的主题数
    # data_num = len(finall_data) # 数据总数
    # topic_num = 15 # 设定最大主题数
    # perplexity_visible_model(topic_num, data_num, corpus)
    # visible_model(topic_num, data_num, dictionary, finall_data)
    # # 可以得出最合适的主题数为7个.

    # LDA模型，num_topics设置主题的个数
    lda = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=7)

    # 打印所有主题，每个主题显示5个词
    for topic in lda.print_topics(num_words=5):
        print(topic)
####

# 计算余弦相似度
    print(cosine_similarity(lda, finall_data[1], finall_data[2]))



if __name__ == '__main__':
    main()
