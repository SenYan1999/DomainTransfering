import os
import glob
import jieba
import re
import pandas as pd
import jieba.posseg as pseg
from collections import defaultdict

#######################################################
# # #       利用分词得到实体种子，再利用词性扩展      # # #
#######################################################

def convert_to_train(filename, out_filename):
    """ 将词典标注数据转为训练数据格式
    """
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    product_set = set()
    words_list = []
    label_list = []
    for line in lines:
        line = line.strip()
        if len(line) > 2:
            for word in line:
                if word != "@" and word != " ":
                    words_list.append(word)
                    label_list.append("O")
            # 非空格非顿号
            p = "@@[^ |^、]+?\@@"
            pattern = re.compile(p)
            products = pattern.findall(line)
            # print(products)
            if len(products) > 1:
                for product in products:
                    product_set.add(product.replace("@@", ""))
    # 将标注结果写入csv
    df = pd.DataFrame({"token": words_list, "label": label_list})

    length = len(df['token'])
    count = 0
    # 处理两个字的产品
    for i in range(0, length - 1):
        word = df['token'][i] + df['token'][i + 1]
        if word in product_set:
            df['label'][i] = 'B-product'
            df['label'][i + 1] = 'E-product'
    print("2个字的产品处理完成！")

    # 处理名字大于2的产品
    for j in range(3, 15):
        for i in range(0, length - 1 - j + 2):
            word = ''
            for n in range(0, j):
                word += df['token'][i + n]
            if word in product_set:
                df['label'][i] = 'B-product'
                df['label'][i + j - 1] = 'E-product'
                for x in range(1, j - 1):
                    df['label'][i + x] = 'M-product'
        print("{}个字的处理完成!".format(j))

    save = pd.DataFrame({"token": df["token"], "label": df["label"]})
    if not os.path.exists(out_filename):
        os.system(r"touch {}".format(out_filename))#调用系统命令行来创建文件

    df.to_csv(out_filename, index=False, sep=" ", header=None)
    print("{}文件处理完毕，csv路径为{}".format(filename, out_filename))
    return product_set


def add_line_blank(filename, out_filename):
    with open(filename, "r", encoding='utf-8-sig') as f:
        lines = f.readlines()
    f = open(out_filename, 'w', encoding='utf-8')
    for line in lines:
        line = line.strip()
        if len(line):
            word, tag = line.split()
            if word != "。":
                f.write(line)
                f.write("\n")
            else:
                f.write(line)
                f.write("\n\n")


def cut_cars_phrase(file_in, file_out):
    f_out = open(file_out, 'w', encoding='utf-8')

    with open(file_in, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        if len(line.split()) >= 2:
            cut_word = " ".join(jieba.cut(line))
            if len(cut_word) >= 2: # 去除一个字
                # print(cut_word)
                f_out.write(cut_word+"\n")

def get_word_count(cars_product_path):
    illegal_word = ['公司', '名称', '行业']
    word_freq = defaultdict(int)
    with open(cars_product_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        words_list = line.strip().split()
        for word in words_list:
            if len(word) > 1 and word not in illegal_word:
                word_freq[word] += 1
    return word_freq


def get_sent(csv_path):
    """ 从csv中获取所有句子
    """
    with open(csv_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    sent = ""
    sent_list = []
    for line in lines:
        line = line.strip()
        if len(line):
            token, label = line.split()
            sent += str(token)
        else:
            sent_list.append(sent)
            sent = ""
    return sent_list

def fowardMaxMatching(strList, maxLen, sentence):
    """ 正向最大匹配产品
    """
    wordList = []
    while (len(sentence) > 0):
        word = sentence[0:maxLen]  # 每次取最大词长的词

        meet = False  # 标记位，判断是否找到该词

        while ((not meet) and (len(word) > 0)):
            # 如果词在词表中
            if (word in strList):
                wordList.append("@@" + word + "@@")
                sentence = sentence[len(word):len(sentence)]  # 后移
                meet = True
            # 词不在词表中：
            else:
                if (len(word) == 1):
                    wordList.append(word)
                    sentence = sentence[len(word):len(sentence)]  # 后移
                    meet = True
                else:
                    word = word[0:len(word) - 1]  # 词语长度减少一位
    return wordList


def exclude_word_single(word):
    illegal_words = ['公司', '企业', '产品', '系统']
    for w in illegal_words:
        if w in word:
            return False
    return True


def exclude_word(word):
    illegal_words = ['公司', '企业']
    for w in illegal_words:
        if w in word:
            return False
    return True

def tag_sent(word_freq, sent, freq = 100):
    legal_pos = ['n', 'nz', 'nr', 'l', 'ng', 'eng']
    pos_list = list(pseg.cut(sent))
    words_set = set()
    word_max_len = 0
    for i in range(len(pos_list)):
        w = pos_list[i]
        # print("cur:", w.word)
        if word_freq[w.word] > 0 and w.flag in legal_pos:
            # print(w.word, w.flag)
            # 向前扩展3个词
            cur_index = i
            pre_index = cur_index - 1
            word = w.word
            while (pre_index >= 0 and cur_index - pre_index <= 3):
                w_pred = pos_list[pre_index]
                if w_pred.flag not in legal_pos:
                    break
                else:
                    pre_index -= 1
                    word = w_pred.word + word
                    # print("pre word:", w_pred.word)

            # 向后扩展
            next_index = cur_index + 1
            while (next_index < len(pos_list) and next_index - cur_index <= 3):
                w_next = pos_list[next_index]
                if w_next.flag not in legal_pos:
                    break
                else:
                    next_index += 1
                    word = word + w_next.word
                    # print("next word:", w_next.word)

            if len(word) > len(w.word) and exclude_word(word):  # 当增加了前后词
                words_set.add(word)
                word_max_len = max(word_max_len, len(word))
            elif exclude_word_single(w.word) and word_freq[w.word] >= freq: # 前后均无词，当前词频率大于等于500
                words_set.add(w.word)
                word_max_len = max(word_max_len, len(w.word))
    words_list = list(words_set)
    print("产品:", words_list)
    if len(words_list):
        taged_sent = "".join(fowardMaxMatching(words_list, word_max_len, sent))
    else:
        taged_sent = sent
    return taged_sent


def combined_closed_product(sent):
    """ 合并一个句子中相邻的产品
    """
    p = "@@[^ |^、]+?\@@"
    pattern = re.compile(p)
    products = pattern.findall(sent)
    if len(products) == 0:
        return sent
    new_product_list = []
    max_len = 0
    if len(products) == 1:
        new_product_list.append(products[0])
        max_len = max(max_len, len(products[0]))
    else:
        i = 0
        while i < len(products)-1:
            cur_product = products[i]
            next_product = products[i+1]
            cur_index = sent.index(cur_product)
            next_index = sent.index(next_product)
            # print(cur_index+len(cur_product), next_index)
            if cur_index+len(cur_product) == next_index:
                new_product = cur_product[2:-2] + next_product[2:-2]
                max_len = max(max_len, len(new_product))
                new_product_list.append(new_product)
                i += 2
                if i == len(products)-1:
                    new_product_list.append(products[-1][2:-2])
                    max_len = max(max_len, len(products[-1][2:-2]))
            else:
                max_len = max(max_len, len(cur_product[2:-2]))
                new_product_list.append(cur_product[2:-2])
                i += 1
                if i == len(products)-1:
                    max_len = max(max_len, len(next_product[2:-2]))
                    new_product_list.append(next_product[2:-2])
                    i += 1
    # print(new_product_list)
    # print(max_len)
    new_sent = sent.replace("@","")
    # print(new_sent)
    wordList = fowardMaxMatching(new_product_list, max_len, new_sent)
    res_sent = "".join(wordList)
    # print(res_sent)
    return res_sent


def handle_number_signal(sent):
    """ 处理并列顿号
    """
    p = "@@[^ |^、]+?\@@"
    pattern = re.compile(p)
    # print(sent)
    products = pattern.findall(sent)
    p_new = "、[^ ]+?\、"
    # print(products)
    new_product_list = []
    max_len = 0
    if len(products) == 0 or len(products) == 1:
        return sent
    else:
        i = 0
        while i < len(products)-1:
            cur_product = products[i]
            next_product = products[i+1]
            cur_index = sent.index(cur_product)
            # print(cur_product)
            next_index = sent.index(next_product)
            str = sent[cur_index+len(cur_product):next_index]
            new_product_list.append(cur_product[2:-2])
            max_len = max(max_len, len(cur_product[2:-2]))
            if str.startswith("、") and str.endswith("、") and 2 < len(str) <= 9: # 除去前后"、"长度小于等于7
                new_product_list.append(str[1:-1])
                max_len = max(max_len, len(str[1:-1]))
            i += 1
            if i == len(products)-1:
                new_product_list.append(next_product[2:-2])
                max_len = max(max_len, len(next_product[2:-2]))
    new_sent = sent.replace("@","")
    update_sent = "".join(fowardMaxMatching(new_product_list, max_len, new_sent))
    return update_sent


def count_word(cars_cuted_phrase_path, csv_path, out_path, freq = 100 , use_rule = True):
    """ （1）利用分词得到实体种子；（2）利用词性进行实体扩展；（3）利用顿号规则
    """

    # get cut words from products
    f = open(out_path, 'w', encoding='utf-8')
    word_freq = get_word_count(cars_cuted_phrase_path)
    sent_list = get_sent(csv_path)
    for sent in sent_list:
        # 标注
        taged_sent = tag_sent(word_freq, sent, freq)
        print("分词标注前:", sent)
        print("分词标注后:", taged_sent)
        if use_rule: # 使用规则
            # 合并相邻产品
            new_sent = combined_closed_product(taged_sent)  # 没有产品的句子返回空字符
            new_sent = combined_closed_product(new_sent)  # 有多个相邻时需要多次合并
            print("相邻合并后:", new_sent)
            # 处理顿号:products[-1][2:-2]
            number_sent = handle_number_signal(new_sent)
            print("顿号处理后:", number_sent)
        else:
            number_sent = taged_sent
        f.write(number_sent)
        f.write("\n")
    f.close()
    # 利用words_set标注


if __name__ == "__main__":
    # 将汽车相关的词条分词
    cut_cars_phrase()

    # 测试集
    freq = 100
    use_rule = True
    cars_cuted_phrase_path = "../../undergraduate_code/tools/data/cars_phrase.txt"

    name_suffix = "_"
    if cars_cuted_phrase_path =="./data/result_removed_brand_5703_cut.txt":
        name_suffix += "5703"
    elif cars_cuted_phrase_path == "./data/cars_phrase.txt":
        name_suffix += "5703_add"
    elif cars_cuted_phrase_path == "./data/result_4755_cut.txt":
        name_suffix += "4755"

    csv_path = "../../undergraduate_code/tools/data/train2.txt"
    if use_rule:
        name_suffix += "_useRule"
    else:
        name_suffix += "_noRule"

    name_suffix += "_freq"+str(freq)

    out_path = "./res/train_cutTag"+name_suffix+".txt"
    count_word(cars_cuted_phrase_path, csv_path, out_path, freq)
    out_file = "./res/train_cutTag"+name_suffix+"_temp.csv"
    convert_to_train(out_path, out_file)
    new_out = "./res/test_cutTag"+name_suffix+".csv"
    add_line_blank(out_file, new_out)