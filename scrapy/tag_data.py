import json
import re
import jieba
import jieba.posseg as pseg
import numpy as np

from collections import Counter

def get_new_product(handcraft_file, new_product):
    pattern = re.compile('@@([^、|^ |。]+?)@@')
    all_products = []
    with open(handcraft_file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            products = re.findall(pattern, line)
            all_products += products

    with open(new_product, 'w', encoding='utf-8') as f:
        json.dump(all_products, f, ensure_ascii=False)

class Tagger:
    def __init__(self, product, source_file, freq = 100, use_rule = True, handcraft_file = None):
        self.use_rule = use_rule
        self.product_freq = self.get_product_freq(product, handcraft_file)
        self.sents = self.get_sents(source_file)
        self.labels = self.get_tag_labels(freq)

    def get_product_freq(self, product_file, handcraft_file):
        product_freq = Counter()
        new_product_freq = Counter()

        # normal file
        with open(product_file, 'r', encoding='utf-8-sig') as f:
            products = json.load(f)

        for product in products:
            product_freq.update(jieba.cut(product))

        # handcraft file
        if handcraft_file:
            with open(handcraft_file, 'r', encoding='utf-8') as f:
                products = json.load(f)
            for product in products:
                new_product_freq.update(jieba.cut(product))

        product_freq.update(new_product_freq)

        return product_freq
    
    def get_sents(self, source_file):
        def cut_sent(lines):
            results = []
            _punk = '。。！!！?？'.replace(' ', '')
            _punk_list = set(list(_punk))

            for line in lines:
                line = line.strip()
                for i in _punk_list:
                    line = line.replace(i, i + '@#@punc@#@')
                sens = line.split('@#@punc@#@')
                for sen in sens:
                    if sen:
                        results.append(sen)
            return results

        sents = []
        with open(source_file, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                sents.append(line[:-1])

        sents = cut_sent(sents)

        return sents

    def get_tag_labels(self, freq):
        legal_pos = ['n', 'nz', 'nr', 'l', 'ng', 'eng']
        all_tag_labels = []

        def is_stop_word(word):
            if word in ['1', '2']:
                return True
            else:
                return False

        def max_matching(cands, sent):
            cands_len = [len(cand) for cand in cands]
            sort_idx = np.argsort(cands_len).tolist()
            cands = [cands[i] for i in sort_idx]

            labels = ['O' for _ in range(len(sent))]

            spans = []
            for cand in reversed(cands):
                start_idx = sent.find(cand)
                end_idx = start_idx + len(cand) - 1
                cur_span = (start_idx, end_idx)

                involved = False
                for span in spans:
                    # start idx in longer cand span
                    if span[0] <= start_idx <= span[1]:
                        involved = True
                        break
                    if span[0] <= end_idx <= span[1]:
                        involved = True
                        break
                
                if not involved:
                    spans.append(cur_span)

            # tag sents based on spans
            for span in spans:
                for i in range(span[0]+1, span[1]):
                    labels[i] = 'M-product'

                labels[span[0]] = 'B-product'
                labels[span[1]] = 'E-product'

            if self.use_rule:
                # combine product
                for span in spans:
                    if labels[span[0] - 1] == 'E-product':
                        labels[span[0] - 1] = 'M-product'
                        labels[span[0]] = 'M-product'
                    try:
                        if labels[span[1] + 1] == 'B-product':
                            labels[span[1] + 1] = 'M-product'
                            labels[span[1]] = 'M-product'
                    except:
                        pass

                # tag between two '、'
                pattern = re.compile('、')
                all_occurance = [m.start() for m in re.finditer(pattern, sent)]
                if len(all_occurance) > 1:
                    for i in range(len(all_occurance) - 1):
                        if all_occurance[i+1] == len(sent) - 1:
                            continue
                        if labels[all_occurance[i] - 1] == 'E-product' and labels[all_occurance[i+1] + 1] == 'B-product':
                            if all_occurance[i + 1] - all_occurance[i] < 7:
                                for j in range(all_occurance[i] + 1, all_occurance[i+1]):
                                    labels[j] = 'M-product'
                                labels[all_occurance[i] + 1] = 'B-product'
                                labels[all_occurance[i+1] - 1] = 'E-product'

            return labels

        for sent in self.sents:
            pos_list = list(pseg.cut(sent))
            words_set = set()
            for i in range(len(pos_list)):
                w = pos_list[i]
                # print("cur:", w.word)
                if self.product_freq[w.word] > 0 and w.flag in legal_pos:
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

                    if len(word) > len(w.word) and not is_stop_word(word):  # 当增加了前后词
                        words_set.add(word)
                    elif not is_stop_word(w.word) and self.product_freq[w.word] >= freq:  # 前后均无词，当前词频率大于等于500
                        words_set.add(w.word)
            words_list = list(words_set)

            if len(words_list) > 0:
                tag_labels = max_matching(words_list, sent)
            else:
                tag_labels = ['O' for _ in range(len(sent))]

            all_tag_labels.append(tag_labels)

        return all_tag_labels

    def write_to_file(self, out_file):
        with open(out_file, 'w', encoding='utf-8') as f:
            for sent, label in zip(self.sents, self.labels):
                assert len(sent) == len(label)
                for i in range(len(sent)):
                    f.write(sent[i] + ' ' + label[i] + '\n')
                f.write('\n')

if __name__ == '__main__':
    product = 'data/data.json'
    source_file = 'data/IPO/FundamentalChemical/FundamentalChemical.txt'
    target_file = 'a.txt'
    handcraft_file = 'data/IPO/FundamentalChemical/FundamentalChemical_handcraft.txt'
    get_new_product(handcraft_file, 'data/new_product.json')
    tagger = Tagger(product, source_file, handcraft_file='data/new_product.json')
    tagger.write_to_file(target_file)
