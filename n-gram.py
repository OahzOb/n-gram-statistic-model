import os
os.chdir(os.path.dirname(__file__)) # 确保工作目录为脚本所在地址
print('当前工作目录' + os.getcwd())
import sys
print(sys.version)

# 预处理文本为语料
import re
def corpus_prepare(corpus_unprepared):
    corpus = 'B' + corpus_unprepared + 'E' #语料首尾加标志
    corpus = re.sub(r'\n+', 'EB', corpus)
    corpus = re.sub(r'###.*?###', '', corpus)
    corpus = re.sub(r'\(.*?\)', '', corpus)
    return corpus

# 统计序列个数
from collections import Counter
def build_ngram_sequences(corpus, n):
    ngram_sequences = [corpus[i : i + n] for i in range(len(corpus) - n + 1)]
    return Counter(ngram_sequences)

# 去除人为增加的序列
def remove_unwanted(ngram_sequences):
    ngram_sequences = {key : value for key, value in ngram_sequences.items() if 'EB' not in key and '：E' not in key}
    return ngram_sequences

# 输出进度条
import sys
def print_progress_bar(iteration, total, prefix = '', suffix = '', decimals =2, length = 60, fill = '█'):
    percent = ('{0:.' + str(decimals) + 'f}').format(100 * (iteration / total))
    filled_length = length * iteration // total
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write('\r%s |%s| %s%% %s' %(prefix, bar, percent, suffix)),
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()

# 加一法平滑
def add_one_smoothing(ngram_sequences, n):
    probabilities = {}
    probabilities_per = {}
    history_counter = Counter()
    ngram_sequences_sum = len(ngram_sequences)
    # 每一历史下所有可能的n-gram组合个数
    total_combinations_sum = unique_char_num
    for ngram, count in ngram_sequences.items():
        history_counter[ngram[: n - 1]] += count
    for i, (ngram, count) in enumerate(ngram_sequences.items(), start = 1):
        history = ngram[: n - 1]
        # 计算该历史下每个已出现n-gram的概率
        probabilities[ngram] = (count + 1) / (history_counter[history] + total_combinations_sum)
        # 计算该历史下未出现的n-gram组合概率
        if history not in probabilities_per:
            probabilities_per[history] = 1 / (history_counter[history]+ total_combinations_sum)
        # 输出进度条
        if i % 100 == 0:
            print_progress_bar(i, ngram_sequences_sum, prefix = str(n) + '-gram 加一法平滑进度')
    print_progress_bar(i, ngram_sequences_sum, prefix = str(n) + '-gram 加一法平滑进度')
    return probabilities, probabilities_per

# 绝对减值法平滑
def absolute_discounting_smoothing(ngram_sequences, n, b):
    probabilities = {}
    probabilities_per = {}
    history_counter = Counter()# 计算每一历史下的n-gram个数
    unique_ngram_num = Counter()# 计算每一历史下的不同n-gram组合数
    ngram_sequences_sum = len(ngram_sequences)
    for ngram, count in ngram_sequences.items():
        unique_ngram_num[ngram[: n - 1]] += 1
        history_counter[ngram[: n - 1]] += count
    for i, (ngram, count) in enumerate(ngram_sequences.items(), start = 1):
        history = ngram[: n - 1]            
        # 计算该历史下每个已出现n-gram的概率
        probabilities[ngram] = (count - b) / history_counter[history]
        # 计算该历史下未出现的n-gram组合概率
        probabilities_per.setdefault(history, b * unique_ngram_num[history] / (unique_char_num - unique_ngram_num[history]))
        # 输出进度条 
        if i % 100 == 0:
            print_progress_bar(i, ngram_sequences_sum, prefix = str(n) + '-gram 绝对减值法平滑进度')
    print_progress_bar(i, ngram_sequences_sum, prefix = str(n) + '-gram 绝对减值法平滑进度')
    return probabilities, probabilities_per

# 加一法平滑计算句子概率
def sentence_predict_add(sentence, n):
    with open(f'Model\\{n}-gram_add.json', 'r', encoding = 'utf-8') as file:
        ngram_probabilities = json.load(file)
    with open(f'Model\\{n}-gram_per_add.json', 'r', encoding = 'utf-8') as file:
        ngram_probabilities_per = json.load(file)

    probability = 1
    for i in range(len(sentence) - 1):
        ngram_probability = ngram_probabilities.get(sentence[i : i + n], 0)
        if not ngram_probability:# 给定序列未出现过
            ngram_probability = ngram_probabilities_per.get(sentence[i : i + n - 1], 0)
            if not ngram_probability:# 给定历史未出现过
                ngram_probability = 1 / unique_char_num
        probability *= ngram_probability
    return probability

# 加一法平滑续写句子
import random
def next_word_predict_add(sentence, num, n, corpus_counter):
    with open(f'Model\\{n}-gram_add.json', 'r', encoding = 'utf-8') as file:
        ngram_probabilities = json.load(file)

    for i in range(num):
        history = sentence[-n + 1 :]
        probability_max = 0
        word_next = None
        candidates = []
        check = 0
        for ngram, probability in ngram_probabilities.items():
            if history == ngram[0 : n - 1] and probability_max <= probability:
                check = 1
                if probability_max == probability:# 概率相同则列为候选字符
                    candidates.append(ngram[-1])
                else:# 概率不同则替换为概率大者
                    probability_max = probability
                    candidates = [ngram[-1]]
        if check == 0:# 给定历史在语料中未出现，随机选取字符
            word_next = random.choice(list(corpus_counter.keys()))
        else:
            word_next = random.choice(candidates)
        sentence += str(word_next)

    return sentence

# 绝对减值法平滑计算句子概率
def sentence_predict_discount(sentence, n):
    with open(f'Model\\{n}-gram_discount.json', 'r', encoding = 'utf-8') as file:
        ngram_probabilities = json.load(file)
    with open(f'Model\\{n}-gram_per_discount.json', 'r', encoding = 'utf-8') as file:
        ngram_probabilities_per = json.load(file)

    probability = 1
    for i in range(len(sentence) - n + 1):
        ngram_probability = ngram_probabilities.get(sentence[i : i + n], 0)
        if not ngram_probability:# 给定序列未出现过
            ngram_probability = ngram_probabilities_per.get(sentence[i : i + n - 1], 0)
            if not ngram_probability:# 给定历史未出现过
                ngram_probability = 1 / unique_char_num
        probability *= ngram_probability
    return probability

# 绝对减值法平滑续写句子
def next_word_predict_discount(sentence, num, n, corpus_counter):
    with open(f'Model\\{n}-gram_discount.json', 'r', encoding = 'utf-8') as file:
        ngram_probabilities = json.load(file)

    for i in range(num):
        history = sentence[-n + 1 :]
        probability_max = 0
        word_next = None
        candidates = []
        check = 0
        for ngram, probability in ngram_probabilities.items():
            if history == ngram[0 : n - 1] and probability_max <= probability:
                check = 1
                if probability_max == probability:# 概率相同则列为候选字符
                    candidates.append(ngram[-1])
                else:# 概率不同则替换为概率大者
                    probability_max = probability
                    candidates = [ngram[-1]]
        if check == 0:# 给定历史在语料中未出现，随机选取字符
            word_next = random.choice(list(corpus_counter.keys()))
        else:
            word_next = random.choice(candidates)
        sentence += str(word_next)

    return sentence

import json
import msvcrt
import glob
text_files = glob.glob('Text\\*.txt')

corpus = ""  # 初始化corpus变量

# 遍历找到的文件列表
for file_name in text_files:
    with open(file_name, 'r', encoding='utf-8') as file:
        text = file.read()  # 读取文件内容
        corpus += corpus_prepare(text)  # 对每个文件内容进行预处理并累加到corpus中

print('语料长度：' + str(len(corpus)))

# 计算不重复的字符个数
corpus_counter = Counter(corpus)
unique_char_num = len(corpus_counter)
print('1. 构建n-gram | 2. 根据已有n-gram开始预测 | 3. 退出程序')
while (1):
    choice = msvcrt.getch()
    if choice == b'1':

        while True:
            n = input('请输入n >= 2：')
            try:
                n = int(n)
                if n >= 2:
                    break  # n有效且 >= 2，退出循环
                else:
                    print("输入错误！n必须大于或等于2，请重新输入。")
            except ValueError:
                print("输入错误！请输入一个有效的数字。")

        # 构建序列字典
        ngram_sequences = build_ngram_sequences(corpus, n)
        # 去除人为增加的序列
        ngram_sequences = remove_unwanted(ngram_sequences)
        # 计算概率并存储
        ngram_add, ngram_per_add = add_one_smoothing(ngram_sequences, n)
        with open(f'Model\\{n}-gram_add.json', 'w', encoding = 'utf-8') as file:
            json.dump(ngram_add, file, ensure_ascii = False, indent = 4)
        with open(f'Model\\{n}-gram_per_add.json', 'w', encoding = 'utf-8') as file:
            json.dump(ngram_per_add, file, ensure_ascii = False, indent = 4)

        n1 = 0
        n2 = 0
        for ngram, count in ngram_sequences.items():
            if count == 1:
                n1 += 1
            elif count == 2:
                n2 += 1
        b_max = n1 / (n1 + 2 * n2)
        while True:
            b = input(f'请输入b <= {b_max}：')
            try:
                b = float(b)
                if b > 0 and b <= b_max:
                    break
                else:
                    print(f'输入错误！b必须大于0且小于等于{b_max}，请重新输入。')
            except ValueError:
                print("输入错误！请输入一个有效的数字。")

        ngram_discount, ngram_per_discount = absolute_discounting_smoothing(ngram_sequences, n, b)
        with open(f'Model\\{n}-gram_discount.json', 'w', encoding = 'utf-8') as file:
            json.dump(ngram_discount, file, ensure_ascii = False, indent = 4)
        with open(f'Model\\{n}-gram_per_discount.json', 'w', encoding = 'utf-8') as file:
            json.dump(ngram_per_discount, file, ensure_ascii = False, indent = 4)

        print('1. 构建n-gram | 2. 根据已有n-gram开始预测 | 3. 退出程序')
    elif choice == b'2':
        model_files = glob.glob('Model\\*.json')
        model_counter = Counter(model_file[6] for model_file in model_files)
        print('\n1.预测出现概率 | 2.预测后续语句 | 3.返回上一步')
        while (1):
            choice_1 = msvcrt.getch()
            if choice_1 == b'1':
                user_input = input('请输入要预测的语句：')
                input_str = 'B' + str(user_input) + 'E'
        
                for n, count in model_counter.items():
                    if count == 4 and len(input_str) >= int(n):
                        ngram_probability = sentence_predict_add(input_str, int(n))
                        print(f'{n}-gram加一法平滑概率：{ngram_probability:.10e}')
                        ngram_probability = sentence_predict_discount(input_str, int(n))
                        print(f'{n}-gram绝对减值法平滑概率：{ngram_probability:.10e}')
                    elif count != 4:
                        print(f'{n}-gram缺少模型文件！')
                    elif len(input_str) <= int(n):
                        print(f'未达到{n}-gram预测所需的最小语句长度{n - 2}')

            elif choice_1 == b'2':
                input_str = input('请输入开头：')

                for n, count in model_counter.items():
                    if count == 4 and len(input_str) >= int(n):
                        prediction = next_word_predict_add(input_str, 20, int(n), corpus_counter)
                        print(f'{n}-gram加一法平滑续写：{prediction}')
                        prediction = next_word_predict_discount(input_str, 20, int(n), corpus_counter)
                        print(f'{n}-gram绝对减值法平滑续写：{prediction}')
                    elif count != 4:
                        print(f'{n}-gram缺少模型文件！')
                    elif len(input_str) <= int(n):
                        print(f'未达到{n}-gram续写所需的最小语句长度{n}')

            elif choice_1 == b'3':
                break
            else:
                continue
            print('\n1.预测出现概率 | 2.预测后续语句 | 3.返回上一步')
        print('1. 构建n-gram | 2. 根据已有n-gram开始预测 | 3. 退出程序')
    elif choice == b'3':
        sys.exit(0)
    else:
        continue