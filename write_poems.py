import os
import extract_end_syllable
from collections import defaultdict

def get_last_word(sentence):
    '''
    :param sentence: str
    :return: last_word: str, the last word of sentence
    '''
    sentence = sentence.replace('.', '').replace('?', '')
    last_word = sentence.split()[-1]
    return last_word


def get_last_syllable(input, word_syllable_dict, tag='sentence'):
    '''

    :param input: str, a word with tag being 'word' or a sentence with tag being 'sentence',
    :param word_syllable_dict: a dictionary with key = word, value = last syllable
    :param tag: demonstrate the type of input, 'word' or 'sentence'
    :return: str, the last syllable of current sentence or word
    '''
    cur_word = get_last_word(input) if tag == 'sentence' else input
    return word_syllable_dict.get(cur_word, '')


def classify_sentence(sentences):
    sentence_class = defaultdict(list)
    word_syllable_dict = read_word_syllable()

    for sentence in sentences:
        sentence_end_syllable = get_last_syllable(sentence, word_syllable_dict)
        print(sentence_end_syllable)
        print(sentence)
        print('----------------')
        sentence_class[sentence_end_syllable].append(sentence)
    return sentence_class


def count_sentence_syllables(sentence, word_id_dict, syllables_dict):
    '''
    :param sentence: str, sentence
    :param word_id_dict: dict
    :param syllables_dict: dict
    :return: syllable_count: int, the count of syllable in this sentence
    '''
    sentence = sentence.replace('.', '').replace('?', '')
    words = sentence.split()
    syllable_count = 0
    for idx, cur_word in enumerate(words):
        word_id = word_id_dict[cur_word]
        if idx == len(words)-1:
            syllable_count += syllables_dict[word_id][1]
        else:
            syllable_count += syllables_dict[word_id][0]
    return syllable_count


def read_syllable_template(syllable_path='./data/end_syllable.txt'):
    if not os.path.exists(syllable_path):
        extract_end_syllable.extract_syllable()
    file = open(syllable_path, 'r')
    lines = file.readlines()
    syllable_list = []
    for line in lines:
        line = line.strip()
        syllable_list.append(line.split(" "))
    return syllable_list


def read_word_syllable(syllable_path='./data/word_syllable.txt'):
    if not os.path.exists(syllable_path):
        extract_end_syllable.extract_syllable()
    file = open(syllable_path, 'r')
    lines = file.readlines()
    word_syllable_dict = {}
    for line in lines:
        word, syl = line.strip().split(" ")
        word_syllable_dict[word] = syl
    return word_syllable_dict


def truncate_sentence(sentence, word_id_dict, syllables_dict):
    '''
    :param sentence: str, sentence
    :param word_id_dict: dict
    :param syllables_dict: dict
    :return truncated sentence with exactly 10 syllables if it exists, else return empty string
    '''
    sentence = sentence.replace('.', '').replace('?', '')
    words = sentence.split()
    syllable_count = 0
    for idx, cur_word in enumerate(words):
        cur_word = cur_word.lower()
        word_id = word_id_dict[cur_word]

        if cur_word not in word_id_dict:
            print(cur_word)
            return ""
        count_end = syllables_dict[word_id][1]
        count_not_end = syllables_dict[word_id][0]

        if syllable_count + count_end == 10:
            return " ".join(words[:idx+1])
        elif syllable_count + count_end > 10:
            return ""
        elif syllable_count + count_not_end < 10:
            syllable_count += count_not_end


def format_poem(poem):
    for i in range(0, 14):
        poem[i] = poem[i].replace(' i ', ' I ')

        if i == 3 or i == 7 or i == 11:
            poem[i] += ':'
        elif i == 13:
            poem[i] += '.'
        else:
            poem[i] += ','
        
        poem[i] = poem[i].capitalize()
    return poem


if __name__ == '__main__':
    word = 'love'
    print(get_last_syllable(word))
