import nltk
from nltk.corpus import cmudict
# try:
#     nltk.data.find('cmudict')
# except LookupError:
#     nltk.download('cmudict')


def get_last_word(sentence):
    sentence = sentence.replace('.', '').replace('?', '')
    last_word = sentence.split()[-1]
    return last_word


def get_last_syllable(sentence):
    last_word = get_last_word(sentence)
    syllabus = cmudict.dict().get(last_word, [])
    return syllabus[0][-1] if syllabus else ''


def count_sentence_syllables(sentence, word_id_dict, syllables_dict):
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


if __name__ == '__main__':
    word = 'love'
    print(get_last_syllable(word))