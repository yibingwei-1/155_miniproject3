import HMM
import HMM_helper


def parse_line(line):
    pure_line = ''

    for ch in line:
        if ord('A') <= ord(ch) <= ord('Z'):
            ch = ch.lower()
            pure_line += ch
        elif ord('a') <= ord(ch) <= ord('z') or ch == ' ':
            pure_line += ch

    words = pure_line.split()

    return words


def parse_data(file_name):
    # initialize
    quatrain_lists = []
    volta_lists = []
    couplet_lists = []
    poem_lists = []
    word_to_int = {}
    int_to_word = {}
    current_word_idx = 0

    # read from 'shakespeare.txt'
    file = open(file_name, 'r')
    lines = file.readlines()
    current_line_idx = 0

    while current_line_idx < len(lines):
        line = lines[current_line_idx].strip()
        current_line_idx += 1
        if line.isdigit():

            quatrain_list = []
            volta_list = []
            couplet_list = []
            poem_list = []

            for i in range(14):

                line = lines[current_line_idx].strip()
                current_line_idx += 1

                words = parse_line(line)

                observations = []

                for word in words:
                    if word not in word_to_int:
                        word_to_int[word] = current_word_idx
                        int_to_word[current_word_idx] = word
                        current_word_idx += 1
                    observations.append(word_to_int[word])

                if 0 <= i < 8:
                    quatrain_list.extend(observations)
                elif i < 12:
                    volta_list.extend(observations)
                else:
                    couplet_list.extend(observations)
                poem_list.extend(observations)

            quatrain_lists.append(quatrain_list)
            volta_lists.append(volta_list)
            couplet_lists.append(couplet_list)
            poem_lists.append(poem_list)

    return poem_lists, quatrain_lists, volta_lists, couplet_lists, word_to_int, int_to_word


def syllables_interpreter(filename, word_id_dict):

    """
    :param filename:
                    str, file path to syllable_dictionary.txt
    :param word_id_dict:
                    dict, dictionary with key(str) = word, value(str or int) = word_id
    :return: syllables_dict:
                    dict, dictionary with key(str) = word_id,
                    value(list of int) = [not_end_syllable, end_syllable]

    """
    end_tag = "E"
    with open(filename) as file:
        syllables_dict = {}
        for line in file:
            if not line:
                continue
            data = line.strip().split()
            word = data[0]
            if word in word_id_dict:
                word_id = word_id_dict[word]
                end_syllable, not_end_syllable = -1, -1
                for item in data[1:]:
                    if item[0] == end_tag:
                        end_syllable = int(item[1:])
                    else:
                        not_end_syllable = int(item)
                end_syllable = not_end_syllable if end_syllable == -1 else end_syllable
                syllables_dict[word_id] = [not_end_syllable, end_syllable]
    return syllables_dict


if __name__ == '__main__':
    poem_lists, quatrain_lists, volta_lists, couplet_lists, word_to_int, int_to_word = parse_data('data/shakespeare.txt')
    filename = "./data/Syllable_dictionary.txt"
    word_id_dict = {'true-telling': 5, 'unrespected': 1, 'heart\'s': 1}
    print(syllables_interpreter(filename, word_id_dict))
