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
    filename = "./data/Syllable_dictionary.txt"
    word_id_dict = {'true-telling': 5, 'unrespected': 1, 'heart\'s': 1}
    print(syllables_interpreter(filename, word_id_dict))