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
                    quatrain_lists.append(observations)
                elif i < 12:
                    volta_lists.append(observations)
                else:
                    couplet_lists.append(observations)

    return quatrain_lists, volta_lists, couplet_lists, word_to_int, int_to_word


if __name__ == '__main__':
    quatrain_lists, volta_lists, couplet_lists, word_to_int, int_to_word = parse_data('data/shakespeare.txt')
    print(1)
