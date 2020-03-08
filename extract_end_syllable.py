from preprocess import parse_line
import write_poems


def extract_syllable():

    '''
    extract end syllable from original poems and save in file
    '''

    file_name = './data/shakespeare.txt'
    output_name = './data/end_syllable.txt'
    file = open(file_name, 'r')
    lines = file.readlines()
    current_line_idx = 0

    output_file = open(output_name, 'a')
    while current_line_idx < len(lines):
        line = lines[current_line_idx].strip()
        current_line_idx += 1
        if line.isdigit():
            print('Processing %s ...' % line)
            syllable_list = []

            for i in range(14):
                line = lines[current_line_idx].strip()
                current_line_idx += 1
                word = parse_line(line)[-1]
                syllable = write_poems.get_last_syllable(word, 'word')
                syllable_list.append(syllable)
            if not any(len(syl) == 0 for syl in syllable_list):
                print(syllable_list)
                output_file.write(" ".join(syllable_list) + '\n')

    output_file.close()