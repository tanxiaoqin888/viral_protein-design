import os
import sys
import numpy as np

amino_dict = {}
index_dict = {}
single_dict = {}
pair_dict = {}


def read_data(input_path):
    return open(input_path, "r", encoding="utf-8").readlines()


def initialize_dict(start_index=0):
    lines = read_data("./data/dict.txt")
    for line in lines:
        char = line.strip().split()[0].strip()
        amino_dict[char] = start_index
        index_dict[start_index] = char
        start_index += 1
    amino_dict["-"] = start_index
    index_dict[start_index] = "-"
    return amino_dict, index_dict


def get_single_dict(input_path, output_path):
    lines = read_data(input_path)
    length = len(lines[0].strip())

    for i in range(length):
        single_dict[i] = np.zeros(21)

    for line in lines:
        line = line.strip()
        for pos, char in enumerate(line):
            single_dict[pos][amino_dict[char]] += 1

    for pos in single_dict:
        if np.sum(single_dict[pos]) == 0:
            print(pos)
        single_dict[pos] = single_dict[pos] / np.sum(single_dict[pos])

    wild_type = lines[0].strip()
    single_prob = []
    for i, char in enumerate(wild_type):
        if char == "-":
            continue
        single_prob.append(single_dict[i])
    single_prob = np.array(single_prob)
    print(np.shape(single_prob))
    print(len(single_prob))
    print(len(single_prob[0]))
    np.save(output_path, single_prob)


def get_pair_dict(input_path, output_path):
    lines = read_data(input_path)
    length = len(lines[0].strip())

    for i in range(length):
        pair_dict[i] = np.zeros((length, 21 * 21))

    for line in lines:
        line = line.strip()
        assert len(line) == length

        for m in range(length):
            for n in range(length):
                index = amino_dict[line[m]] * 21 + amino_dict[line[n]]
                pair_dict[m][n][index] += 1

    for i in range(length):
        for j in range(length):
            pair_dict[i][j] = pair_dict[i][j] / np.sum(pair_dict[i][j])

    for i in range(length):
        pair_dict[i][i] = np.zeros_like(pair_dict[i][i])

    find_pos = []
    wild_type = lines[0].strip()
    for i in range(len(wild_type)):
        if wild_type[i] == "-":
            find_pos.append(i)

    pair_prob = []
    new_pair_dict = dict()
    for i in range(length):
        this_prob = []
        for j in range(length):
            if j in find_pos:
                continue
            this_prob.append(pair_dict[i][j])
        new_pair_dict[i] = this_prob

    for i in range(length):
        if i in find_pos:
            continue
        pair_prob.append(new_pair_dict[i])

    pair_prob = np.array(pair_prob)
    print(np.shape(pair_prob))
    np.save(output_path, pair_prob)


def reform_msa_data(input_path, output_path):
    lines = read_data(input_path)
    new_lines = []
    index = 0
    length = len(lines)

    while index < length:
        if lines[index][0] == ">":
            this_lines = []
            index += 1
            while index < length and lines[index][0] != ">":
                this_lines.append(lines[index])
                index += 1
            new_lines.append(this_lines)

    updated_lines = []
    for seqs in new_lines:
        seqs = [seq.strip() for seq in seqs]
        line = "".join(seqs)
        updated_lines.append(line)

    assertion = len(updated_lines[0])
    for line in updated_lines:
        if len(line) != assertion:
            print("Error")

    fw = open(output_path, "w", encoding="utf-8")
    for line in updated_lines:
        fw.write(line + "\n")
    fw.close()


def valid_filtering(line, special_tokens):
    line = line.strip()
    for char in line:
        if char in special_tokens:
            return True
    return False


def filter_data(input_path, output_path):
    lines = read_data(input_path)
    fw = open(output_path, "w", encoding="utf-8")
    special_tokens = ["X", "B", "Z", "J"]
    num = 0

    for line in lines:
        if valid_filtering(line, special_tokens):
            num += 1
            continue
        fw.write(line)
    print(num)
    fw.close()


if __name__ == "__main__":
    data_path = "./data/msa"
    amino_dict, index_dict = initialize_dict()
    msa_path = os.path.join(data_path, "sars.msa.supervision.filtering.txt")
    single_pos_path = os.path.join(data_path, "sars.msa.pair")
    get_pair_dict(msa_path, single_pos_path)
