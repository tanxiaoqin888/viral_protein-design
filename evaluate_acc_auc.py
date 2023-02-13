import os
import sys
import fasttext
import random


def write_file(file_path, lines):
    fw = open(file_path, "w", encoding="utf-8")
    for line in lines:
        fw.write(line + "\n")
    fw.close()


def change_training_data_format(filename, train_file, test_file):
    lines = open(filename, "r", encoding="utf-8").readlines()[1:]

    fits = [float(line.strip().split()[1]) for line in lines]
    avg = sum(fits) / len(fits)

    new_lines = []

    for line in lines:
        phrases = line.strip().split()
        viral = phrases[0].strip()[300: 600]
        viral = " ".join([char for char in viral])
        fit = float(phrases[1])
        if fit < avg:
            label = 0
        else:
            label = 1
        new_line = viral + "\t" + "__label__" + str(label)
        new_lines.append(new_line)

    random.shuffle(new_lines)
    test_lines = new_lines[: 5000]
    train_lines = new_lines[5000:]
    write_file(train_file, train_lines)
    write_file(test_file, test_lines)


def extract_generation_data(input_file, output_file):
    lines = open(input_file, "r", encoding="utf-8").readlines()[9: -3]
    fw = open(output_file, "w", encoding="utf-8")
    index = 0
    srcs, tgts, hypos = [], [], []
    while index < len(lines):
        hypo_line = lines[index + 2]
        index += 6

        hypo = hypo_line.strip().split()[2].strip()
        hypo = hypo.replace("<fit>", "").replace("<seq>", "").strip()
        label = "__label__" + hypo[0]
        text = hypo[1:]
        line = text + "\t" + label
        fw.write(line + "\n")
    fw.close()


def train_fitness_model(train_file, test_file, model="viral_protein_fit.bin", load=False):
    print("train_file=" + train_file + "\ttest_file=" + test_file + "\tmodel_file=" + model + "\tload=" + str(load))
    classifier = fasttext.train_supervised(
        input=train_file,
        label_prefix='__label__',
        dim=512,
        epoch=100,
        lr=0.5,
        lr_update_rate=50,
        min_count=3,
        loss='softmax',
        word_ngrams=4,
        bucket=2000000)

    # classifier = fasttext.train_supervised(
    #     input=train_file,
    #     label_prefix='__label__',
    #     dim=512,
    #     epoch=100,
    #     lr=1,
    #     lr_update_rate=50,
    #     min_count=3,
    #     loss='softmax',
    #     word_ngrams=4,
    #     bucket=2000000)

    if load:
        classifier = fasttext.load_model(model)
    else:
        classifier.save_model(model)

    result = classifier.test(test_file)
    print('测试集上数据量:' + str(result[0]))
    print('测试集上准确率: ' + str(result[1]))
    print('测试集上召回率: ' + str(result[2]))
    # file_data_pair = get_test_groundtruth(test_file)
    # predict_list = classifier.predict(file_data_pair[0])[0]
    # predict_labels = []
    # for predict_line in predict_list:
    #     predict_labels.append(predict_line[0])
    # true_labels = file_data_pair[1]


def get_test_groundtruth(file_path):
    fin = open(file_path, "r", encoding="utf-8")
    all_line = fin.readlines()
    true_labels = []
    all_text = []
    for one_line in all_line:
        pos = one_line.find("__label__")
        text = one_line[0:pos].strip()
        all_text.append(text)
        true_label = one_line[pos:].strip()
        true_labels.append(true_label)
    return all_text, true_labels


def compute_accuracy(test_file, model_path):
    classifier = fasttext.load_model(model_path)

    file_data_pair = get_test_groundtruth(test_file)
    predict_results = classifier.predict(file_data_pair[0])
    predict_list = predict_results[0]
    predict_scores = predict_results[1]
    scores = [score[0] - 0.1 for score in predict_scores]
    predict_labels = []
    for predict_line in predict_list:
        predict_labels.append(predict_line[0])
    new_scores = []
    for score, pred in zip(scores, predict_list):
        if pred == "__label__0":
            new_scores.append(score)
        else:
            new_scores.append(1 - score)

    true_labels = file_data_pair[1]
    num = 0
    for i in range(len(true_labels)):
        if true_labels[i] == predict_labels[i]:
            num += 1
    acc = float(num) / len(true_labels)
    print("Accuracy: %f" % acc)

    labels = [0 if label == "__label__0" else 1 for label in true_labels]
    return new_scores, labels


def calAUC(prob, labels):
    f = list(zip(prob, labels))
    rank = [values2 for values1, values2 in sorted(f, key=lambda x: x[0])]
    rankList = [i + 1 for i in range(len(rank)) if rank[i] == 1]
    posNum = 0
    negNum = 0
    for i in range(len(labels)):
        if (labels[i] == 1):
            posNum += 1
        else:
            negNum += 1
    auc = 0
    auc = (sum(rankList) - (posNum * (posNum + 1)) / 2) / (posNum * negNum)
    print(auc)
    return auc


if __name__ == "__main__":
    # data = "./generation/all_data_v2"
    # input_path = os.path.join(data, "generate-test.txt")
    # otuput_path = os.path.join(data, "generation.all.txt")
    # extract_generation_data(input_path, otuput_path)

    data_path = "../data/sars_cov2"
    file_name = os.path.join(data_path, "sars_cov_2_fitness_105526_combinational.txt")
    train_file = os.path.join(data_path, "train.sars.txt")
    test_file = os.path.join(data_path, "test.sars.txt")
    # change_training_data_format(file_name, train_file, test_file)
    model = "../sars.classification.bin"
    train_fitness_model(train_file, test_file, model, load=False)
    # test_file = "./data/train.txt"
    # scores, labels = compute_accuracy(test_file, model)
    # file_path = "../generation_bert_baseline/generation.bert.baseline.txt"
    # lines = open(file_path, "r", encoding="utf-8").readlines()[-2: ]
    # preds = eval(lines[0].strip())
    # labels = eval(lines[1].strip())
    # calAUC(preds, labels)
