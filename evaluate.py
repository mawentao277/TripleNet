from keras.callbacks import Callback
from preprocess import data_generator, load_test_label
import codecs
import json
import numpy as np
import os


def evaluate_ubuntu(config, model, test_size=500000):
    assert test_size % config.test_batch_size == 0
    predict_result = model.predict_generator(generator=data_generator(config=config, is_train=False),
                                             steps=test_size/config.test_batch_size)
    example_id_list, label_list = load_test_label(config)

    acc_right_num = acc_all_num = 0
    prediction_dict = {}
    for example_id, prediction, label in zip(example_id_list, predict_result, label_list):
        if example_id not in prediction_dict:
            prediction_dict[example_id] = [(prediction, label)]
        else:
            prediction_dict[example_id].append((prediction, label))

        if (prediction > 0.5 and label == 1) or (prediction < 0.5 and label == 0):
            acc_right_num += 1
        acc_all_num += 1

    print(f"acc evaluate acc_right_num: {acc_right_num}\tacc_all_num: {acc_all_num}\tacc: "
          f"{acc_right_num * 1.0 / acc_all_num}")

    recall_ten_at_one = recall_ten_at_two = recall_ten_at_fiv = recall_two_at_one = sample_all_num = 0
    for example_id in prediction_dict:
        prediction_list = prediction_dict[example_id]
        label_pred = prediction_list[0][0]
        sec_pred = prediction_list[1][0]

        if label_pred > sec_pred:
            recall_two_at_one += 1

        sorted_list = sorted(prediction_list, key=lambda x: x[0], reverse=True)
        sorted_label_list = [y for x, y in sorted_list]

        if sorted_label_list[0] == 1:
            recall_ten_at_one += 1
            recall_ten_at_two += 1
            recall_ten_at_fiv += 1
        elif 1 in sorted_label_list[:2]:
            recall_ten_at_two += 1
            recall_ten_at_fiv += 1
        elif 1 in sorted_label_list[:5]:
            recall_ten_at_fiv += 1
        else:
            pass
        sample_all_num += 1

    recall_two_at_one = recall_two_at_one * 1.0 / sample_all_num
    recall_ten_at_one = recall_ten_at_one * 1.0 / sample_all_num
    recall_ten_at_two = recall_ten_at_two * 1.0 / sample_all_num
    recall_ten_at_fiv = recall_ten_at_fiv * 1.0 / sample_all_num

    print(f"rank evaluate sample_all_num: {sample_all_num}\trecall_two_at_one: {recall_two_at_one}\trecall_ten_at_one: {recall_ten_at_one}\t" \
          f"recall_ten_at_two: {recall_ten_at_two}\trecall_ten_at_fiv: {recall_ten_at_fiv}")

    return recall_ten_at_one


def evaluate_douban(config, model, test_size=10000):
    assert test_size % config.test_batch_size == 0
    predict_result = model.predict_generator(generator=data_generator(config=config, is_train=False), \
        steps=test_size / config.test_batch_size)
    example_id_list, label_list = load_test_label(config)

    prediction_dict = {}
    for example_id, prediction, label in zip(example_id_list, predict_result, label_list):
        if example_id not in prediction_dict:
            prediction_dict[example_id] = [(prediction, label)]
        else:
            prediction_dict[example_id].append((prediction, label))

    # del some invalid example
    del_num = 0
    filtered_prediction_dict = {}

    for example_id in prediction_dict.keys():
        temp_list = prediction_dict[example_id]
        if len(temp_list) != 10:
            print(len(temp_list))
            print(example_id)
            print('ERROR')
            print('############')
        label0_num = 0
        label1_num = 0

        for temp in temp_list:
            if temp[1] == 0:
                label0_num += 1
            if temp[1] == 1:
                label1_num += 1

        if label0_num == 10 or label1_num == 10:
            del_num += 1
        else:
            filtered_prediction_dict[example_id] = temp_list

    print(f'there are {del_num} example have been delete')

    # now calculate each metrics
    mrr_list = []
    map_list = []
    recall_1 = 0
    recall_2 = 0
    recall_5 = 0
    p1 = 0
    example_count = 0
    for example_id in filtered_prediction_dict.keys():
        prediction_list = filtered_prediction_dict[example_id]

        # (score, label)
        prediction_list = sorted(prediction_list, key=lambda x: x[0], reverse=True)

        total_positive = 0
        for prediction in prediction_list:
            if prediction[1] == 1:
                total_positive += 1
        if prediction_list[0][1] == 1:
            p1 += 1
            recall_1 += 1 * 1.0 / total_positive
        correct = 0
        for i in range(2):
            if prediction_list[i][1] == 1:
                correct += 1
        recall_2 += correct * 1.0 / total_positive
        correct = 0
        for i in range(5):
            if prediction_list[i][1] == 1:
                correct += 1
        recall_5 += correct * 1.0 / total_positive

        for i in range(len(prediction_list)):
            if prediction_list[i][1] == 1:
                mrr_list.append(1 * 1.0 / (i+1))
                break

        correct_count = 1
        one_map_list = []
        for i in range(len(prediction_list)):
            if prediction_list[i][1] == 1:
                one_map_list.append(correct_count * 1.0 / (i + 1))
                correct_count += 1
        map_list.append(sum(one_map_list) / total_positive)

        example_count += 1

    MRR = sum(mrr_list) / example_count
    MAP = sum(map_list) / example_count
    P1 = p1 / example_count

    R10_1 = recall_1 / example_count
    R10_2 = recall_2 / example_count
    R10_5 = recall_5 / example_count

    print(f'rank evaluate total:{example_count}\tMRR:{MRR}\tMAP:{MAP}\tP1:{P1}\tR10@1:{R10_1}\tR10@2:{R10_2}\tR10@5:{R10_5}')
    return MRR

