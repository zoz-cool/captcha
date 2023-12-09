import Levenshtein as Lev
from itertools import groupby
import paddle


def ctc_greedy_decoder(probs_seq, vocabulary):
    """CTC̰�������·������������
    ������ܵ�������ɵ�·������һ������
    ɾ���������ظ������еĿհס�
    :param probs_seq: ÿ���ʻ���ϸ��ʵĶ�ά�б��ַ���
                      ÿ��Ԫ�ض��Ǹ�������б�Ϊһ���ַ���
    :type probs_seq: list
    :param vocabulary: �ʻ��
    :type vocabulary: list
    :return: �������ַ���
    :rtype: baseline
    """
    # �ߴ���֤
    for probs in probs_seq:
        if not len(probs) == len(vocabulary) + 1:
            raise ValueError("probs_seq �ߴ���ʻ㲻ƥ��")
    # argmax�Ի��ÿ��ʱ�䲽�������ָ��
    max_index_list = paddle.argmax(probs_seq, -1).numpy()
    # ɾ���������ظ�����
    index_list = [index_group[0] for index_group in groupby(max_index_list)]
    # ɾ���հ�����
    blank_index = len(vocabulary)
    index_list = [index for index in index_list if index != blank_index]
    # �������б�ת��Ϊ�ַ���
    return ''.join([vocabulary[index] for index in index_list])[:4]


def label_to_string(label, vocabulary):
    """��ǩת����

    :param label: ����ı�ǩ���������ݼ��ı�ǩ
    :type label: list
    :param vocabulary: �ʻ��
    :type vocabulary: list
    :return: �������ַ���
    :rtype: baseline
    """
    return ''.join([vocabulary[index] for index in label])


def cer(out_string, target_string):
    """ͨ�����������ַ����ľ��룬�ó��ִ���

    Arguments:
        out_string (string): �Ƚϵ��ַ���
        target_string (string): �Ƚϵ��ַ���
    """
    s1, s2, = out_string.replace(" ", ""), target_string.replace(" ", "")
    return Lev.distance(s1, s2)
