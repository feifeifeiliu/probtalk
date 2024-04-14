import re
import numpy as np
import librosa
from interval import Interval
import torch
from transformers import AutoProcessor, HubertModel, Wav2Vec2Model, Speech2TextForConditionalGeneration
from transformers import AutoTokenizer, GPT2Model, CLIPModel, BertModel, T5ForConditionalGeneration


proxies = {
    # 'http': '127.0.0.1:7890',
    # 'https': '127.0.0.1:7890',
}


def split_sentences(line):
    line_split = re.split(r'[,.\'?!]', line.strip())
    line_split = [line.strip() for line in line_split if
                  line.strip() not in [',', '.', '\'', '?', '!'] and len(line.strip()) > 1]
    return line_split


def getFM(audio_name, text_name):
    if audio_name == 'hubert':
        aud_p = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft", proxies=proxies)
        aud_m = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft", proxies=proxies)
        aud_d = 1024
    elif audio_name in ['wav2vec', 'wav2vec_slice']:
        # try:
        #     aud_p = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h", proxies=proxies)
        #     aud_m = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h", proxies=proxies)
        # except:
        # aud_p = AutoProcessor.from_pretrained("/home/yifei/.cache/huggingface/hub/wav2vec/wav2vec", proxies=proxies)
        # aud_m = Wav2Vec2Model.from_pretrained("/home/yifei/.cache/huggingface/hub/wav2vec/wav2vec", proxies=proxies)
        aud_p = AutoProcessor.from_pretrained("D:\Downloads\wav2vec", proxies=proxies)
        aud_m = Wav2Vec2Model.from_pretrained("D:\Downloads\wav2vec", proxies=proxies)
        aud_d = 768
    elif audio_name == 'speech2text':
        aud_p = AutoProcessor.from_pretrained("facebook/s2t-small-librispeech-asr", proxies=proxies)
        aud_m = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-small-librispeech-asr", proxies=proxies)
        aud_d = 768
    elif audio_name == None:
        aud_p = aud_m = aud_d = None
    else:
        raise NameError("The audio model name is incorrect.")

    if text_name == 'gpt2':
        text_p = AutoTokenizer.from_pretrained("gpt2", proxies=proxies)
        text_m = GPT2Model.from_pretrained("gpt2", proxies=proxies)
        text_d = 768
        separator = 'Ġ'
    elif text_name == 'clip':
        text_p = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32", proxies=proxies)
        text_m = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", proxies=proxies)
        text_d = 512
    elif text_name == 'bert':
        text_p = AutoTokenizer.from_pretrained("bert-base-uncased", proxies=proxies)
        text_m = BertModel.from_pretrained("bert-base-uncased", proxies=proxies)
        text_d = 768
    elif text_name == 't5':
        text_p = AutoTokenizer.from_pretrained("t5-base", proxies=proxies)
        text_m = T5ForConditionalGeneration.from_pretrained("t5-base", proxies=proxies)
        text_d = 768
        separator = '▁'
    elif text_name == 'ton':
        text_p = AutoTokenizer.from_pretrained("gpt2", proxies=proxies)
        text_m = talk_or_not()
        text_d = 3
        separator = 'Ġ'
    elif text_name == 'fasttext':
        raise NotImplementedError
    elif text_name == None:
        separator = text_p = text_m = text_d = None
    else:
        raise NameError("The text model name is incorrect.")

    if aud_m is not None:
        aud_m = aud_m.to('cuda')
        aud_m.eval()
    if text_m is not None:
        text_m = text_m.to('cuda')
        text_m.eval()

    fm_dict = {}
    fm_dict['aud_n'] = audio_name
    fm_dict['aud_p'] = aud_p
    fm_dict['aud_m'] = aud_m
    fm_dict['aud_d'] = aud_d
    fm_dict['text_n'] = text_name
    fm_dict['text_p'] = text_p
    fm_dict['text_m'] = text_m
    fm_dict['text_d'] = text_d
    fm_dict['text_pad'] = np.zeros(fm_dict['text_d'])
    fm_dict['separator'] = separator
    if text_name == 'ton':
        fm_dict['text_pad'][-1] = 1
    return fm_dict


def get_textfeat(audio_name, file_name, fm_dict):
    time = librosa.get_duration(filename=audio_name)
    num_frame = int(time) * 30
    text_feat = np.zeros([num_frame, fm_dict['text_d']])

    pattern_sent = re.compile('(?<=Transcript: ).*')
    fh = open(file_name, 'r', encoding='utf-8', errors='ignore')

    length = 0
    l = len(fh.readlines())
    fh.seek(0)
    transcript_list = []
    len_script = []
    lines = []

    # get sentences, and the length of each sentence,
    for i in range(l):
        line = fh.readline()
        if pattern_sent.findall(line) != []:
            if i != 0:
                len_script.append(i - transcript_list[-1] - 1)
            transcript_list.append(i)
        else:
            length = length + 1
        lines.append(line)
    # fh.close()
    # if transcript_list == [] or length==0:
    #     # os.remove(audio_name)
    #     # os.remove(file_name)
    #     return True
    # elif num_frame < 150:
    #     return True
    len_script.append(i - transcript_list[-1])
    len_script = iter(len_script)
    fh.seek(0)
    fh.close()

    start = np.zeros(length)
    stop = np.zeros(length)
    intervals = []
    confidence = np.zeros(length)
    feats = np.zeros([length, fm_dict['text_d']])
    words = []

    line_st = 0
    d = 0

    # get the feat of each time interval
    for i1 in transcript_list:
        d = d + 1
        sentences = pattern_sent.findall(lines[i1])[0]
        len_sen = next(len_script)
        if sentences == '':
            continue
        tokens = fm_dict['text_p'](sentences)
        input_ids = torch.tensor([tokens.encodings[0].ids], dtype=torch.long, device='cuda')  # 一个输入也需要组batch

        if fm_dict['text_n'] == 'gpt2':
            # output = fm_dict['text_m'].wte(input_ids)
            output = fm_dict['text_m'](input_ids).last_hidden_state
        elif fm_dict['text_n'] == 't5':
            output = fm_dict['text_m'].encoder(input_ids)
            # output = fm_dict['text_m'].encoder.embed_tokens(input_ids)
        elif fm_dict['text_n'] == 'clip':
            output = fm_dict['text_m'].text_model(input_ids)
        elif fm_dict['text_n'] == 'ton':
            output = fm_dict['text_m'](input_ids)
        else:
            raise NotImplementedError
        state = output[:, :].cpu().detach().numpy()
        # state = output[:, :-1].cpu().detach().numpy()

        word_num = tokens.encodings[0].word_ids[-1] + 1 #tokens.encodings[0].word_ids[:-1][-1] + 1
        feat_words = np.zeros([word_num, fm_dict['text_d']])
        word_ids = np.array(tokens.encodings[0].word_ids)

        for i2 in range(word_num):
            feat = state[0, np.argwhere(word_ids == i2).squeeze()].reshape(-1, fm_dict['text_d']).mean(axis=0)
            feat_words[i2] = feat

        state = state.reshape(-1, fm_dict['text_d'])

        k = 0
        for i3 in range(1, len_sen + 1):
            sta, sto, word, conf = lines[i1 + i3].split('|')
            start[i1 + i3 - d] = float(sta)
            stop[i1 + i3 - d] = float(sto)
            # confidence[i1 + i3 - d] = float(conf.split(' ')[1][:2]) / 100
            confidence[i1 + i3 - d] = float(conf.split(' ')[1].strip("%")) / 100
            intervals.append(Interval(start[i1 + i3 - d], stop[i1 + i3 - d]))

            for j in range(k, state.shape[0]):
                if j == k:
                    feats[i1 + i3 - d] = feats[i1 + i3 - d] + state[j]
                elif fm_dict['separator'] not in tokens.encodings[0].tokens[j]:
                    feats[i1 + i3 - d] = feats[i1 + i3 - d] + state[j]
                else:
                    break
            feats[i1 + i3 - d] = feats[i1 + i3 - d] / max((j - k), 1.0) * confidence[i1 + i3 - d]
            if np.isinf(feats).sum() > 0:
                print('fuck')
            if np.isnan(feats).sum() > 0:
                print('fuck')
            k = j
        assert j == state.shape[0] - 1, "error"

    # get the feat of each frame
    j = 0
    for f in range(num_frame):
        if f / 30 in intervals[j]:
            text_feat[f] = feats[j]
        else:
            if j == feats.shape[0] - 1:
                text_feat[f] = fm_dict['text_pad']
            elif f / 30 in intervals[j + 1]:
                text_feat[f] = feats[j + 1]
                j = j + 1
            elif j < feats.shape[0] - 2:
                if f / 30 in intervals[j + 2]:
                    text_feat[f] = feats[j + 2]
                    j = j + 2
                else:
                    text_feat[f] = fm_dict['text_pad']
            else:
                text_feat[f] = fm_dict['text_pad']

    return text_feat


def get_textfeat_bin(audio_name, file_name, fm_dict):
    time = librosa.get_duration(filename=audio_name)
    num_frame = int(time) * 30
    text_feat = np.zeros([num_frame, fm_dict['text_d']])

    pattern_sent = re.compile('(?<=Transcript: ).*')
    fh = open(file_name, 'r', encoding='utf-8', errors='ignore')

    length = 0
    l = len(fh.readlines())
    fh.seek(0)
    transcript_list = []
    len_script = []
    lines = []

    # get sentences, and the length of each sentence,
    for i in range(l):
        line = fh.readline()
        if pattern_sent.findall(line) != []:
            if i != 0:
                len_script.append(i - transcript_list[-1] - 1)
            transcript_list.append(i)
        else:
            length = length + 1
        lines.append(line)
    len_script.append(i - transcript_list[-1])
    len_script = iter(len_script)
    fh.seek(0)
    fh.close()

    start = np.zeros(length)
    stop = np.zeros(length)
    intervals = []
    confis = []
    feats = None

    d = 0

    # get the feat of each time interval
    for i1 in transcript_list:
        d = d + 1
        sentences = pattern_sent.findall(lines[i1])[0]
        len_sen = next(len_script)
        if sentences == '':
            continue
        tokens = fm_dict['text_p'](sentences)
        input_ids = torch.tensor([tokens.encodings[0].ids], dtype=torch.long, device='cuda')  # 一个输入也需要组batch

        if fm_dict['text_n'] == 'gpt2':
            # output = fm_dict['text_m'].wte(input_ids)
            output = fm_dict['text_m'](input_ids).last_hidden_state
        elif fm_dict['text_n'] == 't5':
            output = fm_dict['text_m'].encoder(input_ids)
            # output = fm_dict['text_m'].encoder.embed_tokens(input_ids)
        elif fm_dict['text_n'] == 'clip':
            output = fm_dict['text_m'].text_model(input_ids)
        elif fm_dict['text_n'] == 'ton':
            output = fm_dict['text_m'](input_ids)
        else:
            raise NotImplementedError
        state = output[:, :].cpu().detach().numpy().reshape(-1, fm_dict['text_d'])
        # state = output[:, :-1].cpu().detach().numpy()
        if feats is None:
            feats = state
        else:
            feats = np.concatenate([feats, state], 0)

        for i3 in range(1, len_sen + 1):
            sta, sto, word, conf = lines[i1 + i3].split('|')
            start[i1 + i3 - d] = float(sta)
            stop[i1 + i3 - d] = float(sto)
            # confidence[i1 + i3 - d] = float(conf.split(' ')[1][:2]) / 100
            confidence = float(conf.split(' ')[1].strip("%")) / 100

        k = 0
        skip_i3 = 0
        for i3 in range(1, len_sen + 1):
            if skip_i3 > 0:
                skip_i3 = skip_i3 - 1
                continue
            n_intervals = 0
            interval = Interval(start[i1 + i3 - d], stop[i1 + i3 - d])
            n_intervals, k, j, skip_i3 = get_n_intervals(n_intervals, fm_dict, state, tokens, k, i1, i3, d, len_sen, start, stop, skip_i3)

            interval = split_interval(interval, n_intervals)
            for intv in interval:
                intervals.append(intv)
                confis.append(confidence)

    confis = np.asarray(confis)

    assert confis.shape[0] == feats.shape[0]
    assert intervals.__len__() == feats.shape[0]

    # get the feat of each frame
    j = 0
    for f in range(num_frame):
        if f / 30 in intervals[j]:
            text_feat[f] = feats[j] * confis[j]
        else:
            if j == feats.shape[0] - 1:
                text_feat[f] = fm_dict['text_pad']
            elif f / 30 in intervals[j + 1]:
                text_feat[f] = feats[j + 1] * confis[j]
                j = j + 1
            elif j < feats.shape[0] - 2:
                if f / 30 in intervals[j + 2]:
                    text_feat[f] = feats[j + 2] * confis[j]
                    j = j + 2
                else:
                    text_feat[f] = fm_dict['text_pad']
            else:
                text_feat[f] = fm_dict['text_pad']

    return text_feat


class talk_or_not:
    def __call__(self, input_ids):
        feat = torch.zeros([input_ids.shape[1], 3])
        feat[(input_ids[0] == 5) | (input_ids[0] == 6) | (input_ids[0] == 55) | (input_ids[0] == 58), 0] = 1
        feat[feat[:, 0] == 0, 1] = 1
        feat = feat.unsqueeze(0)
        return feat
    def to(self, args):
        return self
    def eval(self):
        pass


def split_interval(interval, n):
    # 计算子时间段的长度
    length = (interval.upper_bound - interval.lower_bound) / n

    # 生成子时间段
    sub_intervals = []
    for i in range(n):
        lower = interval.lower_bound + length * i
        upper = lower + length
        sub_intervals.append(Interval(lower, upper))

    return sub_intervals

def get_n_intervals(n_intervals, fm_dict, state, tokens, k, i1, i3, d, len_sen, start, stop, skip_i3):

    for j in range(k, state.shape[0]):
        if j == k:
            n_intervals = n_intervals + 1
        elif fm_dict['separator'] not in tokens.encodings[0].tokens[j]:
            n_intervals = n_intervals + 1
        else:
            break
    k = j

    if i1 + i3 - d + 1 < len_sen:
        if (start[i1 + i3 - d + 1] - stop[i1 + i3 - d + 1]) == 0:
            n_intervals, k, j, skip_i3 = get_n_intervals(n_intervals, fm_dict, state, tokens, k, i1, i3+1, d, len_sen, start, stop, skip_i3)
            skip_i3 = 1 + skip_i3
    k = j

    return n_intervals, k, j, skip_i3