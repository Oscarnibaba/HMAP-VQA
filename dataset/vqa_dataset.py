import os
import json
from PIL import Image
from torch.utils.data import Dataset
from .utils import pre_question, pre_answer
import random

class vqa_dataset(Dataset):
    def __init__(self, ann_file, transform, vqa_root, eos='[SEP]', split="train", max_ques_words=40,
                 answer_list='', sample_fraction=1.0):
        self.split = split
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f, 'r'))

        if 0 < sample_fraction < 1.0:  # 如果 sample_fraction 是 1.0，则使用完整数据
            subset_size = int(len(self.ann) * sample_fraction)
            self.ann = random.sample(self.ann, subset_size)

        self.transform = transform
        self.vqa_root = vqa_root
        self.max_ques_words = max_ques_words
        self.eos = eos

        if split == 'test':
            self.max_ques_words = 50  # do not limit question length during test
            self.answer_list = json.load(open(answer_list, 'r'))

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        ann = self.ann[index]

        image_path = os.path.join(self.vqa_root, ann['image_name'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        # prompt_info = ', its organ is {}, the type of answer is {}, the type of question is {}'\
        #               .format(ann['image_organ'], ann['answer_type'], ann['question_type'])
        # ann['question'] = ann['question'] + prompt_info


        if self.split == 'test':
            question = pre_question(ann['question'], self.max_ques_words)
            promt_question = ann['question_type'] + ' ' + question
            # 'question_type: ' +
            question_id = ann['qid']
            return image, promt_question, question_id

        elif self.split == 'train':

            question = pre_question(ann['question'], self.max_ques_words)
            # print(question)
            promt_question = ann['question_type'] + ' ' + question
            # 'question_type: ' +
            # print(promt_question)

            answers = ann['answer']
            answers = [pre_answer(answers)]
            answers = [answer + self.eos for answer in answers]

            return image, promt_question, answers