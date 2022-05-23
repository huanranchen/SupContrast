import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
from tqdm import tqdm


def hook(module, grad_in, grad_out):
    print('-' * 100)
    print(module)
    print(f'grad in {grad_in}, grad out {grad_out}')


class SupervisedContrast():
    def __init__(self, loader, num_classes=60):
        self.num_classes = num_classes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # student don't have momentum while teacher have
        self.student = models.wide_resnet50_2(num_classes=256).to(self.device)
        self.teacher = models.wide_resnet50_2(num_classes=256).to(self.device)
        self.momentum_synchronize(m=0)
        self.load_model()
        self.loader = loader
        self.initialize_queue()

    def load_model(self):
        if os.path.exists('teacher.pth'):
            self.teacher.load_state_dict(torch.load('teacher.pth', map_location=self.device))
            print('managed to load teacher')
            print('-' * 100)
        if os.path.exists('student.pth'):
            self.student.load_state_dict(torch.load('student.pth', map_location=self.device))
            print('managed to load student')
            print('-' * 100)

    def initialize_queue(self, queue_size=10):
        '''

        :param queue_size: total samples = queue_size*batch_size
        :return:
        '''
        self.queue = {}
        for i in range(self.num_classes):
            self.queue[i] = []

        self.enqueue(queue_size)
        print('managed to initialize queue!')
        print('-' * 100)

    def enqueue(self, size=1):
        '''

        :param size: queue_size*batch_size
        :return:
        '''
        with torch.no_grad():
            self.teacher.eval()
            for step, (x, y) in enumerate(self.loader):
                if step >= size:
                    break
                x = x.to(self.device)
                y = y.to(self.device)
                for i in range(self.num_classes):
                    mask = y == i
                    self.queue[i].append(self.teacher(x[mask]))

    def dequeue(self, size=1):
        for i in range(self.num_classes):
            for _ in range(size):
                self.queue[i].pop(0)

    def momentum_synchronize(self, m=0.9):
        for s, t in zip(self.student.parameters(), self.teacher.parameters()):
            t.data = m * t.data + (1 - m) * s.data
            t.requires_grad = False

    def get_queue(self):
        y = []
        x = []
        for i in range(self.num_classes):
            now_x = torch.cat(self.queue[i], dim=0)
            x.append(now_x)
            y += [i] * now_x.shape[0]
        return torch.cat(x, dim=0), torch.tensor(y, device=self.device)

    def train(self, lr=1e-4, weight_decay=0, t=1, total_epoch=100):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        criterion = self.chr_loss
        optimizer = torch.optim.AdamW(self.student.parameters(), lr=lr, weight_decay=weight_decay)
        print('now we start training!!!')
        for epoch in range(1, total_epoch + 1):
            self.student.train()
            train_loss = 0
            step = 0
            pbar = tqdm(self.loader)
            for x, y in pbar:
                x = x.to(device)
                y = y.to(device)
                x = self.student(x)  # N, 60
                queue_x, queue_y = self.get_queue()
                x = torch.cat([x, queue_x], dim=0)
                y = torch.cat([y, queue_y], dim=0)
                loss = criterion(x, y, t=t)
                train_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(self.student.parameters(), 0.1)
                optimizer.step()
                step += 1
                # scheduler.step()
                if step % 10 == 0:
                    pbar.set_postfix_str(f'loss = {train_loss / step}')
                self.enqueue()
                self.dequeue()
                self.momentum_synchronize()

            train_loss /= len(self.loader)
            print(f'epoch {epoch}, test loader loss = {train_loss}')

    def save_model(self):
        torch.save(self.student.state_dict(), 'student.pth')
        torch.save(self.teacher.state_dict(), 'teacher.pth')

    def chr_loss(self, logits, labels, t=1, ):
        '''

        :param logits: N, D
        :param labels:N
        :return:
        '''
        logits = F.normalize(logits, dim=1)
        logits = logits @ logits.permute(1, 0) / t  # N, N
        max_logits, _ = torch.max(logits, dim=1)
        logits -= max_logits

        labels = labels.reshape(-1, 1)
        mask = labels == labels.T  # N, N of where is positive
        logit_mask = torch.ones_like(logits) - torch.eye(logits.shape[0],
                                                         device=self.device)  # N,N of where not same with self
        mask = mask * logit_mask
        exponential_logits = F.softmax(logits * logit_mask, dim=1)
        denominator = torch.log(torch.sum(exponential_logits, dim=1))
        log_probs = logits - denominator
        if torch.sum(mask) == 0:
            return torch.sum(mask * log_probs)
        loss = -torch.sum(mask * log_probs) / 128
        return loss


if __name__ == '__main__':
    import argparse

    paser = argparse.ArgumentParser()
    paser.add_argument('-b', '--batch_size', default=128)
    paser.add_argument('-t', '--total_epoch', default=10)
    paser.add_argument('-l', '--lr', default=1e-4)
    args = paser.parse_args()
    batch_size = int(args.batch_size)
    total_epoch = int(args.total_epoch)
    lr = float(args.lr)

    train_image_path = './public_dg_0416/train/'
    valid_image_path = './public_dg_0416/train/'
    label2id_path = './dg_label_id_mapping.json'
    test_image_path = './public_dg_0416/public_test_flat/'
    from data.data import get_loader

    train_loader = get_loader(batch_size=batch_size,
                              valid_category=None,
                              train_image_path=train_image_path,
                              valid_image_path=valid_image_path,
                              label2id_path=label2id_path)
    a = SupervisedContrast(train_loader)
    a.train(total_epoch=total_epoch, lr = lr)
