import math
import torch
from torch import optim
from torch.nn import functional as F
from torcheval.metrics.functional import multiclass_f1_score
from torcheval.metrics.functional import multiclass_precision
from torcheval.metrics.functional import multiclass_recall



def train(_loader, _test_loader, ep, mod, path):  # функция для обучения модели
    er = []
    maxF1 = 0
    opt = optim.Adam(mod.parameters(), lr=0.01)  # задаем оптимизатор
    for epoch in range(ep + 1):  # задаем количество эпох

        total_loss = 0  # обнуляем ошибку
        mod.train()  # переводим модель в режим тренировки
        for batch in _loader:
            opt.zero_grad()  # обнуляем градиенты
            embedding, pred = mod(batch)  # получаем предсказание для батча
            label = batch.y  # получаем метки для батча
            loss = F.nll_loss(pred, label)
            loss.backward()  # обратное распространение ошибки
            opt.step()  # проводим шаг оптимизации
            total_loss += loss.item() * len(batch)
        total_loss /= len(_loader.dataset)
        er.append(total_loss)
        print('epoch: {}, loss: {}'.format(epoch, total_loss))  # выводим ошибку для эпохи
        if epoch % 5 == 0:
            test_acc, test_f1, test_precision, = test(mod, _test_loader)
            print("Epoch {}. Loss: {:.4f}. Test accuracy: {:.4f}. F1: {:.4f} Precision: {:.4f}".format(
                epoch, total_loss, test_acc, test_f1, test_precision))
        if test_f1 > maxF1:
            torch.save(mod.state_dict(), path)
            maxF1 = test_f1
    return er


def test(mod, _test_loader):  # функция проверки на тестовом наборе данных
    mod.eval()
    correct = 0
    total_f1 = 0
    total_precision = 0
    total = len(_test_loader.dataset)
    for data in _test_loader:
        with torch.no_grad():
            emb, pred = mod(data)
            pred = pred.argmax(dim=1)
            # arr[pred] += 1
            label = data.y
            correct += pred.eq(label).sum().item()
            f1 = multiclass_f1_score(pred, label, num_classes=4, average="macro")
            precision = multiclass_precision(pred, label, num_classes=4, average="macro")
            total_f1 += f1.item() * len(data)
            total_precision += precision.item() * len(data)
    return correct / total, total_f1 / total, total_precision / total, 