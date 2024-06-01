!pip install torchtext==0.14.0
!pip install torchdata==0.5.0
!pip install torch==1.13.1
# 코드 실행 이후 Restart_runtime 해주시면 되겠습니다.

from torchtext.datasets import MNLI
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset, Dataset

import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.model_selection import train_test_split

# 데이터 생성 및 확인
dataset = list(MNLI(split='train'))
train_data, val_data = train_test_split(dataset, test_size=0.1, random_state=42)

test_data = list(MNLI(split='dev_matched'))

# 0: entailment, 1: neutral, 2: contradiction
tmp = train_data[0]
print(f"label:      {tmp[0]}")
print(f"premise:    {tmp[1]}")
print(f"hypothesis: {tmp[2]}")

# premise, hypothesis를 입력으로 해서 label을 맞추는 작업

len(train_data)

# 토크나이저
tokenizer = get_tokenizer('basic_english')

train_iter = iter(MNLI(split='train'))
test_iter = iter(MNLI(split='dev_matched'))

# 각 텍스트 문서에 대해서 토크나이징 진행
def yield_tokens(data_iter):
    for _, premise, hypothesis in data_iter:
        yield tokenizer(premise)
        yield tokenizer(hypothesis)

# 토크 나이징 한 리스트에서 어휘 사전을 구축
vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>", " <sep> "])
vocab.set_default_index(vocab["<unk>"])
vocab_size = len(vocab)
print(f"vocab_size: {vocab_size}")

# vocab의 역할: 텍스트를 정수입력 형태로 변환
vocab(['here', 'is', 'an', 'example'])

class TextClassificationModel(nn.Module):

    def __init__(self, vocab_size, embed_dim, num_class = 3):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc1 = nn.Linear(embed_dim, embed_dim)  # fully connected
        self.fc2 = nn.Linear(embed_dim, num_class)
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc1.weight.data.uniform_(-initrange, initrange)
        self.fc1.bias.data.zero_()
        self.fc2.weight.data.uniform_(-initrange, initrange)
        self.fc2.bias.data.zero_()

    def forward(self, text):
        embedded = self.embedding(text)
        embedded = self.fc1(embedded)
        embedded = self.relu(embedded)
        embedded = self.fc2(embedded)
        return embedded

def validate(model, criterion, valid_loader):
    model.eval()  # 모델을 평가 모드로 설정
    valid_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # 그래디언트 계산 비활성화
        for inputs, labels in valid_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            valid_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # 검증 데이터셋에 대한 손실과 정확도 계산
    valid_loss /= len(valid_loader)
    valid_accuracy = correct / total

    return valid_loss, valid_accuracy

class CustomTextDataset(Dataset):
    def __init__(self, data_list, tokenizer, vocab):
        self.data_list = data_list
        self.tokenizer = tokenizer
        self.vocab = vocab

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        label, premise, hypothesis = self.data_list[idx]
        inputs = premise + ' ' + hypothesis
        inputs = self.tokenizer(inputs)
        inputs = [self.vocab[token] for token in inputs]
        inputs = torch.tensor(inputs, dtype=torch.long)
        return inputs, torch.tensor(label, dtype=torch.long)

    def collate_fn(batch):
        inputs, labels = zip(*batch)
        # 패딩된 시퀀스 생성
        inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)
        labels_tensor = torch.stack(labels)
        return inputs_padded, labels_tensor

train_dataset = CustomTextDataset(train_data, tokenizer, vocab)
val_dataset = CustomTextDataset(val_data, tokenizer, vocab)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=CustomTextDataset.collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=CustomTextDataset.collate_fn)

emsize = 64
from torch.optim import SparseAdam

# 모델, 손실 함수, 옵티마이저, 스케줄러 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TextClassificationModel(vocab_size, emsize, num_class=3).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.3)
# SparseAdam, AdaGrad, RMSProp, Adadelta
#optimizer = SparseAdam(model.parameters(), lr=0.001)

#scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, threshold= 0.02 , patience = 5, min_lr=1e-4, verbose=True)
#scheduler = StepLR(optimizer, step_size=5, gamma=0.3)

import matplotlib.pyplot as plt

# gradient tensor 값 분포를 확인하기 위한 시각화
def visualize_gradient_distribution(grad_tensor):
    # 그래디언트 텐서를 1차원 배열로 변환하여 히스토그램 작성.
    grad_dense = grad_tensor.to_dense()
    grad_values = grad_tensor.detach().cpu().numpy().flatten()

    # 히스토그램 작성
    plt.figure(figsize=(8, 6))
    plt.hist(grad_values, bins=50, color='skyblue', alpha=0.7)
    plt.title('Gradient Distribution')
    plt.xlabel('Gradient Values')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

#scheduler = StepLR(optimizer, step_size=5, gamma=0.3)
max_epochs = 20
for epoch in range(max_epochs):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (i + 1) % 1000 == 0 or (i + 1) == len(train_loader):
            print(f'Epoch {epoch+1}/{max_epochs}, Batch {i+1}, Loss: {running_loss / min(1000, i+1):.3f}')
            running_loss = 0.0

    # 검증
    valid_loss, valid_accuracy = validate(model, criterion, val_loader)
    print(f'Epoch [{epoch+1}/{max_epochs}], Validation Loss: {valid_loss:.4f}, Validation Accuracy: {valid_accuracy:.4f}')

    # 학습률 체크
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Current learning rate: {current_lr}")
    if current_lr < 1e-5:
        print("Learning rate is below 1e-7. Stopping training.")
        break

    #scheduler.step(valid_accuracy)  # 에포크 마지막에서 업데이트

total_acc, total_count = 0, 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # gpu가속을 위함
model = model.to(device)

for i, data in enumerate(test_data):
    # 데이터 호출
    label = data[0]
    premise = data[1]
    hypothesis = data[2]

    # Forward pass
    inputs = premise + ' ' + hypothesis  # premise와 hypothesis를 모두 고려하기 위한 입력 구성
    inputs = vocab(tokenizer(inputs))  # 텍스트 형태의 입력을 모델이 이해 가능한 형태로 변환 (정수형)
    inputs = torch.as_tensor(inputs, dtype=torch.int32).unsqueeze(0)  # 모델이 받을 수 있는 데이터 형태로 변환
    label = torch.as_tensor(label, dtype=torch.long).unsqueeze(0) # 모델이 받을 수 있는 데이터 형태로 변환

    inputs = inputs.to(device)

    predicted_label = model(inputs)
    predicted_label = predicted_label.detach().cpu()

    total_acc += (predicted_label.argmax(1) == label).sum().item()
    total_count += label.size(0)

    if (i+1) % 1000 == 0:
        print(f'평가 진행중.. [{i+1}/{len(test_data)}]')

print(f"학습된 모델의 최종 정확도: {format(total_acc/total_count * 100, '.3f')} %")




