## 1. Module import
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import random


## 2. GPU 연산이 간으하다면 GPU 연산을 하고, 그렇지 못하면 CPU 연산
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print('Using PyTorch version:', torch.__version__, '| 학습 Device:', device)
print('-------------------------------------------------------------------')


## 3. 하이퍼파라미터 변수로 두기
training_epochs = 20
# epoch: 전체 훈련 데이터가 학습에 한번 사용된 주기
BATCH_SIZE = 32


## 4. MNIST dataset
mnist_train = dsets.MNIST(root='MNIST_data/', train=True, transform=transforms.ToTensor(), download=True)
mnist_test = dsets.MNIST(root='MNIST_data/', train=False, transform=transforms.ToTensor(), download=True)
# root: MNIST 데이터를 다운받을 경로
# train: True일 시 MNIST의 훈련 데이터를 리턴받으며, False일 시 테스트 데이터를 리턴받는다.
# transform: 현재 데이터를 파이토치 텐서로 변환해준다.
# download: 해당 경로에 MNIST 데이터가 없다면 다운로드받겠다는 의미이다.


## 5. dataset loader
data_loader = DataLoader(dataset=mnist_train, batch_size=BATCH_SIZE, shuffle=False)
# dataset: 로드할 대상
# batch_size: aocl zmrl
# shuffle: 매 에포크마다 미니배치를 셔플할 것인지 여부


'''
## 6. batch 데이터를 시각화하기
for (X_train, y_train) in train_loader:
    print('X_train:', X_train.size(), 'type:', X_train.type())
    print('y_train:', y_train.size(), 'type:', y_train.type())
    break

pltsize=1
plt.figure(figsize=(10*pltsize, pltsize))

for i in range(10):
    plt.subplot(1,10,i+1)
    plt.axis('off')
    plt.imshow(X_train[i,:,:,:].numpy().reshape(28,28), cmap="gray_r")
    plt.title('Class: '+str(y_train[i].item()))
'''


## 6. 모델 설계. input_dim = 28*28 = 784, output_dim = 10
linear = nn.Linear(784, 10, bias=True).to(device)
# 여디서 연산을 수행할지 정하고, 모델의 매개변수를 지정한 장치의 메모리로 보낸다.

## 7. 크로스엔트로피 함수와 경사하강법(SGD)를 정해준다.
crossentropy = nn.CrossEntropyLoss().to(device)
sgd = torch.optim.SGD(linear.parameters(), lr=0.1)
# CrossEntropyLoss 함수는 내부적으로 소프트맥스 함수를 포함하고 있다.


## 8. MNIST
for epoch in range(training_epochs):
    avg_loss = 0
    batch_sum = len(data_loader)

    # 입력 이미지를 [BATCH_SIZE x 784]의 784차원 벡터로 reshape
    #label = one-hot encoding
    for X, Y in data_loader:
        X = X.view(-1, 28, 28).to(device) # label: one-hot encoding이 된 상태가 아니라 range(0, 10)의 정수
        Y = Y.to(device)

        sgd.zero_grad()
        expect = linear(X)
        loss = crossentropy(expect, Y)
        loss.backward()
        sgd.step()

        avg_loss += loss/batch_sum

    print('Epoch:', '%04d' %(epoch+1), '| Loss =', '{:.9f}'.format(avg_loss))

print('Learning Finished!')
print('-------------------------------------------------------------------')


## 9. 테스트 데이터를 사용하여 모델 테스트하기
with torch.no_grad():  # torch.no_grad()를 하면 gradient 계산을 수행하지 않는다.
    X_test = mnist_test.test_data.view(-1, 28*28).float().to(device)
    Y_test = mnist_test.test_labels.to(device)

    predict = linear(X_test)
    correct_predict = torch.argmax(predict, 1) == Y_test
    accuracy = correct_predict.float().mean()
    print('Accuracy:', accuracy.item())

    # MNIST 테스트 데이터에서 랜덤으로 하나를 뽑아서 예측을 해본다.
    r = random.randint(0, len(mnist_test) - 1)
    X_single_data = mnist_test.test_data[r:r+1].voew(-1, 28*28).float().to(device)
    Y_single_data = mnist_test.test_labels[r:r+1].to(device)

    print('Label: ', Y_single_data.item())
    single_predict = linear(X_single_data)
    print('Prediction: ', torch.argmax(single_predict, 1).item())

    plt.imshow(mnist_test.test_data[r:r+1].view(28, 28), cmap='Greys', interpolation='nearest')
    plt.show()