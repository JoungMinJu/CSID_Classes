#1 가중치 매개변수의 최적값을 탐색하는 최적화 방법
#2. 가중치 매개변수 초깃값
#3. 하이퍼파라미터 설정 방법
#4. 가중치 감소와 드롭아웃
#5. 배치정규화

#[매개변수 갱신]

#SGD는 기울어진 방향으로 일정 거리만 가겠다는 것
class SGD:
    def __init__(self, lr=0.01):
        self.lr=lr
    def update(self, params, grads):
        for key in params.keys():
            params[key]-=self.lr*grads[key]

#신경망 학습에서 매개변수 갱신은 optimizer가 책임지고, 우리는 optimizer에 매개변수와 기울기 정보만 넘겨주면 된다.
#최적화를 담당하는 클래스를 분리해 구현하면 기능을 모듈화하기 좋다.

#SGD는 비등방성 함수(방향에 따라 성질, 즉 책 예시에서는 기울기가 달라지는 함수)에서는 탐색 경로가 비효율적임
#이럴때는 SGD 같이 무작정 기울어진 방향으로 진행하는 단순한 방식보다 더 효율적인 방법이 요구됨.

import numpy as np
#[모멘텀]
class Momentum:
    def __init__(self,lr=0.01, momentum=0.9):
        self.lr=lr
        self.momentum=momentum
        self.v=None #속도
    def update(self, params, grads):
        if self.v is None:
            self.v={}
            for key, val in params.items():
                self.v[key]=np.zeros_like(val) 
                #매개변수와 같은 구조의 데이터를 딕셔너리 변수로 저장한다. 
        for key in params.keys():
            self.v[key]=self.momentum*self.v[key]-self.lr*grads[key]
            params[key]+=self.v[key]
        #모멘텀 함수를 파이썬으로 구현한 부분.


#[AdaGrad]
#신경망 학습에서는 학습률 값이 중요. 이게 너무 작으면 학습시간이 너무 길어지고 너무 크면 학습이 제대로 이뤄지지 않는다.
#학습률을 정하는 효과적 기술로 '학습률 감소'가 있다. 이는 학습을 진행하면서 학습률을 점차 줄여가는 방법임.
#AdaGrad는 각각의 매개변수에! 맞춤형 값을 만들어준다.

#식을 통해 매개변수의 원소 중 많이 움직인(크게 갱신된)원소는 학습률이 낮아짐을 파악할 수 있다.
class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr=lr
        self.h=None #식을 보면 기존 기울기 값을 제곱하여 계속 더해줌을 확인할 수 있다.
    def update(self, params, grads):
        if self.h is None:
            self.h={}
            for key, val in params.items():
                self.h[key]=np.zeros_like(val)
        for key in params.keys():
            self.h[key]+=grads[key]*grads[key]
            params[key]-=self.lr*grads[key]/(np.sqrt(self.h[key])+1e-7)
            #마지막에 작은값을 더해서 h=0인 경우를 처리한다.

#[Adam]
#모멘텀과 AdamGrad를 융합?
#또한 하이퍼파라미터의 편향 보정이 진행된다. 
#구체적인 설명은 없었슴

class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None
    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)
        self.iter += 1
        lr_t  = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)         
        for key in params.keys():
            #self.m[key] = self.beta1*self.m[key] + (1-self.beta1)*grads[key]
            #self.v[key] = self.beta2*self.v[key] + (1-self.beta2)*(grads[key]**2)
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])
            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)
            #unbias_m += (1 - self.beta1) * (grads[key] - self.m[key]) # correct bias
            #unbisa_b += (1 - self.beta2) * (grads[key]*grads[key] - self.v[key]) # correct bias
            #params[key] += self.lr * unbias_m / (np.sqrt(unbisa_b) + 1e-7)


#앞서 설명한 매개변수 갱신 기법 4가지를 비교
# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from common.optimizer import *
def f(x, y):
    return x**2 / 20.0 + y**2
def df(x, y):
    return x / 10.0, 2.0*y
init_pos = (-7.0, 2.0)
params = {}
params['x'], params['y'] = init_pos[0], init_pos[1]
grads = {}
grads['x'], grads['y'] = 0, 0

optimizers = OrderedDict()
optimizers["SGD"] = SGD(lr=0.95)
optimizers["Momentum"] = Momentum(lr=0.1)
optimizers["AdaGrad"] = AdaGrad(lr=1.5)
optimizers["Adam"] = Adam(lr=0.3)

idx = 1

for key in optimizers:
    optimizer = optimizers[key]
    x_history = []
    y_history = []
    params['x'], params['y'] = init_pos[0], init_pos[1]
    for i in range(30):
        x_history.append(params['x'])
        y_history.append(params['y'])
        grads['x'], grads['y'] = df(params['x'], params['y'])
        optimizer.update(params, grads)
    x = np.arange(-10, 10, 0.01)
    y = np.arange(-5, 5, 0.01)

    X, Y = np.meshgrid(x, y) 
    Z = f(X, Y)
    
    # 외곽선 단순화
    mask = Z > 7
    Z[mask] = 0
    
    # 그래프 그리기
    plt.subplot(2, 2, idx)
    idx += 1
    plt.plot(x_history, y_history, 'o-', color="red")
    plt.contour(X, Y, Z)
    plt.ylim(-10, 10)
    plt.xlim(-10, 10)
    plt.plot(0, 0, '+')
    #colorbar()
    #spring()
    plt.title(key)
    plt.xlabel("x")
    plt.ylabel("y")
    
plt.show()

#풀어야하는 문제의 종류에 따라 최적의 옵티마이저가 달라지고 하이퍼파라미터의 설정에 관해서도 결과가 바뀐다.

#[MNIST를 이용한 갱신방법 비교]

import os
import sys
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.util import smooth_curve
from common.multi_layer_net import MultiLayerNet
from common.optimizer import *


# 0. MNIST 데이터 읽기==========
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

train_size = x_train.shape[0]
batch_size = 128
max_iterations = 2000


# 1. 실험용 설정==========
optimizers = {}
optimizers['SGD'] = SGD()
optimizers['Momentum'] = Momentum()
optimizers['AdaGrad'] = AdaGrad()
optimizers['Adam'] = Adam()
#optimizers['RMSprop'] = RMSprop()

networks = {}
train_loss = {}
for key in optimizers.keys():
    networks[key] = MultiLayerNet(
        input_size=784, hidden_size_list=[100, 100, 100, 100],
        output_size=10)
    train_loss[key] = []    


# 2. 훈련 시작==========
for i in range(max_iterations):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    for key in optimizers.keys():
        grads = networks[key].gradient(x_batch, t_batch)
        optimizers[key].update(networks[key].params, grads)
    
        loss = networks[key].loss(x_batch, t_batch)
        train_loss[key].append(loss)
    
    if i % 100 == 0:
        print( "===========" + "iteration:" + str(i) + "===========")
        for key in optimizers.keys():
            loss = networks[key].loss(x_batch, t_batch)
            print(key + ":" + str(loss))


# 3. 그래프 그리기==========
markers = {"SGD": "o", "Momentum": "x", "AdaGrad": "s", "Adam": "D"}
x = np.arange(max_iterations)
for key in optimizers.keys():
    plt.plot(x, smooth_curve(train_loss[key]), marker=markers[key], markevery=100, label=key)
plt.xlabel("iterations")
plt.ylabel("loss")
plt.ylim(0, 1)
plt.legend()
plt.show()

#위는 각 층이 100개의 뉴런으로 구성된 5츠 신경망에서 ReLu를 활성화함수로 사용했음
#결과는 SGD가 가장 느리고 AdaGrad가 조금 더 ㅏ르다.
#근데 주의할 점은 하이펖라미터인 학습률과 신경망의 구조(층 깊이 등)에 따라 결과는 달라질 것이라는 거다
#일반적으로 SGD보다 다른 세개가 빠르고, 때로는 최종 정확도도 높다.



#[가중치의 초기값]

#가중치를 작게 만들어 오버피팅을 방지하고 싶음. 가중치를 작게 만들고 싶으면 초깃값도 최대한 작은 값에서 시작하는 것이 정공법
#근데 지금까지 가중치의 초깃값은 0.01*np.random.rand(10,100)처럼 정규분포에서 생성되는 값을 0.01배한 작은 값(표준펴나차가 0.01인 정규분포)를 사용했음
#그렇다고 가중치의 값을 0으로 설정하면 오차역전파법에서 모든 가중치의 값이 똑같이 갱신되어서 안된다.

#은닉층의 활성화함수의 출력데이터의 분포를 관찰하면 중요한 정보를 얻을 수 있다. 
#아래 예시는 활성화 함수로 시그모이드 함수를 사용하는 5층 신경망에 무작위로 생성한 입력 데이터를 흘리며 각 층의 활성화값 분포를 히스토그램으로 그려보겠다.


import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def ReLU(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)
    
input_data = np.random.randn(1000, 100)  # 1000개의 데이터
node_num = 100  # 각 은닉층의 노드(뉴런) 수
hidden_layer_size = 5  # 은닉층이 5개
activations = {}  # 이곳에 활성화 결과를 저장

x = input_data

for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i-1]

    # 초깃값을 다양하게 바꿔가며 실험해보자！
    w = np.random.randn(node_num, node_num) * 1
    # w = np.random.randn(node_num, node_num) * 0.01
    # w = np.random.randn(node_num, node_num) * np.sqrt(1.0 / node_num)
    # w = np.random.randn(node_num, node_num) * np.sqrt(2.0 / node_num)

    a = np.dot(x, w)

    # 활성화 함수도 바꿔가며 실험해보자！
    z = sigmoid(a)
    # z = ReLU(a)
    # z = tanh(a)

    activations[i] = z

# 히스토그램 그리기
for i, a in activations.items():
    plt.subplot(1, len(activations), i+1)
    plt.title(str(i+1) + "-layer")
    if i != 0: plt.yticks([], [])
    # plt.xlim(0.1, 1)
    # plt.ylim(0, 7000)
    plt.hist(a.flatten(), 30, range=(0,1))
plt.show()

#층이 5개가 있으며 각 층의 뉴런은 100개. 입력데이터로 1000개의 데이터를 정규분포로 무작위로 생성하여 이 5층 신경망에 흘린다.
#활성화 함수로는 시그모이드 함수를 이용하고, 각 층의 활성화 결과를 activations 변수에 저장한다.
#이 코드에서는 가중치의 분포에 주의해야함. 이번에는 표준편차가 1인 정규분포를 썼는데, 이 분포된 정도(표준편차)를 바꿔가며 활성화값들의 분포가 어떻게 변화하는지 관찰하는 것이 이 실험의 목적

#결과는 각 층의 활성화값들이 0과 1에 치우쳐 분포되는 결과를 보여준다.
#시그모이드 함수는 출력이 0 또는 1에 가까워지자 미분이 0에 다가간다. 
#그래서 데이터가 0과 1에 치우쳐 분포하게 되면 역전파의 기울기값이 점점 작아지다가 사라진다. 이게 기울기 소실!

#w=np.random.rand(node_num, node_num)*0.01로 바꾸면(표준편차를 0.01로 한 정규분포)
#0.5부근에 집중된다. 기울기소실 문제는 없지만 활성화 값들이 치우쳤다는 것은 표현력 관점에서는 큰 문제가 있는 것
#왜냐하면 다수의 뉴런이 거의 같은 값을 출력하고 있으니 뉴런을 여러개 둔 의미가 없다는 것.
#그래서 활성화값들이 치우치면 표현력을 제한한다는 관점에서 문제가 된다.

#Xavier 초깃값을 일반적인 딥러닝 프레임워크들이 표준적으로 이용ㅇ하고 있다. 
#이것을 사용하면 앞 층에 노드가 많을수록 대상 노드의 초깃값으로 설정하는 가중치가 좁게 퍼진다.

#코드는 node_num=100
#w=np.random.randn(node_num, node_num)/np.sqrt(node_num) 이라 고쳐주면 된다




#[ReLu를 사용할 때의 가중치 초기값]
#xavier은 활성화함수가 선형인것을 전제로 이끈 결과값.
#그래서 ReLu에 특화된 초기값을 He 초깃값이라한다.

#{std=0.01, Xaiver초깃값, He 초깃값 세가지를 모두 실험해보겠다}
# coding: utf-8
import os
import sys

sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.util import smooth_curve
from common.multi_layer_net import MultiLayerNet
from common.optimizer import SGD


# 0. MNIST 데이터 읽기==========
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

train_size = x_train.shape[0]
batch_size = 128
max_iterations = 2000


# 1. 실험용 설정==========
weight_init_types = {'std=0.01': 0.01, 'Xavier': 'sigmoid', 'He': 'relu'}
optimizer = SGD(lr=0.01)

networks = {}
train_loss = {}
for key, weight_type in weight_init_types.items():
    networks[key] = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100],
                                  output_size=10, weight_init_std=weight_type)
    train_loss[key] = []


# 2. 훈련 시작==========
for i in range(max_iterations):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    for key in weight_init_types.keys():
        grads = networks[key].gradient(x_batch, t_batch)
        optimizer.update(networks[key].params, grads)
    
        loss = networks[key].loss(x_batch, t_batch)
        train_loss[key].append(loss)
    
    if i % 100 == 0:
        print("===========" + "iteration:" + str(i) + "===========")
        for key in weight_init_types.keys():
            loss = networks[key].loss(x_batch, t_batch)
            print(key + ":" + str(loss))


# 3. 그래프 그리기==========
markers = {'std=0.01': 'o', 'Xavier': 's', 'He': 'D'}
x = np.arange(max_iterations)
for key in weight_init_types.keys():
    plt.plot(x, smooth_curve(train_loss[key]), marker=markers[key], markevery=100, label=key)
plt.xlabel("iterations")
plt.ylabel("loss")
plt.ylim(0, 2.5)
plt.legend()
plt.show()

#결과를 보면
#층별 뉴런수가 100ㅐ인 5층 신경망에서 활성화 함수로 ReLu를 사용했음
#std=0.01일때는 전혀 학습이 이뤄지지 않았다. 앞서 활성화 값의 분포에서본 것 처럼 순전파 때 너무 작은 값이 흐르기 때문. 이러면 역전파때의 기울기도 작아져 가중치가 거의 갱신되지 않는다.



#[배치 정규화]

#각 층이 활성화를 적당히 퍼뜨리도록  '강제'하는 것
#데이터의 분포를 정규화하는 '배치 정규화 계층'을 신경망에 삽입한다.
#배치정규화는 학습시 미니배치를 단위로 정규화한다. 구체적으로는 데이터 분포가 평균0 분산 1이 되도록 정규화한다.
#또 배치정규화계층마다 이 정규화된 데이터에 고유한 확대와 이동 변환을 수행한다. 

#배치 정규화 MNIST사용ㅎ!
# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.multi_layer_net_extend import MultiLayerNetExtend
from common.optimizer import SGD, Adam

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# 학습 데이터를 줄임
x_train = x_train[:1000]
t_train = t_train[:1000]

max_epochs = 20
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.01


def __train(weight_init_std):
    bn_network = MultiLayerNetExtend(input_size=784, hidden_size_list=[100, 100, 100, 100, 100], output_size=10, 
                                    weight_init_std=weight_init_std, use_batchnorm=True)
    network = MultiLayerNetExtend(input_size=784, hidden_size_list=[100, 100, 100, 100, 100], output_size=10,
                                weight_init_std=weight_init_std)
    optimizer = SGD(lr=learning_rate)
    
    train_acc_list = []
    bn_train_acc_list = []
    
    iter_per_epoch = max(train_size / batch_size, 1)
    epoch_cnt = 0
    
    for i in range(1000000000):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
    
        for _network in (bn_network, network):
            grads = _network.gradient(x_batch, t_batch)
            optimizer.update(_network.params, grads)
    
        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            bn_train_acc = bn_network.accuracy(x_train, t_train)
            train_acc_list.append(train_acc)
            bn_train_acc_list.append(bn_train_acc)
    
            print("epoch:" + str(epoch_cnt) + " | " + str(train_acc) + " - " + str(bn_train_acc))
    
            epoch_cnt += 1
            if epoch_cnt >= max_epochs:
                break
                
    return train_acc_list, bn_train_acc_list


# 그래프 그리기==========
weight_scale_list = np.logspace(0, -4, num=16)
x = np.arange(max_epochs)

for i, w in enumerate(weight_scale_list):
    print( "============== " + str(i+1) + "/16" + " ==============")
    train_acc_list, bn_train_acc_list = __train(w)
    
    plt.subplot(4,4,i+1)
    plt.title("W:" + str(w))
    if i == 15:
        plt.plot(x, bn_train_acc_list, label='Batch Normalization', markevery=2)
        plt.plot(x, train_acc_list, linestyle = "--", label='Normal(without BatchNorm)', markevery=2)
    else:
        plt.plot(x, bn_train_acc_list, markevery=2)
        plt.plot(x, train_acc_list, linestyle="--", markevery=2)

    plt.ylim(0, 1.0)
    if i % 4:
        plt.yticks([])
    else:
        plt.ylabel("accuracy")
    if i < 12:
        plt.xticks([])
    else:
        plt.xlabel("epochs")
    plt.legend(loc='lower right')
    
plt.show()
#이 결과를 통해 배치 정규화가 학습을 빨리 진전시키고 있음을 파악할 수 있따. 
#이 상태에서 가중치 초깃값의 표준편차를 다양하게 바꿔가며 학습경과를 관찰 할 수도 있음.
#거의 모든 경우에서 배치 정규화를 사용할 때 학습 진도가 빠른 것으로 나타난다.


#[오버피팅]

#주로 매개변수가 많고 표현력이 높은 모델, 훈련데이터가 적은 상황에서 발생한다.

#일부러 오버피팅을 발생시킴.
# coding: utf-8
import os
import sys

sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.multi_layer_net import MultiLayerNet
from common.optimizer import SGD

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# 오버피팅을 재현하기 위해 학습 데이터 수를 줄임
x_train = x_train[:300]
t_train = t_train[:300]

# weight decay（가중치 감쇠） 설정 =======================
#weight_decay_lambda = 0 # weight decay를 사용하지 않을 경우
weight_decay_lambda = 0.1
# ====================================================

network = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100], output_size=10,
                        weight_decay_lambda=weight_decay_lambda)
optimizer = SGD(lr=0.01) # 학습률이 0.01인 SGD로 매개변수 갱신

max_epochs = 201
train_size = x_train.shape[0]
batch_size = 100

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)
epoch_cnt = 0

for i in range(1000000000):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grads = network.gradient(x_batch, t_batch)
    optimizer.update(network.params, grads)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        print("epoch:" + str(epoch_cnt) + ", train acc:" + str(train_acc) + ", test acc:" + str(test_acc))

        epoch_cnt += 1
        if epoch_cnt >= max_epochs:
            break
    #train_acc_list 와 test_acc_list 에는 에폭단위의 정확도를 저장한다.


# 그래프 그리기==========
markers = {'train': 'o', 'test': 's'}
x = np.arange(max_epochs)
plt.plot(x, train_acc_list, marker='o', label='train', markevery=10)
plt.plot(x, test_acc_list, marker='s', label='test', markevery=10)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()



#신경망 모델이 복잡해지면 가중치의 감소만으로는 대응하기 어려워진다. 그럴때 드롭아웃을 사용한다.
#드롭아웃은 뉴런을 임의로 삭제하면서 학습하는 방법! 훈련때 은닉층의 뉴런을 무작위로 골라 삭제한다. 
#훈련때는 데이터를 흘릴때마다 삭제할 뉴런을 무작위로 선택하고
#시험 때는 모든 뉴런에 신호를 전달한다. 근데 시험때는 각 뉴런의 출력에 훈련때 삭제 안 한 비율을 곱해서 출력한다.

class Dropout:
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio=dropout_ratio
        self.mask=None
    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask=np.random.rand(*x.shape)>self.dropout_ratio
            return x*self.mask
        else:
            return x*(1.0-self.dropout_ratio)
    def backward(self, dout):
        return dout*self.mask

#훈련시마다 self.mask에 삭제할 뉴런을  False로 표시한다는것
#self.mask는 x와 형상이 같은 배열을 무작위로 생성하고 그 값이 dropout_ratio보다 큰 원소만 True로 설정한다. 
#역전파때의 동작은 ReLu와 같다. 

#드롭아웃의 효과를 MNIST를 이용하여 확인 
# coding: utf-8
import os
import sys
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.multi_layer_net_extend import MultiLayerNetExtend
from common.trainer import Trainer

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# 오버피팅을 재현하기 위해 학습 데이터 수를 줄임
x_train = x_train[:300]
t_train = t_train[:300]

# 드롭아웃 사용 유무와 비울 설정 ========================
use_dropout = True  # 드롭아웃을 쓰지 않을 때는 False
dropout_ratio = 0.2
# ====================================================

network = MultiLayerNetExtend(input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100],
                              output_size=10, use_dropout=use_dropout, dropout_ration=dropout_ratio)
trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=301, mini_batch_size=100,
                  optimizer='sgd', optimizer_param={'lr': 0.01}, verbose=True)
trainer.train()

train_acc_list, test_acc_list = trainer.train_acc_list, trainer.test_acc_list

# 그래프 그리기==========
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, marker='o', label='train', markevery=10)
plt.plot(x, test_acc_list, marker='s', label='test', markevery=10)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()


#하이퍼파라미터의 최적화
#1.하이퍼파라미ㅓ 값의 범위를 설정
#2. 설정된 범위에서 하이퍼파라미터 값을 무작위로 추출
#3. 2에서 샘플링한 하이퍼파라미터값을 사용하여 학습하고, 검증데이터로 정확도를 평가(단 에폭은 작게 설정)
#4. 2와 3을 특정횟수 반복하여 그 정확도의 결과를 보고 하이퍼파라미터의 범위를 좁힘.

