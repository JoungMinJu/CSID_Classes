class MulLayer:
    def __init__(self):
        self.x=None
        self.y=None
    def forward(self, x,y):
        self.x=x
        self.y=y
        out=x*y
        return out
    def backward(self,dout):
        dx=dout*self.y #x와 y를 바꾼다.
        dy=dout*self.x 
        return dx,dy

#책 예제인 사과 구입 예제 구현해보기
apple=100
apple_num=2 #2개
tax=1.1 #과세
#계층들
mul_apple_layer=MulLayer()
mul_tax_layer=MulLayer()
#순전파
apple_price=mul_apple_layer.forward(apple, apple_num)
price=mul_tax_layer.forward(apple_price, tax)
print(price)
#역전파
dprice=1
dapple_price, dtax=mul_tax_layer.backward(dprice)
dapple, dapple_num=mul_apple_layer.backward(dapple_price)
print(dapple, dapple_num, dtax)
#backward가 받는 인수는 순전파의 출력에 대한 미분

#덧셈계층 구혀
class AddLayer:
    def __init__(self):
        pass
    def forward(self, x,y):
        out=x+y
        return out
    def backward(self,dout):
        dx=dout*1
        dy=dout*1
        return dx,dy
#마찬가지로 책의 예제 실행
apple=100
apple_num=2 #2개
orange=150
orange_num=3
tax=1.1 #과세
#계층들
mul_apple_layer=MulLayer()
mul_orange_layer=MulLayer()
add_apple_orange_layer=AddLayer()
mul_tax_layer=MulLayer()
#순전파
apple_price=mul_apple_layer.forward(apple, apple_num)
orange_price=mul_orange_layer.forward(orange, orange_num)
all_price=add_apple_orange_layer.forward(apple_price,orange_price)
price=mul_tax_layer.forward(all_price, tax)
#역전파
dprice=1
dall_price, dtax=mul_tax_layer.backward(dprice)
dapple_price, dorange_price=add_apple_orange_layer.backward(dall_price)
dorange, dorange_num=mul_orange_layer.backward(dorange_price)
dapple, dapple_num=mul_apple_layer.backward(dapple_price)
print(price)
print(dapple, dapple_num, dorange, dorange_num, dtax)
#backward가 받는 인수는 순전파의 출력에 대한 미분

class Relu:
    def __init__(self):
        self.mask = None
        #mask는 True, False를 인수로 가진 넘파이 배열 
        #순전파의 입력이 0이하면 True, 그 외는 False

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx

import numpy as np
class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1/(1+np.exp(-x))
        self.out = out
        return out
    #순전파의 출력을 out에 보관했다가 역전파 계산 때 그 값을 사용한다. 

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
    #/노드는 식에 따르면 역전파때는 상류에서 흘러들어온 값에 -y^2을 곱해서 하류로 전달한다. 
    #+노드는 상류의 값을 여과없이 하류로 보낸다
    #exp 노드는 상류의 값에 순전파 때의 출력을 곱해 하류로 전달
    #x 노드는 순전파때의 값을 서로 바꿔 곱한다.
        return dx
#이를 요약하면 시그모이드 계층의 역전파는 순전파의 출력만으로 계산할 수 있다.

class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.original_x_shape = None
        # 가중치와 편향 매개변수의 미분
        self.dW = None
        self.db = None

    def forward(self, x):
        # 텐서 대응(4차원 데이터)
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x
        out = np.dot(self.x, self.W) + self.b
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        dx = dx.reshape(*self.original_x_shape)  # 입력 데이터 모양 변경(텐서 대응)
        return dx

#Softmax-loss 계층은 역전파를 단순하게 도출한다
#소프트맥스 층에서의 출력이 y1,y2..이고 정답레이블이 t1,t2....이면 y1-t1, y2-t2...를 역전파의 결과로서 내는 것
#이는 교차엔트로피 오차라는 함수가 그렇게 설계되었기 때문이다. 
#회귀의 출력층에서 항등함수의 손실함수로 오차제곱합을 이용하면 역전파의 결과가 y1-t1과 같이 깔끔하게 떨어진다.

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None # 손실함수
        self.y = None    # softmax의 출력
        self.t = None    # 정답 레이블(원-핫 인코딩 형태)
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss
    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size: # 정답 레이블이 원-핫 인코딩 형태일 때
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        return dx

def softmax(a):
    c=np.max(a)
    exp_a=np.exp(a-c)
    sum_exp_a=np.sum(exp_a)
    y=exp_a/sum_exp_a
    return y
def cross_entropy_error(y,t):
    if y.ndim==1:
        t=t.reshape(1,t.size)
        y=y.reshape(1,y.size)
    batch_size=y.shape[0]
    return -np.sum(t*np.log(y+1e-7))/batch_size

#지금까지 구현한 계층을 조합해서 신경망을 구축

#[신경망 학습의 전체 그림]
#1. 신경망에는 적응가능한 가중치와 편향이 있고 이걸 훈련 데이터에 적으아도록 조정하는 과정이 학습
#학습1-미니배치 :: 훈련 데이터 중 일부를 무작위로 가져온다. 이렇게 선별한 데이터를 미니배치라하면 이 미니배치의 손실 함수 값을 줄이는 것이 목표
#학습 2 -기울기 산출 :: 미니배치의 손실 함수 값을 줄이기 위해 ㄱㄱ 가중치 매개변수의 기울기를 구한다. 기울기는 손실함수 값을 가장 작게하는 방향을 제시
#학습 3- 매개변수 갱신 : 가중치 매개변수를 기울기 방향으로 아주 조금 갱신
#학습 4- 1~3을 반복한다.

#여기서 '오차역전파법'이 등장하는 단계는 기울기 산출이다.
#앞장에서는 기울기를 구학 ㅣ위해 수치미분을 사용했는데 이는 시간이 오래걸리니까 오차역전파법을 사용하면 느린 수치 미분과 달리 기울기를 효율적이고 빠르게 구할 수 있음

# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from common.layers import *
from common.gradient import numerical_gradient
from collections import OrderedDict


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
        # 가중치 초기화
        #입력사이즈, 은닉층입력사이즈, 출력층 입력 사이즈, 가중치 초기화시 정규분포 스케일
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size) 
        self.params['b2'] = np.zeros(output_size)

        # 계층 생성
        self.layers = OrderedDict() 
        #순서가 있는 딕셔너리! 신경망의 계층을 순서까지 기억하겠다는 것.
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.lastLayer = SoftmaxWithLoss() #마지막레이어는 소프트맥스손실계층
        
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x) #순서대로 호출하여 순전파 기법을 사용해 예측 시작
        return x
        
    # x : 입력 데이터, t : 정답 레이블
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
        
    # x : 입력 데이터, t : 정답 레이블
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t) #앞장에서 배운 수치미분 방식으로 가중치 매개변수의 기울기 구함
        
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        
        return grads
        
    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 결과 저장
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads


#+ 여태 배운 기울기 구하는 두가지 방법
# 1. 수치미분 사용 2. 해석적으로 수식을 풀어구하기(=해석적방법)(=오차역전파법을 사용하면 매개변수가 많아도 효율적 계산 가능)

#오차역전파법으로 구한 기울기 검증하기
#수치미분은 오차역전파법을 정확히 구현했는지 확인하기 위해 필요하다.

# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

# 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

x_batch = x_train[:3]
t_batch = t_train[:3]

grad_numerical = network.numerical_gradient(x_batch, t_batch)
grad_backprop = network.gradient(x_batch, t_batch)
#훈련데이터 일부를 수치로 구한 기울기와 오차역전파법으로 구한 기울기의 오차를 확인한다.

# 각 가중치의 차이의 절댓값을 구한 후, 그 절댓값들의 평균을 낸다.
for key in grad_numerical.keys():
    diff = np.average( np.abs(grad_backprop[key] - grad_numerical[key]) )
    print(key + ":" + str(diff))

#[오차역전파법을 사용한 학습 구현하기]
# -- 기울기를 오차역전파법으로 구한다는 것만 차이가 있음

# coding: utf-8
import sys, os
sys.path.append(os.pardir)

import numpy as np
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

# 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    # 기울기 계산
    #grad = network.numerical_gradient(x_batch, t_batch) # 수치 미분 방식
    grad = network.gradient(x_batch, t_batch) # 오차역전파법 방식(훨씬 빠르다)
    
    # 갱신
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(train_acc, test_acc)
