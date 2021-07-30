#신경망의 특징은 데이터를 보고 학습할 수 있다는 점. 
#가중치 매개변수의 값을 데이터를 보고 자동으로 결정한다는 것.

#이미지에서 특징을 추출하고 그 특징의 패턴을 기계학습 기술로 학습하는 방법
#특싱은 입력데이터에서 본질적 데이터를 정확하게 추출할 수 있도록 설계된 변환기를 지칭
#이미지의 특징ㅇ은 보통 벡터로 기술하고 컴퓨터 비전 분야에서는  SIFT, SURF, HOG 등의 특징을 많이 사용한다

#이런 특징을 사용하여 이미지 데이터를 벡터로 변환하고 변환된 벡터를 가지고 지도학습 방식의 대표 분류 기법인 SVM,KNN등으로 학습

#근데 이미지를 벡터로 변환할 때 사용하는 특징은 여전히 사람이 설계하는 것.

#딥러닝은 처음부터 끝까지 사람의 개입이 없이 이루어진다
#손실함수라는 하나의 지표를 기준으로 최적의 매개변수 값을 탐색하낟.
#신경망 학습에서 사용하는 지표는 손실함수. 일반적ㅇ로 오차제곱합과 교차 엔트로피 오차를 사용한다.


#오차 제곱합
#각 원소의 출력(추정값)과 정답 레이블(참값)의 차를 제곱한 후 그 총합을 구한다.
import numpy as np
from numpy.testing._private.utils import integer_repr

def sum_squares_error(y,t):
    return 0.5*np.sum((y-t)**2)

#교차 엔트로피 오차
def cross_entropy_error(y,t):
    delta=1e-7
    return -np.sum(t*np.log(y+delta))
#y=0이면 오류나서 아주 작은 수인 delta를 더해주는 것으로 구현했다.


#총 데이터 중 일부만 골라서 학습하는 것을 미니배치 학습이라한다.


import sys,os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
(x_train, t_train), (x_test, t_test)=load_mnist(normalize=True, one_hot_label=True)

print(x_train.shape)
#(60000,784)
print(t_train.shape)
#(60000,10)

#이 훈련데이터에서 10장만 빼오겠다 무작위로
train_size=x_train.shape[0]#총 60000개의 데이터
batch_size=10
batch_mask=np.random.choice(train_size, batch_size) #미니배치로 뽑아낼 인스로 지정. 여기서 0~5999의 수 중 무작위로 10개를 골라내기 때문
x_batch=x_train[batch_mask]
t_batch=t_train[batch_mask]


#--

#배치데이터를 지원하는 교차 엔트로피 오차
def cross_entropy_error(y,t):
    if y.ndim==1: #y가 1차원이라면, 즉 데이터 하나 당 교차 엔트로피를 구하는 경우는 reshape함수로 형상을 바꿔준다. 
        t=t.reshape(1,t.size)
        y=y.reshape(1,y.size)
    batch_size=y.shape[0]  
    return -np.sum(t*np.log(y+ 1e-7 ))/batch_size #배치 크기로 나눠서 정규화하고 이미지 한장당 평균의 교차 엔트로피 오차를 계산함.
#y는 신경망의 출력 t는 정답레이블


#아래는 정답 레이블이 원핫인코딩이 아니라 숫자레이블로 주어졌을 때의 교차 엔트로피 오차 구하기
def cross_entropy_error(y,t):
    if y.ndim==1:
        t=t.reshape(1,t.size)
        y=y.reshape(1, y.size)
    batch_size=y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size),t]+1e-7))/batch_size
#이 함수는 숫자레이블 t가 0인건 교차엔트로피오차도 0이므로 그 계산은 무시해도 좋다는 것이 핵심임
#긍까 정답에 해당하는 신경망의 출력만으로 교차 엔트로피 오차를 계산할 수 있는 것
#np.log(y[np.arange(batch_size_,t])는?
#np.arange(batch_size)는 0부터 batch_size-1까지 배열을 새성
#y[np.arange(batch_size),t]는 각 데이터의 정답 레이블에 해당하는 신경망의 출력을 추출. 


#중심차분을 이용하여 수치미분 구현((x+h)와 (x-h)일때의 함수 f의 차분을 계산하는 것)

def numerical_diff(f,x):
    h=1e-4
    return (f(x+h)-f(x-h))/(2*h)


#수치미분의 예
def function_1(x):
    return 0.01*x**2+0.1*x

import numpy as np
import matplotlib.pyplot as plt

x=np.arange(0.0,20.0,0.1)
y=function_1(x)
plt.xlabel("x")
plt.ylabel('f(x)')
plt.plot(x,y)
plt.show()


#x=5일떄와 10일때의 함수의 미분 ㅜ하기
print(numerical_diff(function_1, 5))
print(numerical_diff(function_1,10))
#이게 바로 아주 작은 차분으로 미분하는 '수치미분'
#dy/dx 이렇게 구하는것이 오차를 포함하지 않는 진정한 미분을 구하는 해석적 미분. 해석적 해



#편미분

#새로운 식 정의= 인수들의 제곱합을 계산하는 식
def function_2(x):
    return x[0]**2+x[1]**2
    #3차원 그래프가 나온다

#여기서는 변수가 2개이다. 그래서 어느 변수에 대한 미분이냐를 구별해야한다.
#이와 같이 변수가 여럿이 ㄴ함수에 대한 미분을 편미분이라한다. 

#이 문제들을 변수가 하나인 함수를 정의하고 그 함수를 미분하는 형태로 구현하여 풀 수 있다.
def function_tmp1(x0):
    return x0*x0+4.0**2.0
numerical_diff(function_tmp1, 3.0)
#x1=4로 고정된 새로운 함수 정의하고 변수가 x0 하나인 함수에 대해 수치미분함수를 적용했음. 해석적 미분의 결과와 거의 같다.

#이처럼 편미분은 변수가 하나 뿐인 미분과 마찬가지로 특정 장소의 기울기를 구한다.
#다만 목표 변수 하나에만 초점을 마주고 다른 변수는 값을 고정한다. 



#기울기


#편미분을 동시에 계산하고 싶으면??
#모든 변수의 편미분을 벡터로 정리한 것을 기울기라고 한다.
def numerical_gradient(f,x):
    h=1e-4
    grad=np.zeros_like(x) #x와 형상이 같은 배열을 생성

    for idx in range(x.size):
        tmp_val=x[idx]
        #f(x+h) 계산
        x[idx]=tmp_val+h
        fxh1=f(x)
        #f(x-h) 계산
        x[idx]=tmp_val-h
        fxh2=f(x)

        grad[idx]=(fxh1-fxh2)/(2*h)
        x[idx]=tmp_val
    return grad

print(numerical_gradient(function_2, np.array([3.0,4.0])))
print(numerical_gradient(function_2, np.array([0.0,2.0])))
#이런식으로 각 점에서의 기울기를 게산할 수 있다. 
#기울기 그림은 방향을 가진 벡터로 그려진다. 기울기는 가장 낮은 장소(최솟값)을 가리키는 것 처럼 보인다. 그리고 가장 낮은 곳에서 멀어질 수록 화살표의 크기가 커지고 있다.

#기울기가 가리키는 쪽은 각 장소에서 함수의 출력값을 가장 크게 줄이는 밭ㅇ향잉다.






#기울기를 잘 이용해 함수의 최솟값을 찾으려는 것이 경사법.
#근데 기울기가 가리키는 곳에 정말 함수의 최솟값이 있는지는 보장할 수 없다.
#거기가 최솟값을 가리키는 것은 아니지만 그 방향으로 가야 함수의 값을 줄일 수 있다. 그래서 최솟값이 되는 장소를 찾는 문제에서는 기울기 정보를 단서로 나아갈 방향을 정해야한다.

#경사법은 현 위치에서 기울어진 방향으로 일정 거리만큼 이동한다. 이러게 함수의 값을 점차 줄이는 것이 경사법이다. 
#최솟값을 찾는 것이 경사 하강법, 최댓값을 찾는 것이 경사 상승법이다. 근데 어차피 부호 바꾸면 본질은 동일.

#경사하강법 구현
def gradient_descent(f, init_x, lr=0.01,step_num=100):
    x=init_x
    for i in range(step_num): #함수의 기울기를 기우기에 학습률을 곱한 값으로 갱신하는 처리를  step_num번 반복한다.
        grad=numerical_gradient(f,x)#함수 기울기(편미분값) 구함
        x-=lr*grad
    return x



#신경망에서의 기울기

#여기서의 기울기는 가중치 매개변수에 대한 손실함수의 기울기
#간단한 신경망을 예로 들어 실제로 기울기를 구하ㅡㄴ 코드
import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient

class simpleNet:
    def __init__(self):
        self.W=np.random.rand(2,3)#정규분포로 초기화 0~1사이의 난수를 2행 3열로 만들었음.
    def predict(self, x):
        return np.dot(x, self.W)
    def loss(self,x,t):
        z=self.predict(x)
        y=softmax(z) #0~1사이의 숫자 반환한다.
        loss=cross_entropy_error(y,t)
        return loss
net=simpleNet()
print(net.W)#가중치 매개변수
x=np.array([0.6,0.9])
p=net.predict(x)
print(p)
np.argmax(p) #최댓값의 인덱스
t=np.array([0,0,1])#정답레이블
net.loss(x,t)

#이어서 기울기를 구해보기 numeical_gradient(f,x) 써서 구하면 된다.


def f(W):
    return net.loss(x,t)
dW=numerical_gradient(f,net.W) #기울기를 구하는
print(dW)#2*3의 2차원 배열!
#이 결과를 통해 솔실함수를 줄인다는 관점에서 어떤 변수?를 양의 방향으로 갱신하고 음의 방향으로 갱신해야하는지 파악할 수 있다.

#람다도 사용 가능
f=lambda w:  net.loss(x,t)
dW=numerical_gradient(f,net.W)


#학습 알고리즘 구혀하기

#절차
#전체: 신경망에는 가중치와 편향이 있고 이 가중치와 편향을 훈련데이터에 적응하도록 조저아는 과정을 학습이라 한다. 신경망학습은 4단계로
#1단계 -미니배치 : 훈련 데이터중일부를 무작위로 가져온다.이렇게 선별한 데이터가 미니배치이며 이런 미니배치의 손실함수 값을 줄이는 것이 목표
#2단계 - 기울기 산출: 미니배치의 손실함수 값을 줄이기 위해 각 가중치 매개변수의 기울기를 구한다. 기울기는 손실함수의 값을 가장 작게하는 방향을 제시한다.
#3단계 - 매개변수 갱신: 가중치 매개변수를 기울기 방향으로 아주 조금 갱신한다.
#4단계 - 반복: 1~3단계를 반복

#이것을 확률적 경사 하강법이라 부른다.  SGD라고 함

import sys, os
sys.path.append(os.pardir)
from common.functions import *
from common.gradient import numerical_gradient
class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params={}
        self.params["W1"]=weight_init_std*np.random.rand(input_size, hidden_size)
        self.params['b1']=np.zeros(hidden_size)
        self.params['W2']=weight_init_std*np.random.rand(hidden_size,output_size)
        self.params['b2']=np.zeros(output_size)

    def predict(self,x):
        W1,W2=self.params['W1'],self.params['W2']
        b1,b2=self.params['b1'],self.params['b2']
        a1=np.dot(x,W1)+b1
        z1=sigmoid(a1)
        a2=np.dot(z1,W2)+b2
        y=softmax(a2)
        return y
    #x:입력데이터 t:정답레이블
    def loss(self, x,t):
        y=self.predict(x)
        return cross_entropy_error(y,t)
    def accuracy(self, x,t):
        y=self.predict(x)
        y=np.argmax(y,axis=1)
        t=np.argmax(t, axis=1)
        accuracy=np.sum(y==t)/float(x.shape[0])
        return accuracy
    def numerical_gradient(self, x,t):
        loss_W=lambda W: self.loss(x,t) #loss로 반환되는 값의 인수 x는 이미지 데이터 t는 정답 레이블
        grads={}
        grads["W1"]=numerical_gradient(loss_W, self.params['W1'])#가중치 매개변수의 기울기를 구한다.
        grads["b1"]=numerical_gradient(loss_W, self.params['b1'])
        grads["W2"]=numerical_gradient(loss_W, self.params['W2'])
        grads["b2"]=numerical_gradient(loss_W, self.params['b2'])

        return grads

#params는 신경망의 매개변수를 보관하는 딕셔너리 변수
#prams['W1']은 첫번째 층의 가중치, b1은 편향

#gads는 기울기를 보관하는 딕셔너리 변수


net=TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
net.params['W1'].shape#(784,100)
net.params['b1'].shape#(100,)
net.params['W2'].shape#(100,10)
net.params['b2'].shape#(10,)



#미니배치 학습 구현
from ch04.two_layer_net import TwoLayerNet

(x_train,t_train),(x_test,t_test)=load_mnist(normalize=True, one_hot_label=True)
train_loss_list=[]
train_acc_list=[]
test_acc_list=[]

#하이퍼파라미터
iters_num=1000
tain_size=x_train.shape[0]
batch_size=100
learning_rate=0.1


#1에폭당 반복 수
iter_per_epoch=max(train_size/batch_size,1)


network=TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
for i in range(iters_num):
    #미니배치획득
    batch_mask=np.random.choice(train_size, batch_size)
    x_batch=x_train[batch_mask]
    t_batch=t_train[batch_mask]

    #기울기 계산
    grad=network.numerical_gradient(x_batch, t_batch)
    #성능 개선판
    
    #매개변수 갱신
    for key in ("W1",'b1','W2','b2'):
        network.params[key]-=learning_rate*grad[key]

    #학습 경과 기록
    loss=network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    #1에폭당 정확도 계산
    if i%iter_per_epoch==0:
        train_acc=network.accuracy(x_train, t_train)
        test_acc=network.accuracy(x_test,t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print('train acc, test acc |'+str(train_acc)+","+str(test_acc))

#매번 60,000개의 훈련데이터 중 100개의 데이터를ㄹ 추려내고 그 100개의 미니배치를 대상으로 확률적 경사하강법을 수행해 매개변수를 갱신한다
#경사법에 의한 갱신 횟수를 10,000번으로 설정하고 갱신할 때마다 훈련데이터에 대한 손실함수를 계산하고 그 값을 배열에 추가한다.
