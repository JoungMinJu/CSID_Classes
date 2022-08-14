#머신러닝 지도학습

#K-NN

install.packages('caret', dependencies = TRUE)
#R이 오픈소스임. caret 패키지는 아래의 패키지들을 이용하기 떄문에 dependencies 설정

library(caret)
#trainControl()= train과정의 파라미터 설정
#(method = 'repeatcv'(크로스밸리데이션 반복), number = 10, repeat=5)

#expand.grid() 모든 벡터 혹은 인자(factor) 조합인 데이터 프레임 생성
#train() 데이터 학습을 통한 모델 생성

#kappa 통계량 = (관측된 정확도 - 기대 정확도) / (1-기대정확도) 
#기대정확도 예시 : 동전 앞면 1/2 뒷면 1/2
#kappa 통계량은 -1<<1


#와인 데이터를 이용한 KNN
# 다 숫자형 변수

rawdata<-read.csv(file="C:/Users/pc/Desktop/패스트캠퍼스_데이터분석/part-6-강의자료-장철원강사님/Part 6_강의자료_장철원강사님/Ch02. k-Nearest Neighbor/Data/wine.csv", header=TRUE)
rawdata$Class<-as.factor(rawdata$Class)
str(rawdata)

analdata <- rawdata
set.seed(2020)
#train/test 분할
datatotal<-sort(sample(nrow(analdata), nrow(analdata)*0.7))
train<-rawdata[datatotal,]
test<-rawdata[-datatotal,]

train_x<-train[,1:13]
train_y<-train[,14]

test_x<-test[,1:13]
test_y<-test[,14]


ctrl<-trainControl(method='repeatedcv', number= 10, repeats=5)
customGrid<-expand.grid(k=1:10)
knnFit<-train(Class~.,#타켓변수
              data=train,
              method='knn',
              trControl=ctrl,
              preProcess =c('center','scale'),#표준화
              tuneGrid=customGrid,#k는 1부터 10까지
              metric='Accuracy')#평가방법은 정확도

knnFit
#결과
#전처리도 되어있고 여러버 ㄴ반ㅈ복되어있고
#accuracy, kappa 기준이 있다. 
#k는 반복의 횟수
plot(knnFit)
#가로축은 k
#세로축은 정확도

pred_test<-predict(knnFit, newdata=test)
confusionMatrix(pred_test, test$Class)



#변수 중요도 확인
importance_knn<-varlmp(knnFit, scale=FALSE)
plot(importance_knn)

