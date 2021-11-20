#naive Bayes classification

#caret -> naivebayes 패키지 
#     -> bnclassify 패키지
#     -> klaR 패키지

#naivebayes 패키지 가장 추천하신대. 편리해서


#나이브베이즈 결과해석

#usekernel -> 커널 밀도 추정 (KDE) 데이터의 히스토그램을 보고 실제 분포 추정하는 것. 
#         -> 커널 밀도 추정할거니 말거니

#bandwith 값이 달라지면 추정 커널 밀도 함수 형태가 달라진다. 이거 바꾸면서 실제 분포 추정하는 것
#adjust -> bandwith 값 조정하는 것

#laplace -> 라플라스스무딩! (스무딩의 한 방식)
#xi=i가 나온 횟수 N=전체 시행횟수 알파 = 스무딩파라미터(이값이 0이면 스무딩 없음)
#동전에서 i는 1,2(앞, 뒤) or 0,1
#데이터 수가 적을 때 0 또는 1과 같이 극단적인 값으로 추정하는 것 방지하기 위해 더한다.(라플라스 스무딩 식 참조)


library(caret)

rawdata<-read.csv(file='C:/Users/pc/Desktop/패스트캠퍼스_데이터분석/part-6-강의자료-장철원강사님/Part 6_강의자료_장철원강사님/Ch02. k-Nearest Neighbor/Data/wine.csv',header=TRUE)
rawdata$Class<-as.factor(rawdata$Class)
str(rawdata)

analdata<-rawdata

set.seed(2020)
datatotal<-sort(sample(nrow(analdata), nrow(analdata)*.7))
train<-rawdata[datatotal,]
test<-rawdata[-datatotal,]
str(train)
train_x<-train[,1:13]
train_y<-train[,14]
test_x<-test[,1:13]
test_y<-test[,14]


ctrl<-trainControl(method='repeatedcv', repeats=5)
nbFit<-train(Class~.,
             data=train,
             method='naive_bayes',
             trControl=ctrl,
             preProcess = c('center','scale'),
             metric='Accuracy')
nbFit
# laplace =0. 알파값이 0이다. laplace 스무딩을 하지 않았다.
# 커널도 사용하지 않았다.
#adjust =1이다. 이건 usekernel TRUE일때만 의미가 있음.

plot(nbFit)
#커널 사용안했ㅇ르 때 정확도가 높다.

pred_test<-predict(nbFit, newdata=test)
confusionMatrix(pred_test, test$Class)

#정확도가 94.4%

#변수 중요도
importance_nb<-varlmp(nbFit, scale=FALSE)
plot(importance_nb)
#ROC 커비의 면적이 넓을수록 중요도가 상승한다.
# 하나의 feature가 클래스를 얼마나 잘 분류를 하느냐
