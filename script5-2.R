#로지스틱 regression

#boosted logistic reression
# 약한 분류기 여러개 더하는 것.
# 피쳐 하나만 가지고 분류기를 만든 약한 분류기 여러개 하비는 것이다.

#logistic model tree LMT
#로지스틱 회귀와 의사결정나무를 합친 모형

#penalized logistic regression #plr
# 제한을 두고 베타를 구하는 것(모델의 복잡성 조절을 위해서)
# 람다를 곱해줘서 베타 영역의 크기를 조절할 수 있다. 

#regularized logistic regression
#l1 정규화(라쏘)
#l2 정규화(릿지)
# regLogistic
# loss = l1, l2, --> primal = 파라미터 기준 최적화! dual 제약 변수 기준 최적화
# 입실론 = 제약 기준 (학습을 멈추는 일종의 기준)
# cost =  평균을 구함 loss의 평균! (전체 데이터셋 대상)
# loss = ex) 실제값-예측값 (각각의 데이터 포인트 대상)

#로지스틱 regresion 예제 실습
library(caret)
rawdata<-read.csv(file='C:/Users/pc/Desktop/패스트캠퍼스_데이터분석/part-6-강의자료-장철원강사님/Part 6_강의자료_장철원강사님/Ch03. Logistic Regression/Data/heart.csv', header=TRUE)
str(rawdata)
#bool값이 많구나 전처리가 필요하고나

#타켓클래스 범주화
rawdata$target<-as.factor(rawdata$target)
unique(rawdata$tar)

#연속형 독립변수 푲ㄴ화
rawdata$age<-scale(rawdata$age)
rawdata$trestbps<-scale(rawdata$trestbps)
rawdata$chol<-scale(rawdata$chol)
rawdata$thalach<-scale(rawdata$thalach)
rawdata$oldpeak<-scale(rawdata$oldpeak)
rawdata$slope<-scale(rawdata$slope)

#범주형 독립변수 as.factor
newdata <- rawdata
factorVar <- c("sex", "cp", "fbs", "restecg", "exang", "ca", "thal")
newdata[ ,factorVar] = lapply(newdata[ ,factorVar], factor)

#트레이닝 테스트 나누기 7:3
set.seed(2020)
datatotal <- sort(sample(nrow(newdata), nrow(newdata)*.7))
train <- newdata[datatotal,]
test <- newdata[-datatotal,]
train_x <- train[,1:12]
train_y <- train[,13]
test_x <- test[,1:12]
test_y <- test[,13]

#학습
ctrl<-trainControl(method='repeatedcv', repeats=5)
logitFit<-train(target~.,
                data=train,
                method='LogitBoost',
                trControl=ctrl,
                metric='Accuracy')
logitFit
#21번 반복시 가장 높은 정확도가 나온다.

plot(logitFit)
#x 축 부스팅횟수, y축 정확도도
#21번째 이후로 정확도가 급격히 떨어지므로 우리는 21을 선택해야한다.

pred_test<-predict(logitFit, newdata=test)
confusionMatrix(pred_test, test$target)
#Accuracy 그닥 높지 않네
importance_logit <-varlmp(lofitFit, scale=FALSE)
plot(importance_logit)
#변수 중요도에 대해 plot을 그려준다. 