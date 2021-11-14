#3강

#R로 그래프 만들기
#- 2차원 그래프/3차원 그래프/지도그래프/네트워크그래프/모션차트/인터랙티브 그래프

#산점도 막대그래프 선그래프 박스플롯 많이 쓴다.
#ggplot2 레이어구조 사용하기
#1단계 배경설정(축)
#2딴계 그래프 추가(점, 막대, 선)
#3단계 설정추가(축,범위, 색, 표식)

#산점도 만들기

library(ggplot2)
#축 형성 첫번째 레이어
ggplot(data=mpg, aes(x=displ, y=hwy))
#산점도 추가
ggplot(data =mpg, aes(x=displ, y=hwy))+geom_point()
#x축에 제한값
ggplot(data =mpg, aes(x=displ, y=hwy))+geom_point()+xlim(3,6)
ggplot(data =mpg, aes(x=displ, y=hwy))+geom_point()+xlim(3,6)+ylim(10,30)
#ggplot은 함수를 +로 연결해준다.



#mpg 데이터의 cy와 hwy 연비 간 어떤 관계가 있는지 알아보기
ggplot(data=mpg, aes(x=cty, y=hwy))+geom_point()

#midwest 데이터 이용해서 전체 인구와 아시아인 인구간의 관계 찾기
ggplot(data=midwest, aes(x=poptotal, y=popasian)) +geom_point()+ ylim(0, 10000) + xlim(0,500000)

#막대그래프 = 집단 간 차이 표현하기
#성별 소득 차이 처럼 집단 간 차이를 표현할 때 주로 사용한다.

#집단별 평균표 만들기
library(dplyr)
df_mpg = mpg %>% group_by(drv) %>% summarise(mean_hwy =mean(hwy))
df_mpg

ggplot(data=df_mpg, aes(x=drv, y=mean_hwy))+geom_col() #막대그래프프
#근데 막대그래프가 알파벳 순으로 출력되는 중
#x축 정렬
ggplot(data=df_mpg, aes(x=reorder(drv,-mean_hwy), y=mean_hwy))+geom_col() 
#mean_hwy 기준 내림차순 정렬(x축을)

#빈도 막대 그래프 (값의 개수 세서 막대의 길이 표현하는 그래프)
ggplot(data=mpg, aes(x=drv))+geom_bar()

ggplot(data=mpg, aes(x=hwy))+geom_bar()


#1. suv차종 대상으로 평균cty가 가장 높은 회사 다섯곳을 막대그래프로 표현하기
df_mpg = mpg %>% group_by(manufacturer) %>% filter(class=='suv') %>% summarise(mean_cty=mean(cty))
df_mpg
ggplot(data = df_mpg, aes(x=reorder(manufacturer, -mean_cty), y=mean_cty)) + geom_col()

#자동차 종류별 빈도
ggplot(data= mpg, aes(x=class))+geom_bar()


#선그래프는 시간에 따라 달라지는 데이터 표현하기
ggplot(data=economics, aes(x=date, y=unemploy))+geom_line()

#psavert(개인 저축률 시간에 따라 변화한 거 보여주기)
ggplot(data=economics, aes(x=date, y=psavert))+geom_line()

#상자그림 만들기
#집단간 분포 차이를 표현하는 것.
ggplot(data=mpg, aes(x=drv, y=hwy))+geom_boxplot()

#상자 가로선 중앙값/상자 밑면 1분위수/상자윗면 3분위수
#세로선은 최소값 최대값
#점은 극단치 (+-1.5IQR 밖에있는 애들)

#compact, subcompact, suv인 자동차의cty가 어떻게 다른지 비교하기
df_mpg = mpg %>% filter(class=='compact'|class=='subcompact'|class=='suv')
df_mpg = mpg %>% filter(class%in%c('compact','subcompact','suv'))
ggplot(data=df_mpg, aes(x=class, y=cty))+geom_boxplot()


#결측치 정제하기
#누락된 값, 비어있는 값

#결측치 표기 NA
df=data.frame(sex=c('m','f',NA,'m','f'),
              score = c(5,4,3,4,NA))
df

#결측치 확인 절차
is.na(df)
#결측치 빈도 출력
table(is.na(df))

#filter써서 결측치 빼자
#근데 조건 부분에 어떤 변수에 대해서 결측치가 있는지 지정해야함.
table(is.na(df$sex))
table(is.na(df$score))
#변수별로na개수 파악가능해짐

mean(df$score)
#NA
sum(df$score)
#NA

#결측치 행 제거하기
df %>% filter(!is.na(score))
df_nomiss = df %>% filter(!is.na(score),!is.na(sex))
df_nomiss = df %>% filter(!is.na(score)&!is.na(sex))

mean(df_nomiss$score)
sum(df_nomiss$score)


#결측치 제거하는 거 있음
na.omit(df)
#근데 성별, 소득 관계 분석하는데 지역 결측치까지 제거해버림
#데이터 손실이 많ㄴ이 나와서 사용하지 않는다.
#결측치가 하나라도 있으면 모델 못만드는 머신러닝 알고리즘에서 활용한다.

#함수의 결측치 제외 기능 활용하기
mean(df$score, na.rm =T) #결측치 제외하고 평균 산출
sum(df$score, na.rm=T)
#근데 R에서 쓰는 모든 함수가 이 파라미터를 갖고 있진 않는다.

exam=read.csv('csv_exam.csv')
exam[c(3,8,15),'math']=NA
#평균 구하기
exam %>% summarise(mean_math = mean(math))
exam %>% summarise(mean_math = mean(math, na.rm = T))

exam %>% summarise(mean_math = mean(math, na.rm=T),
                   sum_math =sum(math, na.rm=T),
                   median_math = median(math, na.rm=T))


#결측치 대체법
#대표값으로 일괄 대체
#통계 분석 기법 적용, 예측값 추정해서 대체

mean(exam$math, na.rm=T)
exam$math = ifelse(is.na(exam$math), 55, exam$math)
table(is.na(exam$math))

#분석해보기
mpg=as.data.frame(ggplot2::mpg)
mpg[c(65,124,131,153,212),'hwy']= NA

#drv 별로 hwy 평균이 어떻게 다른지 알아볼 것이다. 
#drv와 hwy 에 결측치 몇개 있는지 확인하기
table(is.na(mpg$drv))
table(is.na(mpg$hwy))

#filter활용해서 hwy 변수의 결측치 제외하고 어떤 구동방식의hwy 평균이 높은지 알아보기
mpg %>% filter(!is.na(hwy)) %>%group_by(drv) %>% summarise(mean_hwy =mean(hwy)) %>% arrange(desc(mean_hwy))


#이상치 정제하기
#이상치 포함시 분석 결과 왜곡
#결측 처리 후 제외하고 분석


outlier=data.frame(sex =c(1,2,1,3,2,1),
                   score =c(5,4,3,4,2,6))

#이상치 있는지 알고싶으면 빈도분석 하면 되지 머
table(outlier$sex)
table(outlier$score)

#일단 이를 결측치로 바꿔놓자 
outlier$sex = ifelse(outlier$sex ==3 , NA, outlier$sex)
outlier$score = ifelse(outlier$score >5, NA, outlier$score)

outlier %>% filter(!is.na(sex)&!is.na(score)) %>% 
  group_by(sex) %>% 
  summarise(mean_score = mean(score))


#극단적인 값! 정상범위 기준 정해서 벗어나면 결측치처리
#논리적 판단으로 내맴대로 할 수도 있고
#통계적으로 판단할수도 있다. 1.5 IQR

mpg=as.data.frame(ggplot2::mpg)
boxplot(mpg$hwy)

#통계치 출력
boxplot(mpg$hwy)$stats
# 두 경계값을 알 수 있음

mpg$hwy = ifelse(mpg$hwy <12 |mpg$hwy>37, NA, mpg$hwy)
table(is.na(mpg$hwy))

#결측치 제외하고 분석하기
mpg %>% group_by(drv) %>% summarise(mean_hwy=mean(hwy,na.rm=T))


#혼자 분석하기
mpg <- as.data.frame(ggplot2::mpg)
mpg[c(10,14,58,93), 'drv']='k' #이상치 할당
mpg[c(29,43,129, 203),'cty'] = c(3,4,39,42)

#drv에 이상치가 있는지 확인하고 결측 처리하기
table(mpg$drv)
mpg$drv = ifelse(mpg$drv %in% c('k'), NA, mpg$drv)
mpg$drv = ifelse(mpg$drv %in% c('4','f','r'), mpg$drv, NA)
mpg

#상자 그림을 이용해서 cty에 이상치가 있는지 확인하기
#상자그림의 통계치를 이용해서 정상 범위를 벗어난 값을 결측처리한 후 
#다시 상자그림을 만들어 이상치가 사라졌는지 확인하세요
boxplot(mpg$cty)$stats
mpg$cty = ifelse(mpg$cty > 26 |mpg$cty < 9, 17, mpg$cty)
boxplot(mpg$cty)

#이상치 결측처리했으니 분석하기 
#이상치 제외하고drv 별로cty 평균이 어떻게 다른지 알아보기
mpg %>% filter(!is.na(drv)&!is.na(cty)) %>% group_by(drv) %>% summarise(mean_cty = mean(cty, na.rm=T))
