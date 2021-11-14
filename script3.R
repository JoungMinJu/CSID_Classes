#한국 복지패널 데이터를 활용한 한국인의 삶 분석

#변수 1000여개
#가구원 배경 및 개인사, 가구여건 및 복징욕구, 사회복지 가입 및 수급여부
#경제상황, 근로 등

setwd('C:/Users/pc/Desktop/패스트캠퍼스_데이터분석/part-1,2,3-강의자료-김영우강사님/강의자료_김영우강사님/공유/Data/한국복지패널데이터')
dir()


#분석 미션
#1. 성별에 따른 소득 차이
#2. 나이대와 소득의 관계
#3. 연령대에 따른 소득
#4. 연령대 및 성별에 따른 소득


install.packages('foreign')
#어떤 데이터든 열어볼 수 있는 

library(foreign)
library(dplyr)
library(ggplot2)

raw_welfare = read.spss('data_spss_Koweps2014.sav',to.data.frame=T)
welfare=raw_welfare

#데이터 검토
dim(welfare)
str(welfare)
head(welfare)
summary(welfare)
View(welfare)

#변수명
welfare =rename(welfare, sex =h0901_4, birth=h0901_5, income = h09_din)


#[분석 1 성별에 따른 소득]
#1) 변수 검토 및 정제 - 성별
#1-1) 변수 검토, 수정
#1-2) 정제-이상치 확인 및 결측 처리

#2) 변수 검토 및 정제 - 소득
#2-1) 변수 검토, 수정
#2-2) 정제 - 이상치 확인 및 결측 처리

#3) 성별 소득 평균 분석
#3-1) 성별 소득 평균표 생성
#3-2) 그래프 생성


#1-1) 변수 검토 및 수정
class(welfare$sex)
summary(welfare$sex)
table(welfare$sex)
#오류가 없구나. 결측치 처리 할 필요없구나
#항목 이름 부여
welfare$sex = ifelse(welfare$sex ==1, 'male','female')
table(welfare$sex)
qplot(welfare$sex)

# 1-1) 변수 검토, 수정
class(welfare$sex)
summary(welfare$income)
qplot(welfare$income) + xlim(0,10000)
#정제
table(is.na(welfare$income))
#결측치 없구나

#성별 소득 평균표 생성
sex_income = welfare %>% group_by(sex) %>% summarise(mean_income=mean(income))
sex_income
ggplot(data=sex_income, aes(x=sex, y=mean_income))+geom_col()

#태어난 연도 변수 검토
class(welfare$birth)
summary(welfare$birth)
qplot(welfare$birth)

#이상치도 없고 결측치도 없고

welfare$age = 2014-welfare$birth +1
summary(welfare$age)
qplot(welfare$age)


#나이별 소득 평균표 생성
age_income = welfare %>% group_by(age) %>% summarise(mean_income = mean(income))
ggplot(data=age_income, aes(x=age, y=mean_income))+geom_point()
#근데 이런 분석은 문제가 있다.
#만약에 13살이 한 명밖에 없으면? 그건 대표성이 없어짐.

#초년 중년 노년 그룹핑해서 하면 대표성이 있겠지(연령대별로 끊어서 분석)
welfare =welfare %>% mutate(ageg=ifelse(age<30, 'youg',ifelse(age<=59, 'middle','old')))
table(welfare$ageg)
qplot(welfare$ageg)

#연령대별 소득 평균표 생성
welfare_income =welfare %>% filter(age!='young') %>% 
  group_by(ageg) %>% 
  summarise(mean_income=mean(income))

ggplot(data=welfare_income, aes(x=ageg, y=mean_income))+geom_col()

#연령대 및 성별 소득 평균표! 생성

#초년은 양이 너무 작으니까 초년 제외하기
sex_income = welfare %>% filter(age!='young') %>% 
  group_by(ageg, sex) %>% 
  summarise(mean_income =mean(income))
ggplot(data=sex_income, aes(x=ageg, y=mean_income, fill=sex))+geom_col()
#누적 막대 그래프이다. 
#그냥 옆으로 나열하는게 좋지 않을까

ggplot(data=sex_income,aes(x=ageg, y=mean_income, fill=sex))+
  geom_col(position='dodge') 
#포지션 기본값 stack
