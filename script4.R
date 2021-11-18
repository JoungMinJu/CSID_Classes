#4강
#데이터 속 의문 해결하기

#가설 검정이란 가설을 검정(test)한다는 것.
#가설 검정은 집단간의 차이를 검정한다.


#파란색 집단과 노란색 집단의 '수입' 차이 분석
#1. 가설 세운다.
# -두 집단 수입 차이 없다(귀무가설)
# - 노란 집단이 파란 집단보다 수입 높다(대립가설)

#이때 귀무가설 = > 기존에 존재하던 가설, 영향도 없음.
# 분석가의 주장과 반대 가설

#대립 가설 : 차이가 있거나 영향도 있는 분석가가 채택하고싶은 가설
# - 대립 가설; 두 집단 수입차이가 있다.
#  -- 파란수입 >노란수입 or 노란수입 >파란 수입 (양측검정!)
# 양측검정은 양쪽으로 검정을 한다고 해서 양측검정(정규분포에서)

#대립가설 : 노란집단 수입이 파란 집단보다 높다.
# --- 파란수입 < 노란수입만 고려함! (단측검정)

#실무는 상황에 따라 다르지만 대게 단측검정을 사용한다.

#가설검정 1. 가설을 세운다. (귀무가설과 대립가설을 세운다.)
#2. 기준을 세운다. (뭐가 맞는지 판단하는 기준) == 검정통계량을 구한다. 
#3. 결론을내린다. (p-value 참고)

#가설 검정 방법 !!!!!!!!!!!!!
# 1. 내가 보는 데이터가 분할표인가? (흡연과 폐암의 관계성 보는 것과 같은 연관성 볼 때 자주 사용)
# 1-1 맞다 ==> 카이제곱 검정
# 1-2(=2) 그룹의 개수가 2개가 넘느냐? 
# 2-1 넘는다 ==> ANOVA 검정
# 2-2(=3) 안넘는다 ==> 데이터개수가 30개가 넘느냐?(넘으면 대표본,안넘으면 소표본)
# 3-1 네 ==> z검정
# 3-2(=4) 아니요 ==> 대응표본? (A와 B에 있는 사람이 서로 같은 경우 )(ex. before and after)
# 4-1 대응표본 t 검정
# 4-2 t 검정


#[두 집단 평균 차이 검정]
# t검정에 해당하는 내용
# 1. 데이터의 출처를 생각해야함.
# -- 모집단과 표본. 모집단 = 관심대상 전체집합,표본집합  = 관심대상 부분집합
# 2. 관심대상을 찾아내야한다.

# 관심대상 = 경기도 20대 남성 평균 키
#모집단 == 경기도!!에 살고있는 모든 ! 20대 남성들의 키 모음
# 표본 == 모집단에서 추출된 2-0대 남성들의 키 모음

# 모수= parameter = 모집단을 나타내는수 => 추정할 뿐 정확히 알긴 힘들다.

#3. 두 집단 평균 차이 검정

#대표값 = 집단을 대표하는 값 (평균, 분산, 개수도 중요)

#T-test는 정규성을 따라야한다. <30개면!
#만약 안따르면(비모수 검정)한다.
#가설설정 -> 데이터 정규성 검정 -> 분산 동질성 검정
# (분산 같음다름에 따라서 t값 수식이 달라진다.)
# -> T-test -> 결론 

#T값은 (그룹1평균 - 그룹2평균)/(표준편차) ~> t분포를 따른다.
#정규분포보다 꼬리가 조금 더 두껍다.
#자유도 (데이터의 수와 관련있음)의 영향을 받는데 이 자유도가 커질수록 정규분포에 수렴한다.

#pvalue가 <0.05면 귀무가설 기각.
#pvalue = 귀무가설이 참이라고 했을 때 표본 데이터가 수집될 확률
#p-value가 낮을수록 대립가서 ㄹ채택
#0.05를 유의수준이라하며 대게 0.05 또는 0.01 중 선택


#R데이터 분석
#각 집단 샘플사이즈가 3인 소표본 검정
rawN3 <- read.csv(file="C:/Users/pc/Desktop/패스트캠퍼스_데이터분석/part-4-강의자료-장철원강사님/Part 4 강의자료_장철원강사님/Data/htest01.csv", header=TRUE)
groupA = rawN3[rawN3$group=='A',]
groupB = rawN3[rawN3$group =='B',]

#각 집단의 평균 구해서 비교하기
mean(groupA[,2])
mean(groupB[,2])

#정규성 검정
shapiro.test(groupA[,2])
#p-value >0.05 귀무가설 채택(데이터세싱 정규분포를 따른다.)

qqnorm(groupA[,2])
qqline(groupA[,2])
#데이터가 거의 그 정규성을 따르는구나.직선이랑 가깝다..

shapiro.test(groupB[,2])
qqnorm(groupB[,2])
qqline(groupB[,2])

#분산 동질성 검정
#귀무가설 = 두 집단간 분산이 동일하다

var.test(groupA[,2], groupB[,2])
#0.05보다 크므로 귀무가설이 옳다.

#T-test
t.test(groupA[,2], groupB[,2], alternative='less', var.equal=TRUE)
#alternative는 less=>대립가설에서 왼쪽 값이 오른쪽 값보다 작다.
#p-value >0.05 == 유의미한 차이가 없다. 

#[검정2]
#각 집단의 샘플사이즈가 10(소표본)
rawN10 <- read.csv(file="C:/Users/pc/Desktop/패스트캠퍼스_데이터분석/part-4-강의자료-장철원강사님/Part 4 강의자료_장철원강사님/Data/htest02.csv", header=TRUE)
groupA = rawN10[rawN10$group=='A',]
groupB = rawN10[rawN10$group =='B',]

mean(groupA[,2])
mean(groupB[,2])

#데이터 정규성 검정
shapiro.test(groupA[,2])
shapiro.test(groupB[,2])
#데이터 분산
var.test(groupA[,2],groupB[,2])
#--> 데이터 분산이 서로 다르다!

t.test(groupA[,2], groupB[,2], alternative = 'less',var.equal=FALSE)
#대립가설 채택


#[대응표본 t검정]
raw_d <- read.csv(file="C:/Users/pc/Desktop/패스트캠퍼스_데이터분석/part-4-강의자료-장철원강사님/Part 4 강의자료_장철원강사님/Data/htest02d.csv", header=TRUE)
#마케팅에 따른 판매액의 차이를 검정할 것.
groupAd<-raw_d[,1]
groupBd<-raw_d[,2]
mean(groupAd)
mean(groupBd)
#대립가설 = 판매액이 증가했나보네

#분산 동일성 검정할 필요가 없음. 대응표본 t검정에서.
#왜냐면 분산의동질성을 검증한다는 것은 집단이 두개라는 것인데
#우리는 A,B에서 파생된 d라는 집단 하나를 보기 때문에 필요가 없다.

#정규성 검정함. 그룹의 차이를!
d=groupAd-groupBd
shapiro.test(d)
#정규성을 따른다.

#t-test
t.test(groupAd, groupBd, alternative='less', paired=TRUE)
#대응표본이냐? paid 옵션
#대립가설 채택. 마케티을 통해 판매액이 증가했다.


#z 검정(데이터 개수 30개 이상 == 대표본)
#z값은 z분포 따른다. z분포는 정규분포와 같은 말!
#t분포 z분포 모두 0기준 좌우대칭인데, 정규분포(z분포)는 꼬리가 얇음
rawN30 <- read.csv(file="C:/Users/pc/Desktop/패스트캠퍼스_데이터분석/part-4-강의자료-장철원강사님/Part 4 강의자료_장철원강사님/Data/htest03.csv", header=TRUE)
groupA3<-rawN30[rawN30$group =='A', ]
groupB3<-rawN30[rawN30$group =='B', ]
mean(groupA3[,2])
mean(groupB3[,2])
#대립가설 B가 키가 더 크구나

#z-test
#얘는 정규성검정, 동일분산검정 안해도됨.
z-test
#z-test 함수 강사님이 만드신거..
z.test(groupA3[,2], groupB3[,2])

#z테스트 써야하는데 t테스트 쓴다면?
#결과가 달라지게 된다!

#[여러 집단 평균 차이 검정]
#집단 내 오차와 집단 간 오차를 비교한다. 

#총 오차=집단간오차+집단내오차
#(ANOVA 특징)

#집단 내 오차 = 각 데이터값 - 집단 내 평균
#집단 간 오차= 각 집단 평균 - 전체 평균

# 집단간 오차가 집단 내 오차보다 크게 나타났다면? 
# 집단 간 서로 다르다는 뜻

#F 통계량 사용한다. --> F 분포 따른다. 

raw_anova<- read.csv(file="C:/Users/pc/Desktop/패스트캠퍼스_데이터분석/part-4-강의자료-장철원강사님/Part 4 강의자료_장철원강사님/Data/htest04.csv", header=TRUE)
groupA4<- raw_anova[raw_anova$group=='A', ]
groupB4 <- raw_anova[raw_anova$group=='B',]
groupC4 <- raw_anova[raw_anova$group == 'C',]

mean(groupA4[,2])
mean(groupB4[,2])
mean(groupC4[,2])

#A집단, B집단,C집단 키가 동일하냐!
#대립가설 : 세 집단간 평균 차이가 있다.

#정규성검정
shapiro.test(groupA4[,2])
shapiro.test(groupB4[,2])
shapiro.test(groupC4[,2])

#분산 동질성 검정. 여러 집단 동질성 검정하므

install.packages('lawstat')
library(lawstat)

levene.test(raw_anova$height, raw_anova$group)
#귀무가설이 채택. 세 집단간 분산이 동일하다.
  
#등분산성 검정
bartlett.test(height~group, data=raw_anova)
#세 집단간 분산이 동일하다

#anova test
rawAnova <- aov(height~group, data=raw_anova)
summary(rawAnova)
#p-value가 작으니 세 집단간 평균이 다르다는 것!


#분할표를 이용한 연관성분석 - 카이제곱 검정
# 카이제곱 통계량을 구한다 -> 카이제곱 분포를 따른다.
# F분포와 비슷하게 생겼음. 0보다 큰 수!

raw_chisq<- read.csv(file="C:/Users/pc/Desktop/패스트캠퍼스_데이터분석/part-4-강의자료-장철원강사님/Part 4 강의자료_장철원강사님/Data/htest05.csv", header=TRUE)
rawTable<-table(raw_chisq)

rawTable

#카이제곱 검정
chisq.test(rawTable, correct=FALSE)
#correct 기대값이 5보다 크면 FALSE
#p-value가 작음. 대립가설채택
#흡연 여부와 폐암 유무는 연관성이 있다.

