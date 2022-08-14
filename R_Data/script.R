#CH_1 , CH_2

library(ggplot2)
#ggplot2 패키지 안의 데이터
head(mpg)
#각 열 하나하나가 변수가 된다.

mean(mpg$hwy)
max(mpg$hwy)
min(mpg$hwy)
#히스토그램그리기
hist(mpg$hwy)

#변수만들기
a<-1
b<-2
c<-3
ab<-3.5

a+b
4/b
5*b

d<-c(1,2,3,4,5)
e<-c(1:5)
f<-seq(1,5)
g<-seq(1, 10, by=2)
d
d+2
d+e

a2<-"a"
b2<-"text"
c2<-"Hello WOrld"
e2<-c('Hellow','World','is','good')
e2

#빈도 그래프 만들기
b = c('a','a','b','c')
qplot(b)

#빈칸 구분자로 이어붙이기
paste(e2, collapse = ' ')

#함수 사용하려면 패키지 설치 및 로드해야한다.
#패키지는 함수 꾸러미
#내장함수는 굳이 안그래도된다
#ex) ggplot2 = 시각화 패키지
# qplot(), geom_histogram(), geom_line()
mpg
qplot(data = mpg, x = hwy)
qplot(data=mpg, y =hwy, x = drv,geom= 'point')
#geom이 그래프의 모양을 결정한다.
qplot(data=mpg, y =hwy, x = drv,geom= 'boxplot')
qplot(data=mpg, y =hwy, x = drv,geom= 'boxplot', color = drv)
#drv 별로 색깔 다르게

#함수의 사용법이 궁금하면
?qplot
#매뉴얼 예시 가져온거
qplot(mpg, wt, data =mtcars)
qplot(mpg, wt, data =mtcars, color = cyl)
qplot(mpg, wt, data =mtcars, size =cyl)
qplot(mpg, wt, data =mtcars, facets =vs ~am)
#자주쓰는 함수 가져오고싶으면 ggplot2 cheat sheet 하면됨.

#데이터프레임 만들기
history <- c(90, 80,60,70)
math <- c(50,60,100,20)

#변수 합해서 데이터프레임 만들기
mf_midterm <- data.frame(history, math)
mf_midterm

#반 추가하기
class <- c(1,1,2,2)
df_midterm <- data.frame(mf_midterm, class)
mean(df_midterm$history)
mean(df_midterm$math)

setwd('C:/Users/pc/Desktop/패스트캠퍼스_데이터분석/part-1,2,3-강의자료-김영우강사님/강의자료_김영우강사님/공유/Data')

#엑셀 데이터 불러오기
install.packages('readxl')
library(readxl)

#파일 불러오기
df_finalexam = read_excel('finalexam.xlsx', sheet =1, col_names =T)
df_finalexam
mean(df_finalexam$math)

#csv 파일 불러오기
#메모장으로 열어보면 ,로 구분되어있음을 알 수 있음! 
#콤마 seperate value #csv는 범용 데이터 파일!
#동일한 데이터 파일이어도 용량이 적고 데이터 로드 속도가 빠름

#csv 파일은 시트의 개념이 없음
read.csv('csv_exam.csv', header=T)
#csv로 저장하기
write.csv(df_finalexam, file='output_newdata.csv')


#데이터의 특성 파악하기
exam = read.csv('csv_exam.csv')
exam
head(exam) #default = 6개
head(exam, 10)
tail(exam, 10)
#뷰어 창에서 보여줌
View(exam)
dim(exam)
str(exam)
#요약통계량 출력
summary(exam)

#ggplot2의 mpg 데이터를 데이터 프레임 형태로 불러오기
#긍까 일단 mpg 데이터만 쏙 뽑아오겄다.ggplot2::mpg
mpg <- as.data.frame(ggplot2::mpg)
mpg
head(mpg)

#데이터의 변수명 바꾸기
install.packages('dplyr')
library(dplyr)

df_raw = data.frame(var1 = c(1,2,1),
                    var2 = c(2,3,2))
df_raw

#데이터 수정 전에 무조건 백업본을만들어놓아야한다.
df_new = df_raw
#변수명 바꾸기
df_new = rename(df_new, v2 = var2 ) #var2를 v2로 변경
#할당 안하면 df_new는 변경이 안됨! 할당 무조건해야지 상태유지 가능
df_new

df_raw = as.data.frame(ggplot2::mpg)
head(df_raw)
df_new = df_raw
head(df_new)
df_new = rename(df_new, city = cty, highway = hwy)
head(df_new)

#파생 변수 만들기
df=data.frame(var1=c(4,3,8),
              var2= c(2,6,1))

df$var_sum = df$var1 + df$var2
df$var_mean = (df$var1 + df$var2) /2

#mpg 통합 연비 변수 만드릭
mpg$total = (mpg$cty + mpg$hwy)/2
mean(mpg$total)

#조건문 잉용해 파생변수 만들기
summary(mpg$total)
hist(mpg$total)
#20을 기준으로 잡자. 20이상이며 합격
mpg$test = ifelse(mpg$total>=20,'pass','fail')
head(mpg)

#빈도표로 합격 판정 자동차 수 살펴보기
table(mpg$test)

#그래프로 보기
library(ggplot2)
qplot(mpg$test)

#중첩 조건문활용해서 연비 등급 변수 만들기
mpg$grade = ifelse(mpg$total>=30, 'A', ifelse(mpg$total >=20, 'B','C'))
head(mpg)                   
table(mpg$grad)
qplot(mpg$grade)

#문제 1. ggplot2의 midwest 데이터를 데프 형태로 불러와서 특성파악
df_raw = as.data.frame(ggplot2::midwest)
str(df_raw)
summary(df_raw)
head(df_raw)

#문제2 poptotal전체인구를 total로, popasian 아시아인구를 asian으로 수정
df_new =df_raw
df_new = rename(df_new, asian=popasian, total = poptotal)
head(df_new)

#total과 asian 이용해서 '전체인구대비 아시아 인구 백분율'만들고
#히스토그램을 만들어 도시들이 어떻게 분포하는지 살펴보세요
df_new$prop = (df_new$asian / df_new$total) * 100
hist(df_new$prop)

#아시아 인구 백분율 전체 평균을 구하고large, small
df_new$asia=ifelse(df_new$prop >= mean(df_new$prop), 'large','small')
table(df_new$asia)
qplot(df_new$asia)

#데이터 전처리
#1) 조건에 맞는 데이터만 추출하기
exam=read.csv('csv_exam.csv')

#exam에서 class가 1인 경우만 추출해서 출력
exam %>% filter(class ==1)
#단축키 ctrl+shift+m으로 %>% 기호 입력 가능

#1반이 아닌경우
exam %>%  filter(class !=1)

#초과 미만 이상이하 조건 걸기
exam %>% filter(math > 50)

#여러 조건을 충족하는 행 추추라기
exam %>% filter(class ==1 &math >= 50)
exam %>% filter(math >=90 | english >= 90)

#목록에 해당되는 행 추출하기
exam %>% filter(class %in% c(1,3,5))

#추출한 행으로 데이터 만들기
class1 = exam %>% filter(class ==1)
mean(class1$math)

#자동차 배기량에 따라 고속도로 연비가 다른지 알아보려고 한다.
#disp1이 4 이하인 자동차와 5이상이 ㄴ자동차 중 어떤 자동차의hwy가 평균적으로 더 높은지 알아보세요
df_raw = as.data.frame(ggplot2::mpg)
df_new = df_raw

df_filter1=df_new %>% filter(displ<=4)
df_filter2= df_new %>% filter(displ>=5)
mean(df_filter1$hwy)
mean(df_filter2$hwy)

#자동ㅇ차 제조회사에 따라서 연비가 다른지 알아보기
df_filter3 = df_new %>% filter(manufacturer=='audi')
df_filter4= df_new %>% filter(manufacturer=='toyota')
mean(df_filter3$cty)
mean(df_filter4$cty)

#chevrolet과ford, honda의 자동차의 고속도로 연비 평균
#hwy 전체 평균 구할 것
df_filter5= df_new %>% filter(manufacturer=='chevrolet'|manufacturer=='ford'|manufacturer=='honda')
mean(df_filter5$hwy)

#필요한 변수만 추출하기
exam %>% select(math)
exam %>% select(math, english)
exam %>% select(-math)
exam %>% select(-math, -english)

#dplyr 함수 조합하기
exam %>% filter(class ==1) %>% select(english)
# %>% 은 파이프 함수!

exam %>% select(id, math) %>% head
exam %>% select(id, math) %>% head(10)

#mpg 데이터는 11개의 변수로 되어있습니다. 이중 일부만 추출해서 분석에 활용할 것
#mpg에서 class cty 추출해 새로 데이터 만들기
select1=df_new %>% select(class, cty)
head(select1)
#ㅇ추출한 데이터 이용해서 cty가 더 높은지 확인하기
select_filt = select1 %>% filter(class=='suv') %>% select(cty)
select_filt2 =select1 %>% filter(class=='compact') %>% select(cty)
mean(select_filt$cty)
mean(select_filt2$cty)


#데이터를 정렬하기
exam %>% arrange(math)
exam %>% arrange(desc(math))
exam %>% arrange(class, math)

#아우디에서 생산한 자동차 중 어떤 자동차가 모델의 hwy가 높은지 1~5위
df_new
df_new %>% filter(manufacturer=='audi') %>% arrange(desc(hwy)) %>% head(5)

#파생변수 추가하기
exam %>% mutate(total= math +english+science) %>% head
#훨씬 간결해진 코드
exam %>% mutate(total = math+english+science, 
                mean =(math+english+science)/3) %>% head
exam %>% mutate(test =ifelse(science >= 60, 'pass','fail')) %>% head
#할당안해도 정렬할 수도 있음
exam %>% mutate(total= math +english+science) %>% arrange(total)

#mpg 복사본만들고cty+hwy 더한 합산연비 변수 추가
df_new=df_new %>% mutate(sum=cty+hwy)
head(df_new)
df_new = df_new %>% mutate(m = sum/2)
df_new %>% arrange(desc(m)) %>% head(3)
df_raw %>% mutate(sum  = cty+ hwy, m = sum/2) %>% arrange(desc(m)) %>% head(3)

#집단별로 데이터 요약하기

#요약통계
#group_by 함수와 연동해서 쓴다.
exam %>% summarise(mean_math=mean(math))
#동시에 여러개 요약 통계량 구할 수 있음
#n()는 각 행의 개수 세는 것
exam %>% group_by(class) %>%  summarise(mean_math = mean(math))
exam %>% group_by(class) %>%  summarise(mean_math = mean(math),
                                        sum_math = sum(math),
                                        median_math = median(math),
                                        n=n()) #학생수

#각 집단별로 다시 집단 나누기
#제조사별로 나누고 drv별로 또 나눔
mpg %>% group_by(manufacturer, drv) %>% summarise(mean_cty=mean(cty)) %>% head(10)

#mpg 어떤 차종의 연비가 높은지
mpg %>% group_by(class) %>% summarise(mena_cty =mean(cty))

#위가 알파벳 순으로 정렬되었는데 이를 cty 평균순 정렬
mpg %>% group_by(class) %>% summarise(mena_cty =mean(cty)) %>% arrange(desc(mena_cty))

#어떤 회사의 hwy가 가장 높은지 평균이 가장 높은 회사 세 곳 출력
mpg %>% group_by(manufacturer) %>% summarise(mean_hwy=mean(hwy)) %>% arrange(desc(mean_hwy)) %>% head(3)

#어떤회사에서 compact 차종을 가장 많이 생성하는지 
mpg %>% group_by(manufacturer) %>% filter(class=='compact') %>% summarise(n=n())


#데이터 합치기
test1 = data.frame(id = c(1,2,3,4,5),
                   midterm = c(60,80,70,90,85))
test2= data.frame(id=c(1,2,3,4,5),
                 final=c(70,80,65,95,80))
total = left_join(test1, test2, by='id') 
#id 기준 결합 "" 붙여야함
total

test1 = data.frame(class = c(1,2,3,4,5),
                   midterm = c(60,80,70,90,85))
test2= data.frame(class=c(1,2,3,4,5),
                  final=c(70,80,65,95,80))
total = left_join(test1, test2, by='class') 
name = data.frame(class = c(1,2,3,4,5),
                  teacher = c('min','man','mon','men','mym'))
total = left_join(total, name, by ='class')
total

#세로로 합치기
d1 = data.frame( id = c(1:5), test=c(seq(70,100, length = 5)))
d1
d2=data.frame(id=c(6:10), test= c(seq(80,100, length = 5)))
group_all = bind_rows(d1,d2)
#여기선 by가 없음
group_all

#mpg데이터 이용해서 분석 문제 해결하기
#f1변수는 연료를 의미
#연료와 가격으로 구성된 데이터 프레임 만들기
fuel = data.frame(fl= c('c','d','e','p','r'),
                  price_fl= c(2.35, 2.38, 2.11, 2.76, 2.22),
                  stringsAsFactors = F)
#stringsAsFactors factor타입이 있음 남자여자,뭐 그런거 같은 범주
#factor로 변환하면 분석할 때 불편한 점이 있을 수 있다요
fuel

#mpg에는 연료 종류 나타내는 fl변수는 있지만 가격은 없음
#mpg 데이터에price_fl 변수 추가하기
head(mpg)
mpg=left_join(mpg, fuel, by='fl')
mpg %>% select(model, fl, price_fl) %>% head(5)


#분석과제

df_raw =as.data.frame(ggplot2::midwest)
df_new =df_raw

#전체 인구 대비 미성년 인구 백분율 변수
df_new = df_new %>% mutate(df_new, t = popadults / poptotal*100)
df_new
#미성년 인구 백분율이 가장 높은 상위 다섯개 지역 미성년인구 백분률 출력
df_new %>% arrange(desc(t)) %>% select(t) %>% head(5)
#미성년 비율등급 변수 추가. 각 등급에 몇 개의 지역이 있는지 알아보기
df_new = df_new %>% mutate (df_new, class = ifelse(t>=40, 'large', ifelse(t>=30,'middle','small')))
table(df_new$class)

#전체인구 대비 아시아인 인구 백분율 변수 추가하고 하위 10개 지역의 주 지역명 백분율 추출
df_new %>% mutate(df_new, t2= popasian/poptotal * 100) %>% arrange(desc(t2)) %>% select(state, county, t2) %>% tail(10)
