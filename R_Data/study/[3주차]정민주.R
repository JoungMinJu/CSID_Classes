#2018112451
#3주차

#iris 데이터 셋에 다음 작업 수행하시오

#(a) iris데이터셋의 데이터 구조를 확인하고 숫자 변수만으로 구성된 새로운 데이터 프레임 iris.new를 생성하시오
print(str(iris))
iris.new=subset(iris, select=-Species)
print(iris.new)

#(b)새로운 데이터 프레임 iris.new의 열 평균을 구하시오
print(colMeans(iris.new))


#--------------------------------------------------------

#표준 패키지에 포함되어있는 USArrests 데이터셋에 대해서 다음 작업으로 수행하시오
#(a) USArrests 데이터세스이 데이터 구조 확인하고, 다섯행 간격으로 데이터를 화면에 출력하시오
print(str(USArrests))
print(USArrests[seq(1,dim(USArrests)[1],5),])

#(b) 10만명당 살인사건 발생 건수가 15를 초과하는 주는?
print(USArrests[USArrests$Murder>5,])

#--------------------------------------------------------

#VADeaths 데이터셋에 대해 작업수행

#(a) VADeaths 데이터 구조를 확인하며 데이터 프레임으로 변환하시오
print(VADeaths)
print(data.frame(VADeaths))
VADeaths=data.frame(VADeaths)

#(b) 행 이름 이용하여 Age 변수 추가하고 기존의 행 일므 삭제
row.names(VADeaths)
VADeaths$Age=row.names(VADeaths)
row.names(VADeaths)=NULL
print(VADeaths)

#(c) 네개 사망률 변수에 대한 평균을 계산하여 Average 변수에 추가하시오
VADeaths$Average=rowMeans(VADeaths[1:4])
print(VADeaths)

#(d) Age 변수와 Average 변수가 첫번째 두번쨰 열에 위치하도록 순서 조정
VADeaths=VADeaths[,c(5,6,1:4)]
print(VADeaths)


#--------------------------------------------------------

#MASS 패키지의 mammals 데이터셋에 대해 다음 작업 수행

#(a) mammals 데이터셋의 구조를 확인하고 size 열을 추가하여 몸무게가 중위수보다 큰 동물에는 large를 작은 동물에는 small을 저장하시오
library(MASS)
str(mammals)

mammals$size=ifelse(mammals$body>median(mammals$body), "large","small")
print(mammals)


#(b) size가 large인 동물 추출
m=mammals
print(m) #그냥 쉽게 지칭하기위해

print(row.names(m[m$size=='large',]))

#(c)몸무게가 가장 큰 동물과 가장 작은 동물 추출
print(row.names(m[m$body==max(m$body),]))
print(row.names(m[m$body==min(m$body),]))

#(d) 두뇌무게 대 몸무게의 비율을 퍼센트로 환산하여 brain.percent열에 추가하고 이 비율이 가장 큰 동물과 가장 작은 동물을 추출하시오
print(with(m, brain/body))
m$percent=with(m,brain/body)      
print(row.names(m[m$percent==max(m$percent),]))
print(row.names(m[m$percent==min(m$percent),]))

#(e) 두뇌 대 몸무게 비율이 중위수보다 크면서 크기가 large인 동물 추출
print(row.names(m[m$percent>median(m$percent)&m$size=='large',])) #--1번

print(subset(m, subset=(percent>=median(percent)&size=='large'))) #--2번
