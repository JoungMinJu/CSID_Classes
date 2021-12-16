#텍스트와 날짜

#--------------------
#1. 'Happy' 'Birthday' 'to' 'you'로 구성된 텍스트벡터 생성한 후, 
#길이와문자 개수의 합을 계산하시오

vec=c('Happy','Birthday','to','you')
length(vec)+sum(nchar(vec))


#--------------------
#2.paste() 함수와 Letters 상수벡터를 이용하여 다음과 같은 문자 벡터 생성
#'A 1' 'B 2' ... 'J 10'
paste(LETTERS[0:10], (1:10))
#'A1','B2',...'J10'
paste(LETTERS[0:10], (1:10), sep='')

#--------------------
#3. outer()함수와 paste()함수를 이용해서 다음과 같은 결과를 만드시오

outer((1:6),(1:6),FUN=paste)

#-------------------
#4.USArrests 데이터셋으로부터 New 단어가 포함된 주로 구성된 서브셋을 생성하고
#각 변수의 평균을 구하시오
a=grep("New",row.names(USArrests), value=TRUE)
a
colMeans(USArrests[a,])


#------------------
#1. 2020년 6월 1일부터 7일간의 월, 일, 요일을 seq()함수를 이용하여 생성하고
# 다음과 같은 형식으로 출력하시오
start=as.Date('2020-06-01')
format(seq(from=start,by='day',len=7),"%a-%m%d")
