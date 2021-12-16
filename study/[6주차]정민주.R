#함수

#퀴즈1.
#for문을 활용하여 month.name 벡터로부터 다음과 같이 월 이름을 출력하기
for(i in month.name){
  cat('the month of ', i,'\n')
}

#퀴즈2.
#짝수이면 TRUE 홀수이면 FALSE를 출력하는 함수를 작성하고 다음 벡터에 대해 테스트하여 결과 제시
f1<- function(x){
  print(x%%2==0)
}
vec = c(-5:5, Inf, -Inf, NA, NaN)
f1(vec)


#짝수의 개수를 세는 함수를 작성하고 vec에 대해 테스트해보기
f2<-function(x){
  count= 0
  for(val in x){
    count = count + (val%%2 ==0)
    print(count)
  }
}
f2(vec)

#주어진 데이터 벡터를 평균1 표준편차 1로 표준화하는 함수를 작성하고 vec2에 대해 테스트하여 결과 제시
f3 <- function(x){
  return((x-mean(x))/sd(x))
}
vec2=c(1:5)
f3(vec2)
