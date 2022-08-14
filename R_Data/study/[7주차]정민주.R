# 퀴즈 1 
# A 대학교는 R과 파이썬에서 90점 이상을 맞아야만 졸업할 수 있음.
# 졸업 가능한 학생의 이름을 출력하기

r=read.csv(file = 'C:/Users/pc/Desktop/R=VD 3학년 2학기 학점 4.2 넘었다/R프로그래밍/r_programming_10th_data/r_scores.csv', header=TRUE)
python=read.csv(file='C:/Users/pc/Desktop/R=VD 3학년 2학기 학점 4.2 넘었다/R프로그래밍/r_programming_10th_data/python_scores.csv', header=TRUE)

r
python

# 졸업 가능한 학생의 이름
grad_r=r$Name[r$Score>=90]
grad_python= python$Name[python$Name>=90]

grad_r
grad_python
#두개 다 포함되는 학생의 이름 도출
index = match(grad_r, grad_python)
#NA 제거
index =index[!is.na(index)]
#정답
grad_python[index]



#퀴즈 2
# 주사위를 세번던져서나온 눈들의 평균을 계산하는 과제
#이를 100번 반복하여 시행해보고 나온 값들의 평균과 표준편차 구하기(seed는 1로 고정)

set.seed(1)
dice=1:6
vec=1:100
for (i in  1:100){
  value=sample(dice, 3,replace=TRUE)
  vec[i]=mean(value)
}
print(mean(vec))
print(sd(vec))


#퀴즈 3
#Netflix score가지고 그림 그려보기

net=read.csv(file = 'C:/Users/pc/Desktop/R=VD 3학년 2학기 학점 4.2 넘었다/R프로그래밍/r_programming_10th_data/NetflixOriginals.csv', header=TRUE)

library(ggplot2)

#실수 벡터로 변환
net$Score=as.double(net[['IMDB.Score']])
ggplot(data=net, aes(x=Score))+ geom_histogram()
