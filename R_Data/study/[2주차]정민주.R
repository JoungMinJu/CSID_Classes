#2018112451 정민주
#2주차

#--------------

#퀴즈 1. 
#1부터 12까지 3x4 행렬 생성하여 mtx에 할당하고 letters 상수벡터 이용해 행이름 열이름 지정
mtx<-matrix(c(1:12),3,4)
dimnames(mtx)=list(letters[1:3], letters[1:4])
mtx

#--------------

#퀴즈 2. 
# mtx 행렬로부터 첫번쨰와 세번쨰의 열을 추출하여 부분행렬을 만드록
#두번째와 네번쨰 열을 추출하여 부분행렬을 만든 후 
#이들을 열의 방향으로 결합한 새로운 행렬 mtx.c를 생성하시오

one=mtx[,1:3]
two=mtx[,2:4]
mtx.c=cbind(one, two)
mtx.c

#아니면
mtx.c=cbind(mtx[,1:3],mtx[,2:4])



#--------------

#퀴즈 3. 
#다음 코드의 결과 예상

matrix(1:4,nrow=2)^(1:4)
#     [,1] [,2]
#[1,]  1^1  3^3  
#[2,]  2^2  4^4


#--------------

#퀴즈 4.
mtx=matrix(1:9999, ncol=3)
mtx[(dim(mtx)[1]-2):(dim(mtx)[1]), (dim(mtx)[2]-1):(dim(mtx)[2])]           


#--------------

#퀴즈 5. 
#777번쨰 행 3번째 여에 있는 값 추출

matrix(1:10000,1000)[777,3]


#--------------

#퀴즈 6.
#계층이동 모델 확률 테이블을 행렬로 만들기
mtx=matrix(c(0.43,0.04,0.01,0.49,0.72,0.52,0.08,0.24,0.47),ncol=3, dimnames=list(c('lower','middle','upper'),c('lower','middle','upper')))           
mtx

#행의 합이 1이 됨을 확인
rowSums(mtx)

#두 세대에 걸친 계층 이동 모델의 확률 테이블 생성
mtx.two=mtx%*%mtx
mtx.two

#하위 중위 상위 계층이 각각 손자세대에서 상위계층으로 이동할 확률
mtx.two[,'upper']

#--------------

#퀴즈 1.
#A를 alpha로 대체

lst=list(c(3,5,7),c('A','B','C'))
lst[[2]][[1]]='alpha'
lst


#--------------

#퀴즈 2.
#다음 리스트에서 첫 번쨰 원소의 값에 각 1을 더하시오

lst=list(alpha=0:4,beta=sqrt(1:5),gamma=log(1:5))
lst[[1]]=lst[[1]]+1
lst

#--------------

#퀴즈 3.
#리스트를 생성하여 lst에 할당한 후 다음을 수행하시오

lst=list(month.name, month.abb)
lst

#두 원소의 이름을 각각 mon.name과 mon.abb를 지정하시오
names(lst)=c('mon.name','mon.abb')
lst

#원소 mon.name과 mon.abb의 길이의 합을 구하시오
length(lst$mon.name)+length(lst$mon.abb)

#lst 리스트에 1부터 12까지의 숫자를 세번째 원소 mon.num이름으로 추가하시오
lst$mon.num=c(1:12)
lst


#--------------

#퀴즈

temp=list()
temp[month.abb]=c(-2.4,0.4,5.7,12.5,17.8,22.2,24.9,25.7,21.2,14.8,7.2,0.4)
temp

#0도 미만인 월 추출
names(temp[temp<0])
#연 평균 기온보다 작은 월을 리스트로부터 제거하시오
temp[temp<mean(unlist(temp))]=NULL

