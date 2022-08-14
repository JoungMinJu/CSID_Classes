# 시각화를 진행할 타이타닉 데이터 
str(Titanic)
t=Titanic 
#4차원 데이터
t
t=as.data.frame(t)
t
#타이타닉 데이터 프레임에서 한 번도 등장하지 않은 데이터는 제거하기
index = which(t$Freq != 0)
t=t[index,]
row.names(t) = NULL
ttn = data.frame()
t[1,]$Freq
nrow(t)

for(i in (1:nrow(t))){
  for(j in (1:t[i,]$Freq)){
    ttn=rbind(ttn, t[i,][1:4]) 
  }
}
#시각화에 사용할, Titanic을 정제한 데이터프레임
ttn
row.names(ttn)=NULL

ttn

#Class와 survive
plot(ttn$Class, ttn$Survived)
#Sex와 Survive
plot(ttn$Sex, ttn$Survived)
#Age와 Survive
plot(ttn$Age, ttn$Survived)

plot(ttn$Sex)
plot(ttn$Survived)
plot(ttn$Age)

plot(ttn$Sex, ttn$Class)
table(ttn)

t=table(ttn)
margin.table(t, c(1,4))
plot(margin.table(t, c(1,4)))
plot(margin.table(t, c(2,4)))
plot(margin.table(t,c(3,4)))
