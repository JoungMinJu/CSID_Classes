###############
# 데이터 로드 #
###############
df = read.csv(file = 'C:/Users/pc/Downloads/archive/TV_final.csv', header=TRUE)

###############
# 데이터 구조 #
###############
head(df)
str(df)
dim(df)

###############
# 결측치 확인 #
###############
sum(is.na(df$Brand))
sum(is.na(df$Resolution))
sum(is.na(df$Size))
sum(is.na(df$Selling.Price))
sum(is.na(df$Original.Price))
sum(is.na(df$Operating.System))
sum(is.na(df$Rating))

### OS의 결측치(빈문자열) 비율 확인
s=as.numeric(sum(df$Operating.System==''))
t=as.numeric(length(df$Operating.System))
(s/t)*100

#원본 데이터프레임의 복사본 copy_df
copy_df=df

### ---> OS의 최빈값 Android로 결측치 대체
copy_df$Operating.System[df$Operating.System=='']='Android'
sum(copy_df$Operating.System=='')

### Rating의 결측치 비율 확인
s.rate = as.numeric(sum(is.na(df$Rating)))
t.rate = as.numeric(length(df$Rating))
(s.rate/t.rate)*100
### ---> Rating의 평균값으로 결측치 대체
copy_df$Rating[is.na(copy_df$Rating)]=mean(copy_df$Rating, na.rm=TRUE)

### Brand의 unique 값 확인
sort(unique(copy_df$Brand))

### ---> SAMSUNG과 samsung의 통합
copy_df[copy_df$Brand=='Samsung',]$Brand='SAMSUNG'

### ---> 공백 처리
install.packages('stringr')
library(stringr)
copy_df$Brand=str_trim(copy_df$Brand)



###############
#   시각화    #
###############

### Brand별 개수
library(ggplot2)
ggplot(copy_df,aes(Brand,fill=Brand))+geom_bar()+coord_flip()

### Resolution별 개수
ggplot(copy_df, aes(Resolution, fill=Resolution))+geom_bar(width=0.5)+scale_fill_brewer(palette='Oranges')

### Resolution별 개수 - pie plot
resol.one= sum(copy_df$Resolution=='Ultra HD LED')
resol.two= sum(copy_df$Resolution=='QLED Ultra HD')
resol.three = sum(copy_df$Resolution=='HD LED')
resol.four = sum(copy_df$Resolution=='Full HD LED')
resol.five = sum(copy_df$Resolution=='HD Plasma')
resol.df=data.frame(group=c('Ultra HD LED', 'QLED Ultra HD', 'HD LED','Full HD LED','HD Plasma'),value=c(resol.one,resol.two, resol.three, resol.four, resol.five))
ggplot(resol.df, aes(x="", y=value, fill=group)) + geom_bar(stat='identity') + coord_polar('y', start=0)+scale_fill_brewer(palette='Blues')


### Size별 개수
plot(table(copy_df$Size),type='b',xlab='size',ylab='count')


### Brand별 Selling.Price
ggplot(data=copy_df, aes(x=Brand, y=Selling.Price, fill=Brand))+geom_boxplot()+coord_flip()
### -- Brand별 평균 Selling.Price의 최고 다섯개 브랜드, 최저 다섯개 브랜드 도출
par(mfrow=c(1,2))
barplot(sort(tapply(copy_df$Selling.Price, copy_df$Brand, FUN=mean),decreasing =TRUE)[0:5],ylim=c(0,140000),main='High Selling.Price',col= c("lightblue", "mistyrose", "lightcyan", "lavender", "cornsilk"))
barplot(sort(tapply(copy_df$Selling.Price, copy_df$Brand, FUN=mean))[0:5],ylim=c(0,140000),main='Low Selling.Price',col= c("lightblue", "mistyrose", "lightcyan", "lavender", "cornsilk"))


### 해상도별 Selling.Price의 시각화
par(mfrow=c(1,1))
dev.off()
ggplot(data=copy_df, aes(x=Resolution, y=Selling.Price, fill=Resolution)) + geom_boxplot() + scale_fill_brewer(palette='Pastel1') + geom_jitter(color = "black", alpha = .2) + ggtitle('Selling.Price by Resolution') +theme(plot.title=element_text(size=20, color='Chocolate 1',face='bold'))


### OS별 Selling.Price의 시각화
ggplot(data=copy_df, aes(x=Operating.System, y=Selling.Price, fill=Operating.System))+geom_boxplot()+scale_fill_brewer(palette='Pastel2') + geom_jitter(color='black',alpha=0.2)+ ggtitle('Selling.Price by OS')+theme(plot.title=element_text(size=20, color='Orange Red 1',face='bold'))


### 해상도별 평균 Rating의 시각화
tapply(copy_df$Rating, copy_df$Resolution, FUN=mean)
barplot(sort(tapply(copy_df$Rating, copy_df$Resolution,FUN=mean)),xlab='Resolution',ylab='Rating',main='Rate by Resolution',col= c("lightblue", "mistyrose", "lightcyan", "lavender", "cornsilk"))
### OS별 평균 Discount Rate의 시각화
copy_df$Discount = ((copy_df$Original.Price-copy_df$Selling.Price)/copy_df$Original.Price) *100  #할인율 정의
barplot(sort(tapply(copy_df$Discount, copy_df$Operating.System, FUN=mean)),xlab='OS',ylab='Discount(%)', main='DC rate by OS', col= c("Light Pink", "Plum", "Lavender Blush", "Violet", "Deep Pink"))


### --> 우산 개수가 많은 순으로 여섯개의 Brand 도출
sort(table(copy_df$Brand),decreasing=TRUE)[1:6]
top_six_brand = subset(copy_df, subset=(Brand %in% c('SAMSUNG','LG','SONY','TCL','Panasonic','Micromax')),select=c("Brand",'Resolution','Selling.Price'))
head(top_six_brand)
### Resolution별, Brand별 Selling.Price의 시각화
library(lattice)
bwplot(Brand ~ Selling.Price|Resolution,data=top_six_brand , scales = list(cex=0.65), 
       auto.key=TRUE, horizontal=TRUE,
       main=list(label='Selling price by brand and OS.',
                 col='brown',cex=1.4), 
       xlab = list(label = 'Selling.Price', cex=1.2),
       ylab=list(label='Brand',cex=1.2))



##################
#회귀 모형 시각화#
##################

# Resolution별 Size에 의한 Selling Price의 변화에 대한 단순선형회귀 모형 시각화
xyplot(Selling.Price ~Size |Resolution, data=df , layout = c(5,1), type=c('p','r'), col = 'Black',
       main = list(label='Size vs Selling Price by Resolution',col='Orange',cex=1.5),
       xlab = 'Size', 
       ylab = 'Selling.Price', 
       strip = strip.custom(bg = 'Gold' , 
                            par.strip.text = list(col='black', cex=0.8 , 
                                                  font=4)))



##################
#   통계검정(1)  #
##################

barplot(sort(tapply(copy_df$Selling.Price, copy_df$Brand, FUN=mean),decreasing =TRUE)[0:5],ylim=c(0,140000),main='High Selling.Price',col= c("lavender", "gray 90", "lavender", "gray 90", "gray 90"))

# (대립가설) 제조사 Sharp와 LG의 평균 Selling.Price에는 차이가 있다.

# 1) 통계 분석을 위한 데이터프레임 만들기
t.df=subset(copy_df, subset = (Brand %in% c('LG', "Sharp")), select=c('Brand','Selling.Price'))

# 2) 정규성 검정
sum(copy_df$Brand=='LG')
shapiro.test(t.df[t.df$Brand=='Sharp',2])

# 3) 등분산 검정
var.test(t.df[t.df$Brand=='Sharp',2],t.df[t.df$Brand=='LG',2])

# 4) 이분산가정 독립평균 t검정 --> Welch's t-test
t.test(Selling.Price ~ Brand, data=t.df,var.equl=FALSE)

# (결론) 제조사 Sharp와 LG의 평균 Selling.Price에는 유의미한 차이가 없다.





##################
#   통계검정(2)  #
##################

barplot(sort(tapply(copy_df$Rating, copy_df$Resolution,FUN=mean)),xlab='Resolution',ylab='Rating',main='Rate by Resolution',col= c("gray 90", "lavender", "gray 90", "gray 90", "lavender"))
# (대립가설) ‘HD LED’와 'QLED Ultra HD’ 의 평균 Rating에는 차이가 있다.

# 1) 통계 분석을 위한 데이터프레임 만들기
r.df= subset(copy_df, subset=(Resolution %in% c('HD LED','QLED Ultra HD')), select=c('Resolution','Rating'))

# 2) 정규성 검정
sum(copy_df$Resolution =='HD LED')
sum(copy_df$Resolution=='Ultra HD LED')

# 3) 등분산 검정
var.test(r.df[r.df$Resolution=='HD LED',2], r.df[r.df$Resolution=='QLED Ultra HD',2])

# 4) 이분산가정 독립평균 t검정 --> Welch's t-test
t.test(Rating ~ Resolution, data=r.df, var.equal=FALSE)

# ( 결론 ) 'HD LED'와 'QLED Ultra HD'의 평균 Rating에는 유의미한 차이가 있다.





##################
#   통계검정(3)  #
##################

barplot(sort(tapply(copy_df$Discount, copy_df$Operating.System, FUN=mean)),xlab='OS',ylab='Discount(%)', main='DC rate by OS', col= c("lavender", "gray 90", "gray 90", "gray 90","gray 90","gray 90", "lavender"))
# (대립가설) OS ‘Tizen’과 ‘HomeOS’ 의 평균 Discount Rate 에는 차이가 있다.

# 1) 통계 분석을 위한 데이터프레임 만들기
d.df=subset(copy_df, subset=(Operating.System %in% c('Tizen','HomeOS')), select=c('Operating.System','Discount'))

# 2) 정규성 검정
sum(copy_df$Operating.System=='Tizen')
shapiro.test(d.df[d.df$Operating.System=='HomeOS',2])

# 3) 등분산 검정
var.test(d.df[d.df$Operating.System=='Tizen',2], d.df[d.df$Operating.System=='HomeOS',2])

# 4) 등분산가정 독립평균 t 검정
t.test(Discount ~ Operating.System, data=d.df, var.equal=TRUE)

# ( 결론 ) OS 'Tizen'과 "HomeOS"의 평균 Discount Rate에는 유의미한 차이가 있다.




##################
#   통계검정(4)  #
##################

# (대립가설) 같은 Ultra HD LED의 Selling.Price 평균이 “SAMSUNG”, “Panasonic"끼리 다르다.

# 1) 통계 분석을 위한 데이터프레임 만들기
s.df =data.frame()
s.df=rbind(s.df, subset(copy_df, subset=(Resolution=='Ultra HD LED' & Brand=='Panasonic'),select = c('Brand','Selling.Price')))
s.df = rbind(s.df, subset(copy_df, subset=(Resolution=='Ultra HD LED' & Brand=='SAMSUNG'),select = c('Brand','Selling.Price')))

# 2) 정규성 검정
nrow(subset(copy_df, subset=(Resolution=='Ultra HD LED' & Brand=='SAMSUNG')))
shapiro.test(s.df[s.df$Brand=='Panasonic',2])

# 3) 등분산 검정
var.test(s.df[s.df$Brand=='SAMSUNG',2],s.df[s.df$Brand=='Panasonic',2])

# 4) 이분산가정 독립평균 t검정 --> Welch's t-test
t.test(Selling.Price ~ Brand, data=s.df,var.equl=FALSE)

# ( 결론 ) “SAMSUNG”, “Panasonic”별 같은 Ultra HD LED의  평균 가격이 유의미한 차이가 있다.






##################
#   통계검정(5)  #
##################

# (대립가설) 해상도 ‘QLED Ultra HD’,’Ultra HD LED’,’Full HD LED’의 평균 Selling Price에는 차이가 있다.

# 1) 통계 분석을 위한 데이터프레임 만들기
anova.df=subset(copy_df, subset=(Resolution %in% c('QLED Ultra HD','Ultra HD LED','Full HD LED')), select=c('Resolution','Selling.Price'))

# 2) 정규성 검정
sum(copy_df$Resolution=='QLED Ultra HD')
sum(copy_df$Resolution=='Ultra HD LED')
sum(copy_df$Resolution=='Full HD LED')

# 3) 등분산 검정
library(lawstat)
levene.test(anova.df$Selling.Price, anova.df$Resolution)

# 4) 이분산가정 독립평균 one-way ANOVA 검정 --> Welch’s ANOVA
oneway.test(Selling.Price ~ Resolution, data=anova.df, var.equal = FALSE)

# ( 결론 )  해상도 ‘QLED Ultra HD’,’Ultra HD LED’,’Full HD LED’의 평균 Selling Price에는 유의미한 차이가 있다.(모두 같지는 않다)

