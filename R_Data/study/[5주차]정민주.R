#입력과 출력
setwd('C:/Users/pc/Documents/[1주차]_R프로그래밍스터디_정민주')

#퀴즈 1
#1. NetfilxOriginalx.csv 파일을 열고 평점이 8점 이상인 영화들의 제목을 ㄷ나열
p= read.csv('NetflixOriginals.csv')
head(p)
p$Title[p$IMDB.Score>=8]

#2. 장르가 드라마인 영화의 개수 출력
sum(p$Genre=='Drama')

#3. 가장 평점이 높은 영화 제목
p$Title[p$IMDB.Score == max(p$IMDB.Score)]


#4. 장르가 comedy인 영화목록 따로 구성하여 netflix._comdey.csv로 저장하기
p_comedy = p[p$Genre=='Comedy',]
write.csv(x=p_comedy, file='netflix._comdey.csv')
