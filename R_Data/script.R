#CH_1 , CH_2

library(ggplot2)
#ggplot2 ��Ű�� ���� ������
head(mpg)
#�� �� �ϳ��ϳ��� ������ �ȴ�.

mean(mpg$hwy)
max(mpg$hwy)
min(mpg$hwy)
#������׷��׸���
hist(mpg$hwy)

#���������
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

#�� �׷��� �����
b = c('a','a','b','c')
qplot(b)

#��ĭ �����ڷ� �̾���̱�
paste(e2, collapse = ' ')

#�Լ� ����Ϸ��� ��Ű�� ��ġ �� �ε��ؾ��Ѵ�.
#��Ű���� �Լ� �ٷ���
#�����Լ��� ���� �ȱ׷����ȴ�
#ex) ggplot2 = �ð�ȭ ��Ű��
# qplot(), geom_histogram(), geom_line()
mpg
qplot(data = mpg, x = hwy)
qplot(data=mpg, y =hwy, x = drv,geom= 'point')
#geom�� �׷����� ����� �����Ѵ�.
qplot(data=mpg, y =hwy, x = drv,geom= 'boxplot')
qplot(data=mpg, y =hwy, x = drv,geom= 'boxplot', color = drv)
#drv ���� ���� �ٸ���

#�Լ��� ������ �ñ��ϸ�
?qplot
#�Ŵ��� ���� �����°�
qplot(mpg, wt, data =mtcars)
qplot(mpg, wt, data =mtcars, color = cyl)
qplot(mpg, wt, data =mtcars, size =cyl)
qplot(mpg, wt, data =mtcars, facets =vs ~am)
#���־��� �Լ� �������������� ggplot2 cheat sheet �ϸ��.

#������������ �����
history <- c(90, 80,60,70)
math <- c(50,60,100,20)

#���� ���ؼ� ������������ �����
mf_midterm <- data.frame(history, math)
mf_midterm

#�� �߰��ϱ�
class <- c(1,1,2,2)
df_midterm <- data.frame(mf_midterm, class)
mean(df_midterm$history)
mean(df_midterm$math)

setwd('C:/Users/pc/Desktop/�н�Ʈķ�۽�_�����ͺм�/part-1,2,3-�����ڷ�-�迵�찭���/�����ڷ�_�迵�찭���/����/Data')

#���� ������ �ҷ�����
install.packages('readxl')
library(readxl)

#���� �ҷ�����
df_finalexam = read_excel('finalexam.xlsx', sheet =1, col_names =T)
df_finalexam
mean(df_finalexam$math)

#csv ���� �ҷ�����
#�޸������� ����� ,�� ���еǾ������� �� �� ����! 
#�޸� seperate value #csv�� ���� ������ ����!
#������ ������ �����̾ �뷮�� ���� ������ �ε� �ӵ��� ����

#csv ������ ��Ʈ�� ������ ����
read.csv('csv_exam.csv', header=T)
#csv�� �����ϱ�
write.csv(df_finalexam, file='output_newdata.csv')


#�������� Ư�� �ľ��ϱ�
exam = read.csv('csv_exam.csv')
exam
head(exam) #default = 6��
head(exam, 10)
tail(exam, 10)
#��� â���� ������
View(exam)
dim(exam)
str(exam)
#�����跮 ���
summary(exam)

#ggplot2�� mpg �����͸� ������ ������ ���·� �ҷ�����
#��� �ϴ� mpg �����͸� �� �̾ƿ��δ�.ggplot2::mpg
mpg <- as.data.frame(ggplot2::mpg)
mpg
head(mpg)

#�������� ������ �ٲٱ�
install.packages('dplyr')
library(dplyr)

df_raw = data.frame(var1 = c(1,2,1),
                    var2 = c(2,3,2))
df_raw

#������ ���� ���� ������ ��������������ƾ��Ѵ�.
df_new = df_raw
#������ �ٲٱ�
df_new = rename(df_new, v2 = var2 ) #var2�� v2�� ����
#�Ҵ� ���ϸ� df_new�� ������ �ȵ�! �Ҵ� �������ؾ��� �������� ����
df_new

df_raw = as.data.frame(ggplot2::mpg)
head(df_raw)
df_new = df_raw
head(df_new)
df_new = rename(df_new, city = cty, highway = hwy)
head(df_new)

#�Ļ� ���� �����
df=data.frame(var1=c(4,3,8),
              var2= c(2,6,1))

df$var_sum = df$var1 + df$var2
df$var_mean = (df$var1 + df$var2) /2

#mpg ���� ���� ���� ���帯
mpg$total = (mpg$cty + mpg$hwy)/2
mean(mpg$total)

#���ǹ� �׿��� �Ļ����� �����
summary(mpg$total)
hist(mpg$total)
#20�� �������� ����. 20�̻��̸� �հ�
mpg$test = ifelse(mpg$total>=20,'pass','fail')
head(mpg)

#��ǥ�� �հ� ���� �ڵ��� �� ���캸��
table(mpg$test)

#�׷����� ����
library(ggplot2)
qplot(mpg$test)

#��ø ���ǹ�Ȱ���ؼ� ���� ��� ���� �����
mpg$grade = ifelse(mpg$total>=30, 'A', ifelse(mpg$total >=20, 'B','C'))
head(mpg)                   
table(mpg$grad)
qplot(mpg$grade)

#���� 1. ggplot2�� midwest �����͸� ���� ���·� �ҷ��ͼ� Ư���ľ�
df_raw = as.data.frame(ggplot2::midwest)
str(df_raw)
summary(df_raw)
head(df_raw)

#����2 poptotal��ü�α��� total��, popasian �ƽþ��α��� asian���� ����
df_new =df_raw
df_new = rename(df_new, asian=popasian, total = poptotal)
head(df_new)

#total�� asian �̿��ؼ� '��ü�α���� �ƽþ� �α� �����'�����
#������׷��� ����� ���õ��� ��� �����ϴ��� ���캸����
df_new$prop = (df_new$asian / df_new$total) * 100
hist(df_new$prop)

#�ƽþ� �α� ����� ��ü ����� ���ϰ�large, small
df_new$asia=ifelse(df_new$prop >= mean(df_new$prop), 'large','small')
table(df_new$asia)
qplot(df_new$asia)

#������ ��ó��
#1) ���ǿ� �´� �����͸� �����ϱ�
exam=read.csv('csv_exam.csv')

#exam���� class�� 1�� ��츸 �����ؼ� ���
exam %>% filter(class ==1)
#����Ű ctrl+shift+m���� %>% ��ȣ �Է� ����

#1���� �ƴѰ��
exam %>%  filter(class !=1)

#�ʰ� �̸� �̻����� ���� �ɱ�
exam %>% filter(math > 50)

#���� ������ �����ϴ� �� ���߶��
exam %>% filter(class ==1 &math >= 50)
exam %>% filter(math >=90 | english >= 90)

#��Ͽ� �ش�Ǵ� �� �����ϱ�
exam %>% filter(class %in% c(1,3,5))

#������ ������ ������ �����
class1 = exam %>% filter(class ==1)
mean(class1$math)

#�ڵ��� ��ⷮ�� ���� ���ӵ��� ���� �ٸ��� �˾ƺ����� �Ѵ�.
#disp1�� 4 ������ �ڵ����� 5�̻��� ���ڵ��� �� � �ڵ�����hwy�� ��������� �� ������ �˾ƺ�����
df_raw = as.data.frame(ggplot2::mpg)
df_new = df_raw

df_filter1=df_new %>% filter(displ<=4)
df_filter2= df_new %>% filter(displ>=5)
mean(df_filter1$hwy)
mean(df_filter2$hwy)

#�ڵ����� ����ȸ�翡 ���� ���� �ٸ��� �˾ƺ���
df_filter3 = df_new %>% filter(manufacturer=='audi')
df_filter4= df_new %>% filter(manufacturer=='toyota')
mean(df_filter3$cty)
mean(df_filter4$cty)

#chevrolet��ford, honda�� �ڵ����� ���ӵ��� ���� ���
#hwy ��ü ��� ���� ��
df_filter5= df_new %>% filter(manufacturer=='chevrolet'|manufacturer=='ford'|manufacturer=='honda')
mean(df_filter5$hwy)

#�ʿ��� ������ �����ϱ�
exam %>% select(math)
exam %>% select(math, english)
exam %>% select(-math)
exam %>% select(-math, -english)

#dplyr �Լ� �����ϱ�
exam %>% filter(class ==1) %>% select(english)
# %>% �� ������ �Լ�!

exam %>% select(id, math) %>% head
exam %>% select(id, math) %>% head(10)

#mpg �����ʹ� 11���� ������ �Ǿ��ֽ��ϴ�. ���� �Ϻθ� �����ؼ� �м��� Ȱ���� ��
#mpg���� class cty ������ ���� ������ �����
select1=df_new %>% select(class, cty)
head(select1)
#�������� ������ �̿��ؼ� cty�� �� ������ Ȯ���ϱ�
select_filt = select1 %>% filter(class=='suv') %>% select(cty)
select_filt2 =select1 %>% filter(class=='compact') %>% select(cty)
mean(select_filt$cty)
mean(select_filt2$cty)


#�����͸� �����ϱ�
exam %>% arrange(math)
exam %>% arrange(desc(math))
exam %>% arrange(class, math)

#�ƿ�𿡼� ������ �ڵ��� �� � �ڵ����� ���� hwy�� ������ 1~5��
df_new
df_new %>% filter(manufacturer=='audi') %>% arrange(desc(hwy)) %>% head(5)

#�Ļ����� �߰��ϱ�
exam %>% mutate(total= math +english+science) %>% head
#�ξ� �������� �ڵ�
exam %>% mutate(total = math+english+science, 
                mean =(math+english+science)/3) %>% head
exam %>% mutate(test =ifelse(science >= 60, 'pass','fail')) %>% head
#�Ҵ���ص� ������ ���� ����
exam %>% mutate(total= math +english+science) %>% arrange(total)

#mpg ���纻�����cty+hwy ���� �ջ꿬�� ���� �߰�
df_new=df_new %>% mutate(sum=cty+hwy)
head(df_new)
df_new = df_new %>% mutate(m = sum/2)
df_new %>% arrange(desc(m)) %>% head(3)
df_raw %>% mutate(sum  = cty+ hwy, m = sum/2) %>% arrange(desc(m)) %>% head(3)

#���ܺ��� ������ ����ϱ�

#������
#group_by �Լ��� �����ؼ� ����.
exam %>% summarise(mean_math=mean(math))
#���ÿ� ������ ��� ��跮 ���� �� ����
#n()�� �� ���� ���� ���� ��
exam %>% group_by(class) %>%  summarise(mean_math = mean(math))
exam %>% group_by(class) %>%  summarise(mean_math = mean(math),
                                        sum_math = sum(math),
                                        median_math = median(math),
                                        n=n()) #�л���

#�� ���ܺ��� �ٽ� ���� ������
#�����纰�� ������ drv���� �� ����
mpg %>% group_by(manufacturer, drv) %>% summarise(mean_cty=mean(cty)) %>% head(10)

#mpg � ������ ���� ������
mpg %>% group_by(class) %>% summarise(mena_cty =mean(cty))

#���� ���ĺ� ������ ���ĵǾ��µ� �̸� cty ��ռ� ����
mpg %>% group_by(class) %>% summarise(mena_cty =mean(cty)) %>% arrange(desc(mena_cty))

#� ȸ���� hwy�� ���� ������ ����� ���� ���� ȸ�� �� �� ���
mpg %>% group_by(manufacturer) %>% summarise(mean_hwy=mean(hwy)) %>% arrange(desc(mean_hwy)) %>% head(3)

#�ȸ�翡�� compact ������ ���� ���� �����ϴ��� 
mpg %>% group_by(manufacturer) %>% filter(class=='compact') %>% summarise(n=n())


#������ ��ġ��
test1 = data.frame(id = c(1,2,3,4,5),
                   midterm = c(60,80,70,90,85))
test2= data.frame(id=c(1,2,3,4,5),
                 final=c(70,80,65,95,80))
total = left_join(test1, test2, by='id') 
#id ���� ���� "" �ٿ�����
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

#���η� ��ġ��
d1 = data.frame( id = c(1:5), test=c(seq(70,100, length = 5)))
d1
d2=data.frame(id=c(6:10), test= c(seq(80,100, length = 5)))
group_all = bind_rows(d1,d2)
#���⼱ by�� ����
group_all

#mpg������ �̿��ؼ� �м� ���� �ذ��ϱ�
#f1������ ���Ḧ �ǹ�
#����� �������� ������ ������ ������ �����
fuel = data.frame(fl= c('c','d','e','p','r'),
                  price_fl= c(2.35, 2.38, 2.11, 2.76, 2.22),
                  stringsAsFactors = F)
#stringsAsFactors factorŸ���� ���� ���ڿ���,�� �׷��� ���� ����
#factor�� ��ȯ�ϸ� �м��� �� ������ ���� ���� �� �ִٿ�
fuel

#mpg���� ���� ���� ��Ÿ���� fl������ ������ ������ ����
#mpg �����Ϳ�price_fl ���� �߰��ϱ�
head(mpg)
mpg=left_join(mpg, fuel, by='fl')
mpg %>% select(model, fl, price_fl) %>% head(5)


#�м�����

df_raw =as.data.frame(ggplot2::midwest)
df_new =df_raw

#��ü �α� ��� �̼��� �α� ����� ����
df_new = df_new %>% mutate(df_new, t = popadults / poptotal*100)
df_new
#�̼��� �α� ������� ���� ���� ���� �ټ��� ���� �̼����α� ��з� ���
df_new %>% arrange(desc(t)) %>% select(t) %>% head(5)
#�̼��� ������� ���� �߰�. �� ��޿� �� ���� ������ �ִ��� �˾ƺ���
df_new = df_new %>% mutate (df_new, class = ifelse(t>=40, 'large', ifelse(t>=30,'middle','small')))
table(df_new$class)

#��ü�α� ��� �ƽþ��� �α� ����� ���� �߰��ϰ� ���� 10�� ������ �� ������ ����� ����
df_new %>% mutate(df_new, t2= popasian/poptotal * 100) %>% arrange(desc(t2)) %>% select(state, county, t2) %>% tail(10)