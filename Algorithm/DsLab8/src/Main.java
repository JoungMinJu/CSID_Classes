import java.util.Scanner;


public class Main {
		
		public static void main(String[] args) {
		// TODO Auto-generated method stub
		Scanner sc=new Scanner(System.in);
		int input=sc.nextInt();
		
		boolean arr[]=new boolean[input+1];
		//�ϴ� �� ~~ false�� ä�����ִ�. 
		
		for(int i=2;i<=input;i++) {//1�� �����ϰ� �˻����
			for(int j=2;j*j<=i;j++) {
				if(i%j==0) {//��������
					arr[i]=true;//�Ҽ���
					break;
				}
				
			}
		}
		int count=0;
		for(int i=2;i<=input;i++) {
			if(!arr[i]) {
				count++;
			}
		}
		System.out.println(count);
	
		}
}
