import java.util.Scanner;


public class Main {
		
		public static void main(String[] args) {
		// TODO Auto-generated method stub
		Scanner sc=new Scanner(System.in);
		int input=sc.nextInt();
		
		boolean arr[]=new boolean[input+1];
		//일단 다 ~~ false로 채워져있다. 
		
		for(int i=2;i<=input;i++) {//1은 제외하고 검사시작
			for(int j=2;j*j<=i;j++) {
				if(i%j==0) {//나눠지면
					arr[i]=true;//소수다
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
