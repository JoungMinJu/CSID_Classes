import java.util.Scanner;
class Stack{
	int[] data;
	int top;//top most index of stack
	
	// ���ÿ� �ƹ� �͵� ���ݾ� �ʱ�ȭ�����ϱ�. �׷� ž ���� ���־�
	// null���� ���� �� �����ϴ�. �ֳ��ϸ� int�ϱ�� Integer()�� ���� �� �ֽ�. 
	//-1�� ���� �����ϴ�. ������ ���� ������ top++�� ���̱� �����̴�.
	
	Stack(){
		top=-1;//�ʱ�ȭ
		data=new int[100];
		}//y
	void push(int e) {
		top++;
		data[top]=e;
	}
	void pop() {
		top--;
	}
	int top() {
		return data[top];
	}
	
	boolean empty() {
		return top==-1;
	}
}
public class Main {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		Scanner sc=new Scanner(System.in);
		Stack stack=new Stack();
		for(int i=0;i<10;i++) {
			stack.push(sc.nextInt());
		}
		while(stack.empty()==false) {
			System.out.println(stack.top());
			stack.pop();
		}
		sc.close();

	}
	}
