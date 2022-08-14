import java.util.Scanner;
class Stack{
	int[] data;
	int top;//top most index of stack
	
	// 스택에 아무 것도 없잖아 초기화했으니까. 그럼 탑 값은 뭐넣어
	// null값은 넣을 수 없습니다. 왜냐하면 int니까요 Integer()면 넣을 수 있슴. 
	//-1이 가장 적당하다. 데이터 넣을 때마다 top++할 것이기 때문이다.
	
	Stack(){
		top=-1;//초기화
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
