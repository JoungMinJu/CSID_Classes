import java.util.Scanner;

//연결리스트로 만들겅요
class Node{
	public int data;
	public Node next;//다음이 어떤 노드인지 가리키는!
	Node(int e){data=e; next=null;}
	}
class Stack{
	Node head;
	Stack(){
		head=null;//초기화
		}
	void push(int e) {
		Node node=new Node(e);
		node.next=head;//현재 head가 가리키는 노드를 새로 만든 노드의next로 저장.
		head=node;
	}//스택에서 제일 위에 있는 값이 바로 head가 가리키는 값이다.
	void pop() {
		head=head.next;
	}
	int top() {
		return head.data;
	}
	
	boolean empty() {
		return head==null;
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

