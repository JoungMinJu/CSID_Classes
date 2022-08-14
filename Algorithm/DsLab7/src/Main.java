import java.util.Scanner;

//���Ḯ��Ʈ�� ����Ͽ�
class Node{
	public int data;
	public Node next;//������ � ������� ����Ű��!
	Node(int e){data=e; next=null;}
	}
class Stack{
	Node head;
	Stack(){
		head=null;//�ʱ�ȭ
		}
	void push(int e) {
		Node node=new Node(e);
		node.next=head;//���� head�� ����Ű�� ��带 ���� ���� �����next�� ����.
		head=node;
	}//���ÿ��� ���� ���� �ִ� ���� �ٷ� head�� ����Ű�� ���̴�.
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

