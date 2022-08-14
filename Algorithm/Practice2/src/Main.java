//��ũ�帮��Ʈ �̿��� ���� ����
import java.util.Scanner;
class Node<K>{
	Node next;
	K data;
	Node(K k){
		next=null;
		data=k;
	}
}
class Stack<K>{
	Node head;
	Stack(){
		head=null;
	}
	void push(K k) {
		Node node=new Node(k);
		node.next=head;
		head=node;
	}
	void pop() {
		head=head.next;
	}
	K peek() {
		return (K)head.data;
	}
	boolean empty() {
		return head==null;
	}
	void show() {
		System.out.print("top--> ");
		Node value=head;
		while(value!=null) {
			System.out.print(value.data+"\t");
			value=value.next;
		}
	System.out.println();	
	}
}


public class Main {
	static Stack<Character> stack;
	//1��ȯ ����
	//-100000000 ��ȯ 1������
	//-200000000 ��ȯ 2�� ����
	//-300000000 ��ȯ 3�� ����
	
	static int check(String s) {
		stack=new Stack();
		
		for(int i=0; i<s.length();i++) {
			char c=s.charAt(i);
			if(c=='{'||c=='['||c=='(') {
				stack.push(c);
			}
			else if(c=='}'||c==']'||c==')') {
				if(stack.empty()) { return -200000000;}
				char v=stack.peek();
				stack.pop();
				if(c=='}'&&v!='{') return -100000000;
				if(c==']'&&v!='[') return -100000000;
				if(c==')'&&v!='(') return -100000000;
				
			}
		}
		if(stack.empty()) return 1;
		return -300000000;
	}
	
	static void checkResult(String s) {
		int result=check(s);
		if(result==-100000000) System.out.println("1������ ����");
		else if(result==-200000000) System.out.println("2������ ����");
		else if(result==-300000000) System.out.println("3������ ����");
		else if(result==1) System.out.println("����");
	}

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		String formula1="{((2+3)/(3-1)}+{3*(5-4)}";
		checkResult(formula1);
		String formula2="{(2+3)/(3-1)})+{3*(5-4)}";
		checkResult(formula2);
		String formula3="{({(2+3)/(3-1)}+{3*(5-4)}";
		checkResult(formula3);
		String formula4="{()()[()]}";
		checkResult(formula4);
		
	}

}
