class LinearProbing<K,V>{
   private int M; //�ؽ� ���̺��� ũ��
   private K[] a; // key ���� ���� ����
   private V[] d; // data ���� ���� ����
   
    //������
    //���̺��� ũ�⸦ ����
   public LinearProbing(int m) {
      M = m;
      a = (K[]) new Object[M]; // key�� ���� list ����
      d = (V[]) new Object[M]; // value�� ���� list ����
   }
   
   //hash �Լ� �������ֱ�
   private int hash(K key) {
      // & 0x7fffffff -> ������ ��� ����� �ٲ��ִ� �۾�
      return (key.hashCode() & 0x7fffffff) % M; // �ؽ����� return
   }
   
   //key���� ���� data�� �� �ؽð��� ����
   public void put(K key, V data) {
      int initialpos = hash(key); // �ؽð� ����
      int i=initialpos, j=1;
      do {
         if(a[i] == null) { // ���� ����ִ� �����̸�
            a[i] = key;      // ó�� ��ġ�� key ���� �ְ�
            d[i] = data;   // data�� �Բ� �־��ش�.
            break;         
         }
         i = (initialpos + j++) % M; //�̹� data�� �ִٸ� ���ο� ��ġ�� ã���ֱ�(���⼭�� ����ĭ),
                               // �ٸ� ������ �̹� ������ �ٽ� ���ο� �� ã���ֱ�
      } while(i != initialpos); // ���ǽ��� false�� ��� do - while�� ����

   }
   
   public void hashtable() {
      System.out.println("�ؽ����̺�");
      // �ؽ����̺��� �ؽð� ���
      for(int i = 0; i<M; i++) {
         System.out.print(i + "\t");
      }
      System.out.println();
      
      // Ű ���� ��� �ؽð��� ������ �Ǿ����� �����ֱ�
      for(int i = 0; i<M; i++) {
         System.out.print(a[i] + "\t");
      }
      System.out.println();
      
      //��� data�� ������ Ȯ�� �ϱ�
      for(int i =0; i<M; i++) {
         System.out.print(d[i]+"\t");
      }
      System.out.println();
       }}

   public class Main {
      public static void main(String[] args) {
      System.out.println("��������");
      //���� �ϼ���
      //(key.hashCode() & 0x7fffffff) % M ��� �Լ��� ���� ���
      System.out.println();    
        System.out.println("�ؽ��Լ�(kety) : �ؽð�");
        System.out.println("hash(71) :"+(71 & 0x7fffffff) % 10);
        System.out.println("hash(23) :"+(23 & 0x7fffffff) % 10);
        System.out.println("hash(73) :"+(73 & 0x7fffffff) % 10);
        System.out.println("hash(49) :"+(49 & 0x7fffffff) % 10);
        System.out.println("hash(54) :"+(54 & 0x7fffffff) % 10);
        System.out.println("hash(89) :"+(89 & 0x7fffffff) % 10);
        System.out.println("hash(39) :"+(39 & 0x7fffffff) % 10);
        //����
        LinearProbing<Integer, String> fruit = new LinearProbing<Integer, String>(10);
        fruit.put(71, "grape");   // key, data
        fruit.put(23, "apple");
        fruit.put(73, "banana");
        fruit.put(49, "cherry");
        fruit.put(54, "mango");
        fruit.put(89, "lime");
        fruit.put(39, "ornage");
        fruit.hashtable();

   }
}

