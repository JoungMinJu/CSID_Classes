class LinearProbing<K,V>{
   private int M; //해쉬 테이블의 크기
   private K[] a; // key 값이 들어가는 공간
   private V[] d; // data 값이 들어가는 공간
   
    //생성자
    //테이블의 크기를 지정
   public LinearProbing(int m) {
      M = m;
      a = (K[]) new Object[M]; // key에 대한 list 형성
      d = (V[]) new Object[M]; // value에 대한 list 형성
   }
   
   //hash 함수 지정해주기
   private int hash(K key) {
      // & 0x7fffffff -> 음수일 경우 양수로 바꿔주는 작업
      return (key.hashCode() & 0x7fffffff) % M; // 해쉬값을 return
   }
   
   //key값에 따른 data가 들어갈 해시값을 지정
   public void put(K key, V data) {
      int initialpos = hash(key); // 해시값 설정
      int i=initialpos, j=1;
      do {
         if(a[i] == null) { // 만약 비어있는 공간이면
            a[i] = key;      // 처음 위치에 key 값을 주고
            d[i] = data;   // data도 함께 넣어준다.
            break;         
         }
         i = (initialpos + j++) % M; //이미 data가 있다면 새로운 위치를 찾아주기(여기서는 다음칸),
                               // 다른 곳에도 이미 있으면 다시 새로운 곳 찾아주기
      } while(i != initialpos); // 조건식이 false인 경우 do - while문 종료

   }
   
   public void hashtable() {
      System.out.println("해시테이블");
      // 해시테이블의 해시값 출력
      for(int i = 0; i<M; i++) {
         System.out.print(i + "\t");
      }
      System.out.println();
      
      // 키 값이 어느 해시값에 배정이 되었는지 보여주기
      for(int i = 0; i<M; i++) {
         System.out.print(a[i] + "\t");
      }
      System.out.println();
      
      //어떠한 data가 들어갔는지 확인 하기
      for(int i =0; i<M; i++) {
         System.out.print(d[i]+"\t");
      }
      System.out.println();
       }}

   public class Main {
      public static void main(String[] args) {
      System.out.println("선형조사");
      //참고 하세요
      //(key.hashCode() & 0x7fffffff) % M 라는 함수에 따른 결과
      System.out.println();    
        System.out.println("해시함수(kety) : 해시값");
        System.out.println("hash(71) :"+(71 & 0x7fffffff) % 10);
        System.out.println("hash(23) :"+(23 & 0x7fffffff) % 10);
        System.out.println("hash(73) :"+(73 & 0x7fffffff) % 10);
        System.out.println("hash(49) :"+(49 & 0x7fffffff) % 10);
        System.out.println("hash(54) :"+(54 & 0x7fffffff) % 10);
        System.out.println("hash(89) :"+(89 & 0x7fffffff) % 10);
        System.out.println("hash(39) :"+(39 & 0x7fffffff) % 10);
        //참고
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

