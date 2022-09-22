import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

public class Main {
    public static void main(String[] args) throws IOException {
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));

        System.out.print("이름을 입력하세요: ");
        String name = br.readLine();

        System.out.print("학번을 입력하세요: ");
        String student_id = br.readLine();

        System.out.print("학과를 입력하세요: ");
        String department = br.readLine();

        System.out.println();

        System.out.println("<출력>");
        System.out.println("이름: " + name);
        System.out.println("학번: " + student_id);
        System.out.println("학과: " + department);
    }
}