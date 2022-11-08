from flask import Flask, render_template, request
app = Flask(__name__)

@app.route('/')  # default URL
def student():
   return render_template('input_info.html')

@app.route('/result', methods = ['POST', 'GET'])
def result():
   if request.method == 'POST':
      result = dict()
      result['Name'] = request.form.get('Name')

      # 학번
      result['StudentNumber'] = request.form.get('StudentNumber')
      # 성별
      result['Gender'] = request.form.get('Gender')
      # 학과
      # 프로그래밍 언어 -> hint) ','.join(list명)을 사용하면 list 안에 있는 항목들이 ','로 나누어져 출력됨.
      result_list = []
      language_list = ['Python','Java', 'HTML', 'C++']
      for i in language_list:
         if request.form.get(i):
            result_list.append(i)

      result['checkbox'] = ','.join(result_list)

      return render_template("result.html",result = result)

if __name__ == '__main__':
   app.run(host="0.0.0.0", port=5000)
