# Heroku 배포하기

![Untitled](/images/Heroku/Untitled.png)

1. Create new app을 통해 이 화면이 뜨게 만든다.
2. [The Heroku CLI | Heroku Dev Center](https://devcenter.heroku.com/articles/heroku-cli#install-the-heroku-cli) 여기서 windows 64비트를 다운 받는다
3. 깃 허브에 들어가서 Repository name 은 heroku-human-seung79 이런식으로 해주고

Add .gitignore는 python으로 해준다.

![Untitled](/images/Heroku/Untitled 1.png)

1. code . 으로 VS code 열고
2. 가상환경 설정(virtualenv venv)⇒ 가상 환경 만들기

![Untitled](/images/Heroku/Untitled 2.png)

1. 가상환경 들어가기 source venv/Scripts/activate

![Untitled](/images/Heroku/Untitled 3.png)

1. pip freeze > requirements.txt 

7-1. pip install FLASK 

7-2. pip install gunicorn

![Untitled](/images/Heroku/Untitled 4.png)

1. export FLASK_APP = index

![Untitled](/images/Heroku/Untitled 5.png)

1. export FLASK_ENV=development

1. export FLASK_App=app

![Untitled](/images/Heroku/Untitled 6.png)

1. 여기까지 했다면 flask run을 통해 실행이 되는 지를 확인한다. 

![Untitled](/images/Heroku/Untitled 7.png)

1. 확인 이후 python -V를 통해 파이썬의 버전을 확인하고(대문자 V이다.)
2. heroku login을 입력하여 heroku에 로그인을 한다.
3. 그리고 이전에 깃허브 리포지토리 이름과 같게 heroku create heroku-human-seung79를 입력해준다.
4. git add .
5. git commit -m “update”
6. git push 까지 해준 다음
7. git push heroku main을 입력한다.