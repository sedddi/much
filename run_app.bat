@echo off
echo Much (머니치료) - 청년 맞춤형 AI 자산관리 서비스
echo ================================================
echo.

echo 1. 가상환경 생성 중...
python -m venv venv

echo 2. 가상환경 활성화 중...
call venv\Scripts\activate

echo 3. 필요한 패키지 설치 중...
pip install -r requirements.txt

echo 4. 애플리케이션 실행 중...
echo 브라우저에서 http://localhost:8501 을 열어주세요.
echo.
echo 로그인 정보:
echo - 아이디: test_user
echo - 비밀번호: test123
echo.
streamlit run app.py

pause
