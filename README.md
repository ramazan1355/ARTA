# ARTA

#Клонируем репозиторий:

git clone <your-repo-url>
cd armeta_document_ai


#Устанавливаем зависимости:

pip install -r requirements.txt


#Запускаем сервер:

uvicorn armeta_app:app --reload --port 8000


#Открываем браузер:
http://localhost:8000
