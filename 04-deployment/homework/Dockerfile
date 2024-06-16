FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim


RUN pip install -U pip
RUN pip install pipenv 

WORKDIR /app

COPY [ "Pipfile", "Pipfile.lock", "./" ]

RUN pipenv install --system --deploy

COPY [ "hw_predict.py", "./" ]

ENTRYPOINT [ "python3", "hw_predict.py", "--year=2023", "--month=05" ]
