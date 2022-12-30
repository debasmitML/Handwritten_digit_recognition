FROM python:3.8.0

WORKDIR /usr/src/app
COPY handwritten_digit_classification_using_ML.ipynb  .
COPY requirements.txt  .
RUN pip install -r requirements.txt  .
CMD["python","/handwritten_digit_classification_using_ML.ipynb"]
