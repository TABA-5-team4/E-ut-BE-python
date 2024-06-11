# E-ut-BE-python

## Getting Started

### ec2 배포

```angular2html
$ sudo yum install python3-pip

$ git clone 

$ aws configure

$ aws s3 cp s3://eut-s3-bucket/summarize_model /home/ec2-user/eut-app/E-ut-BE-python/summary_model --recursive

$ aws s3 cp s3://eut-s3-bucket/sentiment_analysis.pt /home/ec2-user/eut-app/E-ut-BE-python

$ ./python_deploy.sh


```

```angular2html
$ python -m venv venv
$ source venv/bin/activate

$ pip install -r requirements.txt

$ uvicorn main:app --reload

kill -9 $(pgrep -f python)
```

### API KEY
```angular2html
$ export OPENAI_API_KEY=your_api_key

또는

파이참 설정에서 환경변수 추가
```