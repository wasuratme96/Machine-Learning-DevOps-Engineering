FROM python:3.9.6-slim-buster

COPY ./ /churn_library
WORKDIR /churn_library

RUN pip install -r requirements.txt

VOLUME /churn_library/logs
VOLUME /churn_library/images
VOLUME /churn_library/models
VOLUME /churn_library/data/data profile

RUN mkdir -p /churn_library/data/data profile && \
    mkdir -p /churn_library/logs && \
    mkdir -p /churn_library/models && \
    mkdir -p /churn_library/images/eda && \
    mkdir -p /churn_library/images/results

ENTRYPOINT ["python"]
CMD ["churn_script_logging_and_tests.py"]