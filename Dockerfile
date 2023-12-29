FROM pytorch/pytorch:latest


RUN apt-get update && \
    apt-get install -y graphviz-dev python3-dev && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/* && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    python -c 'import sys; assert sys.version_info[:2] == (3, 10)'
	
RUN apt-get update && apt-get -y install python3-pydot python3-pydot-ng graphviz
WORKDIR /app
COPY causica causica
COPY task.py task.py
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
ENTRYPOINT ["python", "task.py"]
