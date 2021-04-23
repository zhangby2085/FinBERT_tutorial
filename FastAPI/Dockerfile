FROM continuumio/miniconda

WORKDIR /app

COPY apienv.yml main.py suomi24_5_topic_model_FinBERT_finnish suomi24_5_topic_model_FinBERT_finnish.preproc ./

RUN conda env create -f apienv.yml

SHELL ["conda", "run", "-n", "BertFastApiEnv", "/bin/bash", "-c"]

# Make sure the environment is activated:
RUN echo "Make sure fastapi is installed:"
RUN python -c "import fastapi"

EXPOSE 80

# The code to run when container is started:

ENTRYPOINT ["conda", "run", "-n", "BertFastApiEnv", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
