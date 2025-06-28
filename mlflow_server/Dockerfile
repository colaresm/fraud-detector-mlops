FROM python:3.11-slim

# Instale dependências
RUN pip install mlflow

# Exponha a porta padrão do MLflow (5000)
EXPOSE 5000

# Comando para iniciar o servidor MLflow
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]