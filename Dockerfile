FROM python:3.13-slim
WORKDIR /app
RUN pip install --no-cache-dir --timeout=100 torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir --timeout=100 -r requirements.txt

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY ml/ ./ml
RUN cd ml && python generate_dataset.py && python train_classifier.py && python train_regressor.py

COPY app/ ./app
COPY static/ ./static

ENV MLFLOW_TRACKING_URI=sqlite:///ml/mlflow.db
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
