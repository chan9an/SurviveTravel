FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

# Run uvicorn pointing to the new server directory
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
