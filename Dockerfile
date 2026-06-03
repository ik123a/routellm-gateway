FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose gateway port
EXPOSE 8000

# Start the gateway
CMD ["uvicorn", "gateway.main:app", "--host", "0.0.0.0", "--port", "8000"]
