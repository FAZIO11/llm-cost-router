FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY llm_cost_router/ ./llm_cost_router/
COPY data/ ./data/
COPY models/ ./models/
COPY .env* ./

# Create runtime directories
RUN mkdir -p logs models

# Expose port
EXPOSE 8000

# Run server
CMD ["python", "-m", "uvicorn", "llm_cost_router.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
