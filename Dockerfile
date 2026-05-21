FROM python:3.12-slim

# System libraries needed by FAISS (OpenMP), scipy, and chromadb
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    gcc \
    g++ \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first (cached layer — only re-runs when requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY agents/       agents/
COPY config/       config/
COPY integrations/ integrations/
COPY mcp_server/   mcp_server/
COPY ui/           ui/
COPY utils/        utils/
COPY knowledge_base/financial_knowledge_base.json knowledge_base/
COPY main.py       .
COPY pyproject.toml .

# Streamlit port
EXPOSE 8501

# Disable Streamlit's browser-open behaviour and bind to all interfaces
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    FA_LOG_LEVEL=INFO \
    PYTHONUNBUFFERED=1

CMD ["python", "-m", "streamlit", "run", "ui/app.py"]
