# Use a slim Python 3.11 image as base
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1

# Install uv for fast dependency management
RUN pip install uv

# Set the working directory
WORKDIR /app

# Copy the pyproject.toml to install dependencies
COPY pyproject.toml .
# COPY uv.lock . # Uncomment after generating uv.lock

# Install dependencies using uv
RUN uv pip install -r pyproject.toml --system

# Copy the rest of the application code
COPY . .

# Expose the API port
EXPOSE 8000

# Default command to run the FastAPI app
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
