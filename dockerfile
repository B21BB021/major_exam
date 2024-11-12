# Use a suitable Python base image for ML applications
FROM python:3.9-slim

# Set environment variables for Python
ENV PYTHONUNBUFFERED=1
ENV PYTHONHASHSEED=0

# Create a non-root user for security
RUN useradd -m mluser

# Set working directory and change ownership to the non-root user
WORKDIR /app
RUN chown mluser:mluser /app

# Copy requirements and install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Install and set up MLflow
RUN pip install mlflow

# Switch to the non-root user
USER mluser

# Copy application code into the container
COPY . /app/

# Expose ports (5000 for MLflow server and 8000 for the application if needed)
EXPOSE 5000 8000

# Command to run the application (can be replaced by entrypoint script)
CMD ["python", "main.py"]
