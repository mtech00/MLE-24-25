FROM python:3.9-slim

# Working directory for the application
WORKDIR /opt/program

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy inference code
COPY inference.py .

# Environment variables
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

# Create model directory
RUN mkdir -p /opt/ml/model

# Port for SageMaker to communicate
EXPOSE 8080

# Command to run when container starts
ENTRYPOINT ["python", "inference.py"]
