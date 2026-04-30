FROM python:3.10-slim

# Install system dependencies (git, ssh, procps for psutil/ps commands)
RUN apt-get update && apt-get install -y procps git openssh-client && rm -rf /var/lib/apt/lists/*

# Set up the expected python path for hardcoded scripts in api.py
RUN mkdir -p /home/ubuntu/anaconda3/bin && \
    ln -s /usr/local/bin/python /home/ubuntu/anaconda3/bin/python

# Create working directory
WORKDIR /home/ubuntu/nevir

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Add to python path
ENV PYTHONPATH=/home/ubuntu/nevir

# We will mount the application code using docker-compose
CMD ["python", "api/api.py"]
