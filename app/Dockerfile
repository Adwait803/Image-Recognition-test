# Dockerfile for both Frontend (Streamlit) and Backend (FastAPI)

# Use a base image with Python and other dependencies
FROM python:3.8-slim-buster

# Install supervisord
RUN apt-get update && apt-get install -y supervisor && apt-get clean

# Create a directory for logs
RUN mkdir -p /var/log/supervisor

# Set the working directory
WORKDIR /app

# Copy the Streamlit and FastAPI app files into the container
COPY . .

# Install any necessary dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirement.txt

# Expose the ports for both Streamlit and FastAPI
EXPOSE 8501 8000

# Copy the supervisord configuration file
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Start supervisord to manage both Streamlit and FastAPI
CMD ["supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
