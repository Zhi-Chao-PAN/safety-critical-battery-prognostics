# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Set environment variable to make sure python outputs to console
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Run the pipeline when the container launches
# Or keep it interactive
CMD ["bash"]
