FROM ubuntu:22.04

# Installer Python
RUN apt-get -y update && \
    apt-get install -y python3-pip

# Copier requirements.txt et installer les dépendances
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copier tous les fichiers du projet dans l'image Docker
COPY . /app

# Définir le répertoire de travail à /app
WORKDIR /app

# Exécuter main.py
CMD ["python3", "main.py"]