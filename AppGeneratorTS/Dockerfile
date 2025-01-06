# Utiliser une image Python comme base
FROM python:3.12.0

# Définir le répertoire de travail
WORKDIR /app

# Copier les dépendances
COPY requirements.txt /app/requirements.txt

# Mettre à jour pip et installer les dépendances
RUN python -m pip install --upgrade pip
RUN python -m pip install -r requirements.txt

# Copier le reste des fichiers source
COPY . /app

# Définir la variable d'environnement Flask
ENV FLASK_APP=app.py

# Exposer le port utilisé par Flask
EXPOSE 5000

# Lancer l'application Flask
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0"]
