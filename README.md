# 📈 API de Prédiction d'Actions

Une API FastAPI qui utilise un modèle de Machine Learning LSTM pour prédire les prix des actions et fournir des données financières en temps réel.

## 🚀 Fonctionnalités

- **Prédiction de prix d'actions** : Modèle LSTM entraîné sur des données historiques
- **API REST** : Endpoints pour récupérer et analyser les données d'actions
- **Données en temps réel** : Intégration avec Yahoo Finance (yfinance)
- **Documentation automatique** : Interface Swagger UI intégrée
- **Dockerisation** : Conteneurisation complète pour le déploiement

## 🛠️ Technologies Utilisées

- **Backend** : FastAPI 0.68.0
- **Machine Learning** : TensorFlow/Keras, Scikit-learn
- **Données** : Pandas, NumPy, yfinance
- **Serveur** : Uvicorn, Gunicorn
- **Conteneurisation** : Docker

## 📋 Prérequis

- Python 3.9+
- Docker (optionnel)

## 🚀 Installation et Démarrage

### Option 1 : Installation Locale avec Environnement Virtuel (Recommandé)

1. **Cloner le repository**

```bash
git clone <repository-url>
cd stocks_predictions
```

2. **Créer un environnement virtuel**

```bash
# Créer l'environnement virtuel
python -m venv venv

# Activer l'environnement virtuel
# Sur macOS/Linux :
source venv/bin/activate
# Sur Windows :
# venv\Scripts\activate
```

3. **Installer les dépendances**

```bash
# Mettre à jour pip
pip install --upgrade pip

# Installer les dépendances
pip install -r requirements.txt
```

4. **Lancer l'application**

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### Option 2 : Installation Locale Simple

1. **Cloner le repository**

```bash
git clone <repository-url>
cd stocks_predictions
```

2. **Installer les dépendances**

```bash
pip install -r requirements.txt
```

3. **Lancer l'application**

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### Option 3 : Avec Docker

1. **Construire l'image**

```bash
docker build -t stocks-predictions .
```

2. **Lancer le conteneur**

```bash
docker run -p 8000:8383 stocks-predictions
```

## 📚 API Endpoints

### 🏠 Page d'accueil

- **GET** `/` - Page d'accueil avec informations sur l'API

### 📊 Données d'actions

- **GET** `/getdata` - Récupérer les données d'actions stockées
- **POST** `/createdata` - Créer de nouvelles données d'actions

### 🔮 Prédictions

- **POST** `/predict` - Générer des prédictions de prix d'actions

### 📖 Documentation

- **GET** `/docs` - Interface Swagger UI
- **GET** `/redoc` - Documentation ReDoc

## 🔧 Utilisation

### Exemple de requête pour créer des données

```bash
curl -X POST "http://localhost:8000/createdata" \
     -H "Content-Type: application/json" \
     -d '{"stocks": "AAPL,MSFT,GOOGL"}'
```

### Exemple de requête pour prédire

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"stocks": "AAPL"}'
```

## 🧠 Modèle de Machine Learning

Le projet utilise un modèle LSTM (Long Short-Term Memory) pour la prédiction :

- **Architecture** : 2 couches LSTM (50 unités chacune) + 2 couches Dense
- **Données d'entraînement** : 80% des données historiques (2012-2019)
- **Fenêtre temporelle** : 60 jours pour la prédiction
- **Métrique** : RMSE (Root Mean Square Error)
- **Optimiseur** : Adam
- **Fonction de perte** : Mean Squared Error

## 📁 Structure du Projet

```
stocks_predictions/
├── app.py                 # Application FastAPI principale
├── create_data.py         # Script pour créer des données d'actions
├── create_prediction.py   # Script pour entraîner le modèle LSTM
├── test_API_Yahoo.py      # Tests de l'API Yahoo Finance
├── requirements.txt       # Dépendances Python
├── Dockerfile            # Configuration Docker
├── predict_stocks.pkl    # Modèle entraîné (généré)
├── stocks.json           # Données d'actions (généré)
├── stocks_predictions.json # Prédictions (généré)
└── nasdaq-listed.csv     # Liste des actions NASDAQ
```

## 🔍 Scripts Utiles

### Créer des données d'actions

```bash
python create_data.py
```

### Entraîner le modèle de prédiction

```bash
python create_prediction.py
```

### Tester l'API Yahoo Finance

```bash
python test_API_Yahoo.py
```

## Create connection REST API to Qlik Cloud

![Connector REST](assets/connector1.png)
![Connector REST](assets/connector2.png)

## 📊 Données Sources

- **Yahoo Finance** : Données historiques des actions
- **Période** : 2012-2019 (configurable)
- **Actions supportées** : Toutes les actions disponibles sur Yahoo Finance
- **Données** : Prix de clôture, ajustés automatiquement

## 🚨 Limitations

- Les prédictions sont basées sur des données historiques
- Le modèle nécessite au moins 60 jours de données pour fonctionner
- Les performances peuvent varier selon la volatilité des actions
- L'API Yahoo Finance peut avoir des limitations de taux

## 🤝 Contribution

1. Fork le projet
2. Créer une branche feature (`git checkout -b feature/AmazingFeature`)
3. Commit les changements (`git commit -m 'Add some AmazingFeature'`)
4. Push vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrir une Pull Request

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## 📞 Support

Pour toute question ou problème, veuillez ouvrir une issue sur GitHub.

---

**Note** : Ce projet est à des fins éducatives et de recherche. Les prédictions ne constituent pas des conseils financiers.
