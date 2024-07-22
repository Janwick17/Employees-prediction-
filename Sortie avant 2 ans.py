
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,mean_squared_error
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


import warnings
warnings.simplefilter(action = "ignore")


# In[254]:


get_ipython().system('pip install xgboost')


# # Lecture du fichier 

# In[255]:


df = pd.read_excel("C:\\Users\\majan\\Downloads\\SortiAv2ans.xlsx")


# # Conversion du fichier

# In[256]:


# Écrire les données dans un fichier CSV avec des tabulations comme délimiteur
df.to_csv("C:\\Users\\majan\\Downloads\\SortiAv2ans_formatted.csv")


# In[257]:


df.columns


# # Nettoyage de données

# # Supprimer les duplicats des matricules employés

# In[258]:


# Supprimer les lignes avec des valeurs dupliquées dans la colonne "Emp_Matricule"
df.drop_duplicates(subset=['Emp_Matricule'], keep='first', inplace=True)

# Réinitialiser les index après la suppression des lignes
df.reset_index(drop=True, inplace=True)


# In[259]:


print(df)


# # Remplacer les dates fictives par des dates logiques :

# In[260]:


# Remplacer les dates de sortie fictives par la dernière date en 2024
last_date_2024 = pd.Timestamp('2024-12-31 00:00:00')
df.loc[df['Date Sortie'] == pd.Timestamp('2999-12-31 00:00:00'), 'Date Sortie'] = last_date_2024

print(df['Date Sortie'])


# # Colonne Salaire 

# In[261]:


# Convertir la colonne "Salaire" en chaîne de caractères
df['Salaire'] = df['Salaire'].astype(str)

# Compter le nombre d'occurrences de chaque valeur dans la colonne "Salaire"
salaire_counts = df['Salaire'].value_counts()

# Obtenir la valeur la plus fréquente
valeur_plus_frequente = salaire_counts.idxmax()

print("La valeur la plus fréquente dans la colonne Salaire est :", valeur_plus_frequente)


# In[262]:


# Remplacer les valeurs de la colonne "Salaire" qui ont moins de 2 caractères par la valeur la plus fréquente
df.loc[df['Salaire'].str.len() < 2, 'Salaire'] = valeur_plus_frequente
#Afficher la colonne Salaire
print(df['Salaire'])


# # Colonne Unité Organisationnelle

# In[263]:


# Compter le nombre d'occurrences de chaque valeur dans la colonne "Unité Organisationnelle"
uniteorganisationnelle_counts = df['UniteOrganisationnelle'].value_counts()

# Obtenir la valeur la plus fréquente
valeur_plus_frequente = uniteorganisationnelle_counts.idxmax()

print("La valeur la plus fréquente dans la colonne Unité Organisationnelle est :", valeur_plus_frequente)


# In[264]:


# Remplacer les valeurs de la colonne "unitegorganisationelle" qui ont moins de 2 caractères par la valeur la plus fréquente
df.loc[df['UniteOrganisationnelle'].str.len() < 2, 'UniteOrganisationnelle'] = valeur_plus_frequente
#Afficher la colonne Unite Organisationnelle
print(df['UniteOrganisationnelle'])


# # Colonne Emploi

# In[265]:


# Compter le nombre d'occurrences de chaque valeur dans la colonne "Emploi"
emploi_counts = df['Emploi'].value_counts()

# Obtenir la valeur la plus fréquente
valeur_plus_frequente = emploi_counts.idxmax()

print("La valeur la plus fréquente dans la colonne Emploi est :", valeur_plus_frequente)


# In[266]:


# Remplacer les valeurs de la colonne "Emploi" qui ont moins de 2 caractères par la valeur la plus fréquente
df.loc[df['Emploi'].str.len() < 2, 'Emploi'] = valeur_plus_frequente
#Afficher la colonne Emploi
print(df['Emploi'])


# # vérification du data frame avant la conversion

# In[267]:


print(df)


# In[268]:


df.info


# In[269]:


df.dtypes


# # Conversion des données en types appropriés

# In[270]:


# Conversion de la colonne 'Salaire' en entier en gérant les valeurs non valides
df['Salaire'] = df['Salaire'].astype('float')

# Conversion de la colonne 'Date Sortie' en datetime64
df['Date Sortie'] = pd.to_datetime(df['Date Sortie'])

# Create and store the mappings
unite_org_map = dict(enumerate(df['UniteOrganisationnelle'].astype('category').cat.categories))
emploi_map = dict(enumerate(df['Emploi'].astype('category').cat.categories))

# Conversion des colonnes 'UniteOrganisationnelle' et 'Emploi' en catégories puis en codes numériques
df['UniteOrganisationnelle'] = df['UniteOrganisationnelle'].astype('category').cat.codes
df['Emploi'] = df['Emploi'].astype('category').cat.codes


# # Vérification des données après conversion 

# In[271]:


print(df)


# In[272]:


df.dtypes


# In[273]:


df.head()


# In[274]:


print(df['Salaire'])


# In[275]:


# Afficher les valeurs uniques de la colonne 'Salaire'
salaires_uniques = df['Salaire'].unique()

# Afficher les valeurs uniques de la colonne 'Salaire'
print(salaires_uniques)


# In[276]:


df.dtypes


# In[277]:


print(df['Emploi'])


# # Ajout d'une nouvelle colonne Durée

# In[278]:


from datetime import datetime

# Obtenir la date actuelle
date_actuelle = datetime.now()

# Calculer la durée en fonction de la condition spécifiée
df['Durée'] = df.apply(lambda row: (date_actuelle - row['Date Entree']).days if row['Date Sortie'] < date_actuelle else (date_actuelle - row['Date Entree']).days, axis=1)



# In[279]:


print(df)


# In[280]:


df.dtypes


# # Visualisation des données

# # Départ avant 2 ans par sexe d'employé

# In[281]:


import seaborn as sns
import matplotlib.pyplot as plt

# Créer un tableau croisé dynamique pour compter les départs avant deux ans par sexe
depart_2ans_par_sexe = pd.crosstab(df['Emp_Sexe'], df['Sortieavant2ans'])

# Tracer le graphique en barres empilées
plt.figure(figsize=(8, 6))
depart_2ans_par_sexe.plot(kind='bar', stacked=True)
plt.title('Départ avant deux ans par Sexe')
plt.xlabel('Sexe')
plt.ylabel('Nombre de départs')
plt.xticks(ticks=[0, 1], labels=['Homme', 'Femme'], rotation=0)

# Ajouter la légende
plt.legend(title='Départ avant deux ans', labels=['Non', 'Oui'])

plt.show()


# # Répartition des emplois par nombre d'employés

# In[282]:


# Sélectionner les emplois les plus courants (par exemple, les 10 premiers)
top_emplois = emploi_counts.nlargest(10)

# Tracer le graphique à barres pour les emplois les plus courants
plt.figure(figsize=(10, 6))
sns.barplot(x=top_emplois.index, y=top_emplois.values)
plt.title('Répartition des emplois (Top 10)')
plt.xlabel('Emploi')
plt.ylabel('Nombre d\'employés')
plt.xticks(rotation=45)  
plt.show()


# # Sortie avant 2 ans par emploi : (Emplois les plus fréquens)

# In[283]:


import seaborn as sns
import matplotlib.pyplot as plt

# Filtrer les emplois les plus fréquents
emplois_communs = df['Emploi'].value_counts().head(10).index.tolist()
df_filtre = df[df['Emploi'].isin(emplois_communs)]

# Créer un tableau croisé dynamique pour compter les départs avant deux ans par emploi
depart_2ans_par_emploi = pd.crosstab(df_filtre['Emploi'], df_filtre['Sortieavant2ans'])

# Tracer le graphique en barres empilées
plt.figure(figsize=(10, 6))
depart_2ans_par_emploi.plot(kind='bar', stacked=True)
plt.title('Départ avant deux ans par Emploi (Top 10)')
plt.xlabel('Emploi')
plt.ylabel('Nombre de départs')
plt.xticks(rotation=45)
plt.legend(title='Départ avant deux ans', labels=['Non', 'Oui'])
plt.show()


# # Sortie avant 2 ans selon l'age des employés :

# In[284]:


# Créer un histogramme pour visualiser le départ avant deux ans par groupe d'âge avec des couleurs personnalisées
plt.figure(figsize=(12, 6))
sns.histplot(data=df, x='Current_Age', hue='Sortieavant2ans', multiple='stack', bins=20,
             palette=['skyblue', 'salmon'], edgecolor='black', alpha=0.7)
plt.title('Départ avant 2 ans par groupe d\'âge')
plt.xlabel('Âge')
plt.ylabel('Nombre d\'employés')
plt.legend(title='Départ avant 2 ans', labels=['Non', 'Oui'])
plt.show()


# # Sélection des features

# In[285]:


# Séparation des caractéristiques (X) et de la cible (y)
X = df[['Emp_Matricule','Salaire', 'conge_nonpaye', 'Current_Age', 'Emp_Sexe',  'Durée', 'UniteOrganisationnelle', 'Emploi']]
y = df['Sortieavant2ans']


# # Partitionnement de données en train et test

# In[286]:


# Partitionnement des données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vérification des dimensions des ensembles d'entraînement et de test
print("Dimensions de l'ensemble d'entraînement X :", X_train.shape)
print("Dimensions de l'ensemble de test X :", X_test.shape)
print("Dimensions de l'ensemble d'entraînement y :", y_train.shape)
print("Dimensions de l'ensemble de test y :", y_test.shape)


# # Choix du modèle 

# # Logistic Regression 

# In[287]:


# Instanciation du modèle
log_reg_model = LogisticRegression()

# Entraînement du modèle sur l'ensemble d'entraînement
log_reg_model.fit(X_train, y_train)

# Prédiction sur l'ensemble de test
y_pred_log = log_reg_model.predict(X_test)


# # SVM 

# In[288]:


from sklearn.svm import SVC  # Importer la classe SVM

# Instanciation du modèle SVM
svm_model = SVC()

# Entraînement du modèle sur l'ensemble d'entraînement
svm_model.fit(X_train, y_train)

# Prédiction sur les données de test
y_pred_svm = svm_model.predict(X_test)


# # Random Forest

# In[289]:


# Initialisation du modèle Random Forest
rf_model = RandomForestClassifier(random_state=42)

# Entraînement du modèle
rf_model.fit(X_train, y_train)

# Prédiction sur les données de test
y_pred_rf = rf_model.predict(X_test)


# # XGBoost

# In[290]:


# Créer un classificateur XGBoost
xgb_classifier = XGBClassifier()

# Entraîner le classificateur sur les données d'entraînement
xgb_classifier.fit(X_train, y_train)

# Faire des prédictions sur les données de test
y_pred_xgb = xgb_classifier.predict(X_test)


# # Calcul des métriques des 4 modèles

# In[291]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Modèles
models = {
    'Logistic Regression': LogisticRegression(),
    'SVM': SVC(),
    'Random Forest': RandomForestClassifier(),
    'XGBoost': XGBClassifier()
}

# Entraîner et évaluer les modèles
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Métriques pour le modèle {name}:")
    print(f"Accuracy Score : {accuracy}")
    print(f"Precision Score : {precision}")
    print(f"Recall Score : {recall}")
    print(f"F1 Score : {f1}")
    print()


# # Test du modèle sur des nouvelles données

# In[292]:


import pandas as pd
from joblib import load

# Charger le modèle depuis le disque
model = load('modele_RandomForestClassifier.joblib')
'Emp_Matricule','Salaire', 'conge_nonpaye', 'Current_Age', 'Emp_Sexe',  'Durée', 'UniteOrganisationnelle', 'Emploi'
# Créer un exemple d'échantillon avec des caractéristiques similaires à celles de votre DataFrame
sample = pd.DataFrame({
   'Emp_Matricule': [123],
   'Salaire': [267400.0],
    'conge_nonpaye': [0],
   'Current_Age': [27],
   'Emp_Sexe': [1],  # Utilisez le même nom exact que celui utilisé dans votre modèle
   'Durée': [5],
   'UniteOrganisationnelle': [47],  
   'Emploi': [7],  # Ajoutez cette ligne
   'Sortieavant2ans': [1]  # Valeur cible pour la prédiction
})

# Faire des prédictions
result = model.predict(sample.drop(columns=['Sortieavant2ans']))  # Exclure la colonne cible des caractéristiques

# Afficher le résultat
if result == 1:
    print("Un employé peut quitter l'organisation avant 2 ans.")
else:
    print("Un employé peut rester avec l'organisation pendant 2 ans.")


# # Engregistrement du modèle

# In[293]:


from sklearn.ensemble import RandomForestClassifier
from joblib import dump

# Supposons que vous avez entraîné un modèle de forêt aléatoire appelé "model"
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Enregistrer le modèle sur le disque
dump(model, 'modele_RandomForestClassifier.joblib')


# In[294]:


import joblib

# Load the serialized model
model = joblib.load("modele_RandomForestClassifier.joblib")


# In[295]:


X_combined = pd.concat([X_train, X_test], axis=0)
y_combined = pd.concat([y_train, y_test], axis=0)
df_predictions_churn = X_combined.copy()
df_predictions_churn['quit'] = y_combined
df_predictions_churn.head()


# In[296]:


def map_back(column, mapping):
    return column.map(mapping)

# Convert numeric codes back to original categories
df_predictions['UniteOrganisationnelle'] = map_back(df_predictions['UniteOrganisationnelle'], unite_org_map)
df_predictions['Emploi'] = map_back(df_predictions['Emploi'], emploi_map)
df_predictions.head()


# In[252]:


df_predictions.to_csv("exit before 2 years_Predictions(2)")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




