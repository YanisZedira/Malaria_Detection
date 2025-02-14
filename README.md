
```
Lien de l'application via streamlit : https://malariadetection-yaniszedira.streamlit.app/

# Détection Automatisée de la Malaria

Ce projet utilise l'intelligence artificielle et des réseaux de neurones convolutionnels pour détecter la malaria sur des images microscopiques de cellules sanguines. Il repose sur l'entraînement de trois modèles :
- **CNN from scratch** (conçu spécifiquement pour ce projet)
- **VGG16** (pré-entraîné sur ImageNet, avec fine-tuning)
- **ResNet50** (pré-entraîné sur ImageNet, avec fine-tuning)

Une **application web Streamlit** permet de charger une image et d'obtenir une prédiction immédiate, simplifiant l'usage pour les professionnels de santé et le grand public.

---

## Installation

```bash
pip install tensorflow opencv-python streamlit matplotlib scikit-learn pillow numpy
```

---

## Utilisation

### 1️⃣ Lancer l’application Streamlit :
```bash
streamlit run ReconaissanceMalaria.py
```

L’application s’ouvrira automatiquement dans votre navigateur.

---

## 🧪 Fonctionnalités

- **Détection automatique de la malaria** sur des images microscopiques.
- **Utilisation de 50 images de test directement disponibles** sur l’interface.
- **Possibilité d'importer vos propres images** pour obtenir une prédiction personnalisée.
- **Chargement rapide des modèles CNN, VGG16 et ResNet50 pré-entraînés**.
- **Onglet d'information** pour comprendre le fonctionnement du CNN from scratch.
- Interface fluide et moderne via **Streamlit**.

---

## 📦 Modèles Utilisés

- **CNN from scratch** : Créé spécifiquement pour ce projet, optimisé pour la détection d’infection sur des images 64x64.
- **VGG16** : Réseau pré-entraîné sur ImageNet, adapté à notre tâche avec fine-tuning des dernières couches.
- **ResNet50** : Réseau pré-entraîné sur ImageNet, fine-tuning des 20 dernières couches.

Les meilleurs poids des modèles sont déjà entraînés et stockés dans le dossier `MalariaProject/` sur Google Drive.

---

## 📊 Métriques d’Évaluation

Les performances des modèles sont évaluées avec les métriques classiques :

| Métrique  | Formule                                                                                         | Description |
|-----------|--------------------------------------------------------------------------------------------------|-------------|
| **Accuracy (Exactitude)** | \( \frac{TP + TN}{TP + TN + FP + FN} \) | Proportion de prédictions correctes. |
| **Precision (Précision)** | \( \frac{TP}{TP + FP} \) | Proportion des infections prédites qui sont réellement infectées. |
| **Recall (Sensibilité)** | \( \frac{TP}{TP + FN} \) | Capacité à détecter toutes les cellules infectées. |
| **F1-score** | \( 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} \) | Moyenne harmonique entre précision et rappel. |

\( TP \) : Vrais positifs  
\( TN \) : Vrais négatifs  
\( FP \) : Faux positifs  
\( FN \) : Faux négatifs

---

## 📷 Jeu de Données

Le jeu de données est issu de l’ensemble public **Cell Images for Malaria Detection** disponible sur Kaggle :  
https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria

Il est composé de :
- **Parasitized** : Cellules infectées par la malaria.
- **Uninfected** : Cellules saines.

---
### Lancer l’application
```bash
streamlit run ReconaissanceMalaria.py
```

---

## 🧠 Comprendre le CNN From Scratch

Le CNN personnalisé est constitué de plusieurs couches :

1. **Convolution** : Extraction de caractéristiques via des filtres.
2. **MaxPooling** : Réduction de la taille tout en gardant les informations importantes.
3. **Batch Normalization** : Stabilisation et accélération de l’apprentissage.
4. **Flatten** : Mise à plat des features extraits.
5. **Dense Layers** : Couches entièrement connectées pour la classification.
6. **Dropout** : Prévention du surapprentissage.
7. **Sigmoid** : Activation pour une sortie binaire (infecté ou sain).

L’architecture est optimisée spécifiquement pour les images microscopiques de cellules.

---

## 🎨 Capture d’Écran de l’Interface Streamlit

*![image](https://github.com/user-attachments/assets/eeffb837-dc32-48eb-9bdc-5f1f05b9f3f7)
*

---

## 🛠️ Développements Futurs

- Ajout de **DenseNet** ou **EfficientNet** pour encore améliorer les performances.
- Ajout d’une **API Flask** pour l’intégration dans d’autres systèmes.
- Optimisation des temps d’inférence.
- Déploiement sur **HuggingFace Spaces**.

---

## 📄 Licence

Ce projet est sous licence **MIT**.

---

## 🤝 Contributions

Les contributions sont les bienvenues !  
Vous pouvez ouvrir une issue ou soumettre une pull request.

---

## 🧑‍💻 Auteur

- **Yanis Zedira** – https://www.linkedin.com/in/yaniszedira/
- Contact : myanis.zedira@gmail.com

---

## 🔗 Ressources Utiles

- [Documentation TensorFlow](https://www.tensorflow.org/)
- [Documentation Streamlit](https://docs.streamlit.io/)
- [Jeu de données Kaggle](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria)
```

Ce code peut être directement copié dans un fichier `README.md` pour afficher correctement le contenu sur GitHub.
