# Presentation : Inpainting

Le but de ce projet est de nettoyer une image bruité ou ayant une partie manquante à l'aide d'outils de machine learning. Vous pouvez trouver ci-dessous les résultats de notre algorithme : format (originale, bruit, correction)


![Correction de bruit](res/README_images/small_noise_example)


<!-- ![Example Lena original](https://github.com/Kabegami/Inpainting/tree/master/res/README_images/lena_original.png) -->


## Instructions

1. Il faut mettre les images dans data/images.
2. Il faut lancer : `python main.py --image_name <image_name> --method alpha` pour calculer les paramètres utilisés par l'algorithme de machine learning. Cette étape est **obligatoire**.
3. Vous pouvez choisir entre nettoyer une image d'un bruit avec --method noise ou un trou dans l'image avec heuristique ou neighbour.

## Explications

On décompose l'image en petit patch de taille h qui sont des bouts d'images. On utilise ensuite une régression Lasso afin d'exprimer les pixels du patch à corriger en tant que combinaison linéaire des patchs non bruités. Une des propriétés de cet algorithme est qu'il retourne des résultats sparses ce qui signifie qu'on reconstruit l'erreur avec peu de patch. Cette propriété est intéressante car prendre tous les coefficients conduit à avoir des effet de flou car il "moyenne" les résultats.

Pour la correction de trous dans l'image, l'ordre de réparation des pixels est primordial et différentes heuristiques sont étudiées.
Une heuristique consiste à corriger en priorité les pixels qui ont le plus de voisins non bruités(avec une préférence pour les pixels qui n'ont pas été corrigés).
Nous avons également implémenté une heuristique à base de confiance décrite plus en détail dans le rapport.




## Sources

* Bin Shen and Wei Hu and Zhang, Yimin and Zhang, Yu-Jin, Image Inpainting via Sparse Representation Proceedings of the 2009 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP ’09)

* Julien Mairal Sparse coding and Dictionnary Learning for Image Analysis INRIA Visual Recognition and Machine Learning Summer School, 2010

* A. Criminisi, P. Perez, K. Toyama Region Filling and Object Removal by Exemplar-Based Image Inpainting IEEE Transaction on Image Processing (Vol 13-9), 2004
