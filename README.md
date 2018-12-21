# Presentation : Inpainting

Le but de ce projet est de netoyer une image bruité ou ayant une partie manquante à l'aide d'outils de machine learning. Vous pouvez trouver ci-dessous les résultats de notre algorithme :

<center>
  <img src="https://github.com/Kabegami/Inpainting/blob/master/res/README_images/lena_noise" alt="Example Lena 50%">
    <img src="https://github.com/Kabegami/Inpainting/blob/master/res/README_images/lena_original" alt="Example Lena original">
</center>


## Instuctions

1. Il faut mettre les images dans data/images.
2. Il faut lancer main --image_name <image_name> --method alpha pour calculer les paramètres utilisés par l'algorithme de machine learning. Cette étape est **obligatoire**.
3. Vous pouvez choisir entre netoyer une image d'un bruit avec --method noise ou un trou dans l'image avec heuristique ou neighbour.

## Explications

On décompose l'image en petit patch de taille h qui sont des bouts d'images. On utilise ensuite une régression Lasso afin d'exprimer les pixel du patch à corriger en tant que combinaison linéaire des patchs non bruités. Une des propriétées de cette algorithme est qu'elle retourne des résultats sparse ce qui signifie qu'on reconstruit l'erreur avec peu de patch. Cette propriété est interessante car prendre tous les coefficients conduit à avoir des effet de flou car il "moyenne" les résultats.

Pour la correction de trous dans l'image, l'ordre de réparation des pixels est primordial et différentes heuristiques sont étudiées.

Pour plus d'information, vous pouvez lire le rapport dans report


## Sources

* Bin Shen and Wei Hu and Zhang, Yimin and Zhang, Yu-Jin, Image Inpainting via Sparse Representation Proceedings of the 2009 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP ’09)

* Julien Mairal Sparse coding and Dictionnary Learning for Image Analysis INRIA Visual Recognition and Machine Learning Summer School, 2010

* A. Criminisi, P. Perez, K. Toyama Region Filling and Object Removal by Exemplar-Based Image Inpainting IEEE Transaction on Image Processing (Vol 13-9), 2004
