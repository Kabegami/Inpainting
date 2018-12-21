# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.ndimage as ndimage
import random
import pickle
from heapq import *


def save_alpha(alpha, filename):
        fname = "../parameters/alpha/" + filename
        f = open(fname, 'wb')
        pickle.dump(alpha, f)
        f.close()

def get_alpha(filename):
        fname = "../parameters/alpha/" + filename
        try:
                f = open(fname, 'rb')
                alpha = pickle.load(f)
                f.close()
                return alpha
        except:
                raise ValueError("You must calculate alpha before inpainting !")

def read_im(fn):
	M = plt.imread(fn)[:, :, :3]
	M = colors.rgb_to_hsv(M)
	h, l, b = M.shape
	img = M.reshape((h * l, 3))
	img = matriceMap(shrink, img)
	return img, h, l


def vect_to_img(img, N, M):
        r = img.reshape((N, M, 3))
        return r

def img_to_vect(img):
        N,M, i = img.shape
        r = img.reshape((N*M*3))
        return r


def read_im_tensor(fn):
	M = plt.imread(fn)[:, :, :3]
	h, l, c = M.shape
	return M, h, l


def display_img(M, h, l):
        i = M.copy()
        i = matriceMap(grow, i)
        i = i.reshape((h, l, 3))
        i = colors.hsv_to_rgb(i)
        plt.imshow(i)
        plt.show()


def display_img_tensor(M):
	i = M.copy()
	i[i < 0] = 0
	plt.imshow(i)
	plt.show()


def save_img_tensor(M, name):
        i = M.copy()
        i[i < 0] = 0
        i[i > 1] = 1
        plt.imshow(i)
        plt.show()
        plt.imsave(arr=i, fname=name)
        print('file saved !')


def shrink(x):
	"""Prend un point compris entre 0 et 1 et retourne ce point dans l'intervale -1, 1"""
	return (x - 0.5) / (1 - 0.5)


def grow(x):
	""" Prend un point compris entre -1 et 1 et retourne ce point dans l'intervale 0, 1"""
	return (x + 1) / 2


def delete_rect(img, i, j, height, width, v=np.array([-100, -100, -100])):
        """ Remplace le rectangle par la valeur v, et retourner le nombre de cases supprimées
	Attention, on considère que img est un tenseur (matrice de dimension 3)"""
        N, M, b = img.shape
        b1 = min(i + height, N)
        b2 = min(j + width, M)
        cpt = 0
        for c1 in range(i, b1):
                for c2 in range(j, b2):
                        a = img[c1][c2]
                        if np.any(img[c1][c2] == v):
                                cpt += 1
                        img[c1][c2] = v
        return ((b1 - i) * (b2 - j)) - cpt


def noise(img, prc, h=5):
	""" img doit être un tensor(matrice de dim 3)
        ne bruite pas les bords"""
	N, M, b = img.shape
	taille = N * M
	cpt = 0
	p = 0
	while p < prc:
		i = random.randint(h // 2, N - h // 2 - 2)
		j = random.randint(h // 2, M - h // 2 - 2)
		r = delete_rect(img, i, j, 1, 1)
		cpt += r
		p = cpt / taille

def imgMSE(img1, img2):
        r = img1 - img2
        return r.dot(r.T)


def get_patch(i, j, h, img):
        """im doit être un tensor(matrice de dim 3)"""
        N, M, b = img.shape
        a1 = i - h // 2
        a2 = i + h // 2
        b1 = j - h // 2
        b2 = j + h // 2
        patch = img[a1:a2+1, b1:b2+1]
        if patch.shape != (h, h, 3):
                print("Erreur le patch n'a pas la bonne dimention")
                raise ValueError("Patch dimension incorrect")
        return np.array(patch)


def patch_to_vect(patch):
	N, M, b = patch.shape
	v = np.reshape(patch, (N * M * 3))
	return v


def vect_to_patch(v, h):
	patch = v.reshape((h, h, 3))
	return patch


def dico_to_data(dico, h):
	M = []
	pixels = [[] for i in range(h * h * 3)]
	cpt = 0
	for p in dico.values():
		v = patch_to_vect(p)
		s = v.shape
		# le patch est incomplet, donc on rajoute des 0
		if s != (h * h * 3):
			temp = np.zeros((h * h * 3))
			for i in range(len(v)):
				k = v[i]
				temp[i] = k
			v = temp
		for i in range(len(v)):
			pixels[i].append(v[i])
	
	X = np.array(pixels)
	return X


def make_data(dico, ind):
        X = []
        for p in dico.values():
                v = patch_to_vect(p)
                col = v[ind]
                X.append(col)
        return np.array(X).T


def build_patchs(img, h, step):
	"""Retourne un dictionnaire de patchs"""
	# step = 1  # d'après le sujet on a un patch par pixel
	dico = dict()
	N, M, b = img.shape
	for i in range(h // 2, N - h // 2, step):
		for j in range(h // 2, M - h // 2, step):
			patch = get_patch(i, j, h, img)
			key = (i, j)
			dico[key] = patch
	return dico


def missing_pixels(patch, v=np.array([-100, -100, -100])):
        """Retourne un boolean indiquant s'il manque encore des pixels dans ce patch"""
        return np.any(patch == v)



def split_dico(dico):
        missing = []
        workingDico = dict()
        for k in dico.keys():
                p = dico[k]
                if missing_pixels(p):
                        missing.append(k)
                else:
                        workingDico[k] = p
        return missing, workingDico
                

def get_missing_patch(dico):
	L = []
	for k in dico.keys():
		p = dico[k]
		if missing_pixels(p):
			L.append(k)
	return L


def matriceMap(f, M):
	"""Applique la fonction f à toutes les cases i,j de la matrice M"""
	s = M.shape
	res = np.zeros(s)
	for i in range(s[0]):
		for j in range(s[1]):
			res[i][j] = f(M[i][j])
	return res


def dicoMap(f, dico):
	res = dict()
	for k in dico.keys():
		v = f(dico[k])
		res[k] = v
	return res


def getWorkingPixels(patch, v=-100):
        """ Prend un patch et retourne les index des pixels qui fonctionnent """
        ind = np.where(patch != v)
        return ind

def getMissingPixels(patch, v=-100):
        ind = np.where(patch == v)
        return ind


def getMissingPixelsImg(img, v=np.array((-100,-100,-100))):
        M,N, c = img.shape
        ind = []
        for i in range(M):
                for j in range(N):
                        if np.all(np.equal(img[i][j], v)):
                                ind.append((i,j))
        return ind
                        

def patchIndextoVectIndex(i, j, h):
        ind = j * 3 + (i * 3 * h)
        return int(ind)


def getStartPatch(i, j, h, N):
        """ retourne l'index de départ du patch, pour l'image sous forme d'un vecteur(retourne un int) """
        x = i - h // 2
        y = j - h // 2
        if x < 0 or y < 0:
                raise ValueError("Valeur négative !")
        index =  y* 3 + x * (3 * N)
        return index

        

def global_index(i, j, h, N, pos):
        indPatch = getStartPatch(i,j, h, N)
        return indPatch + pos
        


def vectorIndexToPatchIndex(i, h):
        v = i // 3
        x = v // h
        y = v % h
        return x,y
        

def patchIndexToImgIndex(i,j, px, py, h):
        """ i, j : les coordonnées du point central du patch
            px, py : les coordonnées du pixel dans le patch
        """
        x = (i - h // 2) + px
        y = (j - h // 2) + py
        return x,y
        

def getPixelsToFill():
	def test_func(values):
	    return values.sum()
	
	x = np.array([[1,2,3],[4,5,6],[7,8,9]])
	
	footprint = np.array([[1,1,1],
	                      [1,0,1],
	                      [1,1,1]])
	
	results = ndimage.generic_filter(x, test_func, footprint=footprint)


def heuristiqueBordure(img, ind):
        s = 0
        MISSING_PIXEL = np.array([-100, -100, -100])
        f = lambda x : 1 if np.any(x == MISSING_PIXEL) else -1
        i,j = ind
        try:
                v = img[i+1][j]
                s += f(v)
        except:
                pass
        try:
                v = img[i-1][j]
                s += f(v)
        except:
                pass
        try:
                v = img[i][j+1]
                s += f(v)
        except:
                pass                
        try:
                v = img[i][j-1]
                s += f(v)
        except:
                pass
        return s

def snail(i,j, h1, h2):
        pass

def heuristiquePatch(img, ind, h):
        i,j = ind
        p = get_patch(i,j,h, img)
        MISSING_PIXEL = np.array([-100, -100, -100])
        return np.sum(p == MISSING_PIXEL)

def confidencePatch(confidenceMatrice, ind, h):
        i,j = ind
        p = get_patch(i,j,h, confidenceMatrice)
        return np.mean(p)

def evaluateMissingPixelsConfidence(img, h, M):
        L = getMissingPixelsImg(img)
        heap = []
        for ind in L:
                v = confidencePatch(M, ind, h)
                heappush(heap, (v, ind))
        return heap

def evaluateMissingPixels(img, h):
        L = getMissingPixelsImg(img)
        heap = []
        for ind in L:
                v = heuristiquePatch(img, ind, h)
                heappush(heap, (v, ind))
        return heap


