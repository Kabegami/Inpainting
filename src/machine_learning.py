# coding: utf-8

from tools import *
import sklearn.linear_model as lm
import sklearn.model_selection as ms
import scipy.ndimage as ndimage
import random
from heapq import heappush, heappop
from heuristique import *

def loss(predict_patch, patch):
    Y =  (predict_patch - patch)
    return Y.dot(Y.T)
    

def find_best_aplha(X, Y):
    model = lm.LassoCV()
    print("searching the best alpha, it can take some time please wait")
    model.fit(X,Y)
    return model.alpha_


def patch_fit(X, Y, a=None):
    """ patch : matrice de taille (h, 3), dico : dictionnaire de patch
    de taille (h,3)"""
    if a is None:
        model = lm.LassoCV()
    else:
        model = lm.Lasso(alpha=a)
    model.fit(X,Y)
    coef = model.coef_
    if np.nonzero(coef) == 0:
        print('='*20)
        print('Attention tous les coefficients sont nuls !')
        print('='*20)
    return model


def predict_patch(dico, w):
    X = dico_to_data(dico, 10)
    p = X.dot(w)
    return p


def correct_img(img, p_h, step, alpha):
    NewImg = img.copy()
    dico = build_patchs(img, p_h, step)
    noised_patch, workingDico = split_dico(dico)
    Lind = getMissingPixelsImg(NewImg)
    while Lind != []:
        ind = Lind[0]
        i,j = ind
        patch = get_patch(i,j,p_h, NewImg)
        v = patch_to_vect(patch)
        missingPixels = getMissingPixels(v)[0]
        workingPixels = getWorkingPixels(v)
        trainX = make_data(workingDico, workingPixels)
        trainY = v[workingPixels]
        m = patch_fit(trainX, trainY, alpha)
        testX = make_data(workingDico, missingPixels)
        testY = m.predict(testX)
        for k in range(0, missingPixels.shape[0] - 2, 3):
            v1 = testY[k]
            v2 = testY[k+1]
            v3 = testY[k+2]
            ind = missingPixels[k]
            px, py = vectorIndexToPatchIndex(ind, p_h)
            ind = patchIndexToImgIndex(i, j, px, py, p_h)
            ni, nj = ind
            t = NewImg[ni][nj] == np.array([-100,-100,-100])
            b = np.any(t)
            if not b:
                print("warning the old pixel wasn't dead !")
                print("index :", ind)
                print("old pixel : ", NewImg[ni][nj])
            else:
                NewImg[ni][nj] = np.array([v1, v2, v3])
                Lind.remove(ind)
    return NewImg


def correct_hole(img, p_h, step, alpha):
    NewImg = img.copy()
    dico = build_patchs(img, p_h, step)
    noised_patch, workingDico = split_dico(dico)
    heapPixels = evaluateMissingPixels(NewImg, p_h)
    
    valueP, ind = heappop(heapPixels)
    while heapPixels != []:
        valueRef = valueP
        # parcours d'une couche à corriger
        while heapPixels != [] and valueRef == valueP:
            i,j = ind
            patch = get_patch(i,j,p_h, NewImg)
            v = patch_to_vect(patch)
            missingPixels = getMissingPixels(v)[0]
            if len(missingPixels) == 0:
                if heapPixels != []:
                    valueP, ind = heappop(heapPixels)
                continue
            workingPixels = getWorkingPixels(v)
            trainX = make_data(workingDico, workingPixels)
            trainY = v[workingPixels]
            m = patch_fit(trainX, trainY, alpha)
        
            testX = make_data(workingDico, missingPixels)
            testY = m.predict(testX)
            for k in range(0, missingPixels.shape[0] - 2, 3):
                v1 = testY[k]
                v2 = testY[k+1]
                v3 = testY[k+2]
                ind = missingPixels[k]
                px, py = vectorIndexToPatchIndex(ind, p_h)
                ind = patchIndexToImgIndex(i, j, px, py, p_h)
                ni, nj = ind
                t = NewImg[ni][nj] == np.array([-100,-100,-100])
                b = np.any(t)
                if not b:
                    print("warning the old pixel wasn't dead !")
                    print("index :", ind)
                    print("old pixel : ", NewImg[ni][nj])
                    # raise ValueError("Bad index")
                else:
                    NewImg[ni][nj] = np.array([v1, v2, v3])
            if heapPixels != []:
                valueP, ind = heappop(heapPixels)
        heapPixels = evaluateMissingPixels(NewImg, p_h)
    return NewImg


def correctImgHeuristique(img, p_h, step, alpha):
    NewImg = img.copy()
    dico = build_patchs(img, p_h, step)
    noised_patch, workingDico = split_dico(dico)
    MISSING_PIXEL = np.array([-100,-100,-100])
    confidenceMatrice = (NewImg != MISSING_PIXEL).astype(float) * -1
    heap = evaluateMissingPixelsConfidence(NewImg, p_h, confidenceMatrice)
    while heap != []:
        confidence, ind = heappop(heap)
        i,j = ind
        patch = get_patch(i,j,p_h, NewImg)
        v = patch_to_vect(patch)
        missingPixels = getMissingPixels(v)[0]
        workingPixels = getWorkingPixels(v)
        trainX = make_data(workingDico, workingPixels)
        trainY = v[workingPixels]
        m = patch_fit(trainX, trainY, alpha)

        testX = make_data(workingDico, missingPixels)
        testY = m.predict(testX)
        for k in range(0, missingPixels.shape[0] - 2, 3):
            v1 = testY[k]
            v2 = testY[k+1]
            v3 = testY[k+2]
            ind = missingPixels[k]
            px, py = vectorIndexToPatchIndex(ind, p_h)
            ind = patchIndexToImgIndex(i, j, px, py, p_h)
            ni, nj = ind
            t = NewImg[ni][nj] == np.array([-100,-100,-100])
            b = np.any(t)
            if not b:
                print("warning the old pixel wasn't dead !")
                print("index :", ind)
                print("old pixel : ", NewImg[ni][nj])
                # raise ValueError("Bad index")
            else:
                NewImg[ni][nj] = np.array([v1, v2, v3])
                confidenceMatrice[ni][nj] = confidence
        heap = evaluateMissingPixelsConfidence(NewImg, p_h, confidenceMatrice)
    return NewImg


def findAlphaNoise(M, p_h, step, fname, n=5):
    dico = build_patchs(M, p_h, step)
    missing, workingDico = split_dico(dico)
    bestAlpha = float('inf')
    i = 0
    while i < n:
        p = random.choice(missing)
        v = patch_to_vect(dico[p])
        ind = getWorkingPixels(v)
        X = make_data(dico, ind)
        Y = v[ind]
        alpha = find_best_aplha(X, Y)
        if alpha < bestAlpha:
            bestAlpha = alpha
        i +=1
    save_alpha(bestAlpha, fname)
    return bestAlpha


def evaluate_step(fname, bruit, start, end, n):
    M, h, l = read_im_tensor("images/"+fname+".png")
    img = M.copy()
    a,b,c = M.shape
    p_h = 3
    alpha = get_alpha('a_'+fname)
    stepSpace = np.linspace(start, end, n)
    noisev2(M, bruit, p_h)
    cpt = 1
    Lloss = []
    for st in stepSpace:
        step = int(st)
        N = correct_img(M, p_h, step, alpha)
        loss = imgMSE(img, N)
        Lloss.append(loss)
        cpt += 1
    plt.plot(stepSpace, Lloss)
    plt.xlabel('step')
    plt.ylabel('loss')
    plt.show()
        
    

def processing_img(fname, bruit=0.1):
    M, h, l = read_im_tensor("images/"+fname+".png")
    a,b,c = M.shape
    p_h = 3
    step = 6
    noisev2(M, bruit, p_h)
    name = "noised_images/"+fname+str(bruit*100)+"%.png"
    save_img_tensor(M, name)
    alpha = get_alpha('a_'+fname)
    N = correct_img(M, p_h, step, alpha)
    name = "corrected_images/"+fname+str(bruit*100)+"%.png"
    save_img_tensor(N, name)


def processing_img_hole(fname, num):
    M, h, l = read_im_tensor("images/"+fname+".png")
    a,b,c = M.shape
    p_h = 25
    step = 1
    delete_rect(M, 20, 20, 50, 50)
    name = "noised_images/"+fname+"_hole.png"
    save_img_tensor(M, name)
    alpha = get_alpha('a_'+fname)
    N = correct_hole(M, p_h, step, alpha)
    name = "corrected_images/"+fname+str(num)+"_hole.png"
    save_img_tensor(N, name)


def spiralHeuristique(fname):
    M, haut, l = read_im_tensor("images/"+fname+".png")
    a,b,c = M.shape
    p_h = 5
    step = 5
    alpha = get_alpha('a_'+fname)
    
    i,j = (20, 20)
    h1, h2 = (50, 50)
    delete_rect(M, 20,20, 50, 50)
    N = M.copy()
    
    pos = i, j
    S = SnailDirection()
    last = (i + h1//2 + 1, j + h2 // 2 +1)
    gen = S.moveGenerator(pos, N, last)
    dico = build_patchs(N, p_h, step)
    noised_patch, workingDico = split_dico(dico)
    for p in gen:
        i,j = p
        patch = get_patch(i,j,p_h, N)
        v = patch_to_vect(patch)
        p1 = patchIndextoVectIndex(p_h//2, p_h//2, p_h)
        missingPixels = np.array([p1, p1+1, p1+2])
        workingPixels = getWorkingPixels(v)
        trainX = make_data(workingDico, workingPixels)
        trainY = v[workingPixels]
        m = patch_fit(trainX, trainY, alpha)

        testX = make_data(workingDico, missingPixels)
        testY = m.predict(testX)

        for k in range(0, missingPixels.shape[0] - 2, 3):
            v1 = testY[k]
            v2 = testY[k+1]
            v3 = testY[k+2]
            ind = missingPixels[k]
            px, py = vectorIndexToPatchIndex(ind, p_h)
            ind = patchIndexToImgIndex(i, j, px, py, p_h)
            ni, nj = ind
            t = N[ni][nj] == np.array([-100,-100,-100])
            b = np.any(t)
            if not b:
                print("warning the old pixel wasn't dead !")
                print("index :", ind)
                print("old pixel : ", N[ni][nj])
                # raise ValueError("Bad index")
            else:
                N[ni][nj] = np.array([v1, v2, v3])
    return N
        
