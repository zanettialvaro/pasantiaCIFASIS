# -*- coding: utf-8 -*-
#!/usr/bin/python

from __future__ import print_function
import numpy as np
import dlib
import cv2
import sys
import os
import cPickle
#~ from skimage.measure import compare_ssim as ssim  #http://www.cns.nyu.edu/pub/eero/wang03-reprint.pdf
import multiprocessing
import time
import datetime
from tqdm import tqdm
import random
import copy
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.svm import SVC

import utils
import myRect
import myImage
import myFace
import faceGroup
import fillTrack

parallelExec = False

#~ loadSimilarity = True
#~ loadImgs = True
loadSimilarity = False
loadImgs = False

def group_faces_by_folder():
    #get all folders
    folders = []
    for img in imgs:
        folder,basename = os.path.split(img.get_filepath())
        if not(folder in folders):
            folders.append(folder)

    #group images
    fgID = 0
    allFaceGroups = []
    for currFolder in folders:
        max_detected = 0
        currFaceID = 0
        fGs = [faceGroup(fgID)]
        fGs[0].set_faceID(currFaceID)
        fgID += 1
        imgsInCurrFold = [img for img in imgs if currFolder == os.path.split(img.get_filepath())[0]]
        for img in imgsInCurrFold:
            if img.get_nDetected() > max_detected:
                max_detected = img.get_nDetected()
            for face in img.get_faces():
                accepted = False
                for fg in fGs:
                    if fg.is_empty() or (fg.overlap(face) and not(fg.image_here(face.get_inImage()))):
                        accepted = True
                        face.set_id(fg.get_faceID())
                        fg.add_face(face)
                        break
                if not(accepted):
                    fg_new = faceGroup(fgID)
                    currFaceID += 1
                    fg_new.set_faceID(currFaceID)
                    face.set_id(currFaceID)
                    fgID += 1
                    fg_new.add_face(face)
                    fGs.append(fg_new)

        for fg in fGs:
            fg.calculate_detLimits()

        for img in imgsInCurrFold:
            if img.get_nDetected() < max_detected:
                frameIx = img.get_frameIx()
                for fg in fGs:
                    if not(fg.is_empty()) and not(fg.image_here(img)) and fg.lDet < frameIx < fg.hDet:
                        prevF,nextF = fg.get_prev_next_det(frameIx)
                        if np.abs(frameIx-prevF)<=3 or np.abs(frameIx-nextF)<=3:
                            fg.add_imgToInterpolate(img)

        allFaceGroups = allFaceGroups + fGs
    return allFaceGroups

def group_faces_by_similarity():
    fgID = 0
    allFaceGroups = []
    for (l,h) in scene_limits:
        max_detected = 0
        currFaceID = 0
        fGs = [faceGroup.faceGroup(fgID)]
        fGs[0].set_scene_low_high(l,h)
        fGs[0].set_faceID(currFaceID)
        #~ print("fg creado al principio con id{} en limits: ({},{})".format(fGs[0].fgID,l,h))
        fgID += 1
        for img in imgs[l:h+1]:
            if img.get_nDetected() > max_detected:
                max_detected = img.get_nDetected()
                #~ print("img: {} - max detected: {}".format(img.get_filepath(),max_detected))
            for face in img.get_faces():
                accepted = False
                for fg in fGs:
                    if fg.is_empty() or (fg.overlap(face) and not(fg.image_here(face.get_inImage()))):
                        accepted = True
                        face.set_id(fg.get_faceID())
                        fg.add_face(face)
                        #~ print("face {}-id{} agregada en fg{}".format(os.path.basename(face.get_inImage().get_filepath()),face.get_id(),fg.fgID))
                        break
                if not(accepted):
                    fg_new = faceGroup.faceGroup(fgID)
                    #~ print("fg_new creado id{}".format(fg_new.fgID))
                    fg_new.set_scene_low_high(l,h)
                    currFaceID += 1
                    fg_new.set_faceID(currFaceID)
                    face.set_id(currFaceID)
                    fgID += 1
                    fg_new.add_face(face)
                    #~ print("face {}-id{} agregada2 en fg{}".format(os.path.basename(face.get_inImage().get_filepath()),face.get_id(),fg_new.fgID))
                    fGs.append(fg_new)
        for fg in fGs:
            fg.calculate_detLimits()
        for img in imgs[l:h+1]:
            if img.get_nDetected() < max_detected:
                frameIx = img.get_frameIx()
                for fg in fGs:
                    if not(fg.is_empty()) and not(fg.image_here(img)) and fg.lDet < frameIx < fg.hDet:
                        prevF,nextF = fg.get_prev_next_det(frameIx)
                        if np.abs(frameIx-prevF)<=3 or np.abs(frameIx-nextF)<=3:
                            fg.add_imgToInterpolate(img)
                            #~ print("agregando img {} (fIx={}) a interpolar en fg {} con lDet,hDet = {},{}".format(os.path.basename(img.get_filepath()),img.get_frameIx(),fg.fgID,fg.lDet,fg.hDet))
        allFaceGroups = allFaceGroups + fGs
    return allFaceGroups

#~ def similarity_ssim((myImg1,myImg2)):
    #~ path1 = myImg1.get_filepath()
    #~ path2 = myImg2.get_filepath()
    #~ img1 = cv2.cvtColor(cv2.imread(path1),cv2.COLOR_BGR2GRAY)
    #~ img2 = cv2.cvtColor(cv2.imread(path2),cv2.COLOR_BGR2GRAY)
    #~ return ssim(img1,img2)

def similarity_hist((myImg1,myImg2)):
    path1 = myImg1.get_filepath()
    path2 = myImg2.get_filepath()

    img1 = cv2.cvtColor(cv2.imread(path1),cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(cv2.imread(path2),cv2.COLOR_BGR2GRAY)

    hist1 = cv2.calcHist([img1],channels=[0],mask=None,histSize=[256],ranges=[0,256]).reshape(256,)
    hist2 = cv2.calcHist([img2],channels=[0],mask=None,histSize=[256],ranges=[0,256]).reshape(256,)

    return euclidean_distances([hist1,hist2])[0][1]

def process_and_save_LFW():
    lfwFiles = utils.listFiles("../test_sets/lfw")
    sys.stdout.write("Creando contenedores de imagenes (LFW)...")
    sys.stdout.flush()
    lfwImgs = [myImage.myImage(fIx,lfwFiles[fIx]) for fIx in range(0,len(lfwFiles))]
    print("OK")

    if parallelExec:
        sys.stdout.write("Procesando lfwImgs...")
        sys.stdout.flush()
        lfwImgs = pool.map(pool_process_img, lfwImgs)
        print("OK")
    else:
        print("Procesando lfwImgs...")
        for img in tqdm(lfwImgs):
            pool_process_img(img)

    sys.stdout.write("pickling lfwImgs.obj...")
    sys.stdout.flush()
    with open(utils.pathCheck("./pickled_objs/")+"lfwImgs_detThr{}.obj".format(utils.DETECTION_THR), 'wb') as fp:
        cPickle.dump(lfwImgs, fp)
        fp.close()
    print("OK")

#=======================================================================

red = (0,0,255)
green = (0,255,0)
blue = (255,0,0)
white = (255,255,255)

def pool_process_img(img):
    img.process()
    return img

def pool_process1_fg((fg,willIdentify,set_name)):
    if not(fg.is_empty()):
        fg.calculate_splines(set_name, plot=True)
        fg.interpolate()
        fg.smooth_faces()
        if not(willIdentify):
            fg.save_smoothed_to_disk(cropped_dest,draw_lands=False,folder_structure=False) #args: cropped_dest, split_relevant = False, draw_lands = False, folder_structure = False
    return fg

def pool_process2_fg((fg,svm)):
    if not(fg.is_empty()):
        fg.identify_face(svm)
        fg.save_smoothed_to_disk(cropped_dest,split_relevant=True,draw_lands=False,track_fills=True,folder_structure=False)
    return fg

if __name__ == '__main__':
    #~ sys.exit(1)
    
    startTime = time.time()
    print()
    if len(sys.argv)<3:
        print("Modo de uso: python face_detector.py 'imgs_src' 'cropped_dest' ['svm_imgs']")
        sys.exit()

    if sys.argv[1][-1] == '/':
        set_name = os.path.split(sys.argv[1][:-1])[1]
    else:
        set_name = os.path.split(sys.argv[1])[1]

    cropped_dest = sys.argv[2]
    print("Buscando imágenes en directorio: '{}'".format(sys.argv[1]))
    files = utils.listFiles(sys.argv[1])#[:400]
    if parallelExec:
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    willIdentify = len(sys.argv)>3

    #~ sys.exit(1)

    if willIdentify:
        loadImgs = True
        loadSimilarity = True

#Carga/cálculo de imágenes:
    if loadImgs:
        sys.stdout.write("Cargando imgs desde .obj...")
        sys.stdout.flush()
        with open("./pickled_objs/"+set_name+"_imgs_detThr{}.obj".format(utils.DETECTION_THR), 'rb') as fp:
            imgs = cPickle.load(fp)
            fp.close()
        print("OK")
    else:
        imgsProcStart = time.time()
        sys.stdout.write("Creando contenedores de imagenes...")
        sys.stdout.flush()
        imgs = [myImage.myImage(fIx,files[fIx]) for fIx in range(0,len(files))]
        print("OK")

        if parallelExec:
            sys.stdout.write("Procesando imgs...")
            sys.stdout.flush()
            imgs = pool.map(pool_process_img, imgs)
            print("OK")
        else:
            for img in tqdm(imgs,desc="Procesando imgs"):
                pool_process_img(img)
        imgsProcEnd = time.time()

        sys.stdout.write("pickling imgs.obj...")
        sys.stdout.flush()
        with open(utils.pathCheck("./pickled_objs/")+set_name+"_imgs_detThr{}.obj".format(utils.DETECTION_THR), 'wb') as fp:
            cPickle.dump(imgs, fp)
            fp.close()
        print("OK")

#Carga/cálculo de similarity
    if loadSimilarity:
        sys.stdout.write("Cargando similarity desde .obj...")
        sys.stdout.flush()
        with open("./pickled_objs/"+set_name+"_similarity_detThr{}.obj".format(utils.DETECTION_THR), 'rb') as fp:
            similarity = cPickle.load(fp)
            scene_changes = cPickle.load(fp)
            scene_limits = cPickle.load(fp)
            fp.close()
        print("OK")
    else:
        simProcStart = time.time()
        print()
        imgs1 = imgs[:-1]
        imgs2 = imgs[1:]
        imgsPairs = zip(imgs1,imgs2)
        if parallelExec:
            sys.stdout.write("Calculando similarity...")
            sys.stdout.flush()
            similarity = pool.map(similarity_hist,imgsPairs)
            print("OK")
        else:
            similarity = []
            for pair in tqdm(imgsPairs,desc="Calculando similarity"):
                similarity.append(similarity_hist(pair))

        scene_changes = [(i1,i1+1,sim) for (i1,sim) in enumerate(similarity) if sim > np.mean(similarity)+2*np.std(similarity)]
        scene_limits = [(0,scene_changes[0][0])]
        for i in range(0,len(scene_changes)-1):
            scene_limits.append((scene_changes[i][1],scene_changes[i+1][0]))
        i = len(scene_changes)-2
        scene_limits.append((scene_changes[i+1][1],len(files)-1))
        simProcEnd = time.time()

        sys.stdout.write("pickling similarity.obj...")
        sys.stdout.flush()
        with open(utils.pathCheck("./pickled_objs/")+set_name+"_similarity_detThr{}.obj".format(utils.DETECTION_THR), 'wb') as fp:
            cPickle.dump(similarity, fp)
            cPickle.dump(scene_changes,fp)
            cPickle.dump(scene_limits,fp)
            fp.close()
        print("OK")

#Modo de ejecución:
    if not willIdentify:
        fgsProc1Start = time.time()
        fGs = group_faces_by_similarity()
        print()
        if parallelExec:
            sys.stdout.write("faceGroups: Interpolando, suavizando y guardando SIN identificar...")
            sys.stdout.flush()
            fGs = pool.map(pool_process1_fg, [(fg,willIdentify,set_name) for fg in fGs])
            print("OK")
        else:
            print("faceGroups: Interpolando, suavizando y guardando SIN identificar...")
            for fg in tqdm(fGs):
                pool_process1_fg((fg,willIdentify,set_name))
        fgsProc1End = time.time()

        sys.stdout.write("pickling fGs.obj...")
        sys.stdout.flush()
        with open(utils.pathCheck("./pickled_objs/")+set_name+"_fGs.obj", 'wb') as fp:
            cPickle.dump(fGs, fp)
            fp.close()
        print("OK")
    else:
        print()
        sys.stdout.write("faceGroups: Cargando desde pickle...")
        sys.stdout.flush()
        with open("./pickled_objs/"+set_name+"_fGs.obj", 'rb') as fp:
            fGs = cPickle.load(fp)
            fp.close()
        print("OK")

        svmImgsProcStart = time.time()
        print()
        svmFiles = utils.listFiles(sys.argv[3])
        svmImgs = [myImage.myImage(fIx,svmFiles[fIx]) for fIx in range(0,len(svmFiles))]

        if parallelExec:
            sys.stdout.write("Procesando svmImgs...")
            sys.stdout.flush()
            svmImgs = pool.map(pool_process_img, svmImgs)
            print("OK")
        else:
            for img in tqdm(svmImgs,desc="Procesando svmImgs"):
                pool_process_img(img)

        svmFaces = utils.flattenList([img.get_faces() for img in svmImgs])
        #~ for sf in svmFaces:
            #~ save_face(sf,"./svmFaces")
        svmEmbs = np.array([np.array(face.get_emb()) for face in svmFaces])
        svmTrainCant = len(svmEmbs)
        svmImgsProcEnd = time.time()

        #cargamos las imagenes de lfw ya procesadas
        with open("./pickled_objs/lfwImgs_detThr{}.obj".format(utils.DETECTION_THR), 'rb') as fp:
            lfwImgs = cPickle.load(fp)
            fp.close()
        lfwFaces = utils.flattenList([img.get_faces() for img in lfwImgs])
        seedN = 2017
        random.seed(seedN)
        lfwIxs = random.sample(xrange(0,len(lfwFaces)),svmTrainCant)
        lfwEmbs = np.array([np.array(face.get_emb()) for i,face in enumerate(lfwFaces) if i in lfwIxs])

        #entrenamos la svm con estos datos anteriores.
        meanPerFeat = np.mean(svmEmbs,axis=0,keepdims=True)
        centeredData = svmEmbs - meanPerFeat
        sqSum = np.sum(centeredData*centeredData,axis=1)
        sqrtSqSumMean = np.sqrt(np.mean(sqSum))
        myGamma = 1. / (2.*sqrtSqSumMean**2)
        print("SVM:\n\tCentro:\n{}\n\tGamma: {}".format(meanPerFeat,myGamma))

        svm = SVC(C=10, gamma=myGamma, kernel='rbf')
        X_train = np.vstack([svmEmbs,lfwEmbs])
        y_train = np.hstack([np.zeros(svmTrainCant),np.ones(svmTrainCant)])
        svm.fit(X_train, y_train)

        fgsProc2Start = time.time()
        print()
        if parallelExec:
            sys.stdout.write("faceGroups: Identificando, recortando y guardando...")
            sys.stdout.flush()
            fGs = pool.map(pool_process2_fg, [(fg,svm) for fg in fGs])
            print("OK")
        else:
            print("faceGroups: Identificando, recortando y guardando...")
            for fg in tqdm(fGs):
                pool_process2_fg((fg,svm))
        fillTracks=[]
        for fg in fGs:
            for face in fg.faces:
                track = face.get_fillTrack()
                if track:
                    fillTracks.append(track)
        fgsProc2End = time.time()

        sys.stdout.write("pickling fillTracks.obj...")
        sys.stdout.flush()
        with open(utils.pathCheck("./pickled_objs/")+set_name+"_fillTracks.obj", 'wb') as fp:
            cPickle.dump(fillTracks, fp)
            fp.close()
        print("OK")

    if parallelExec:
        pool.close()
        pool.join()
    endTime = time.time()
    print()
    if not(loadImgs):
        print("Tiempo para procesar imgs '{}': {}".format(set_name,datetime.timedelta(seconds=imgsProcEnd-imgsProcStart)))
    if not(loadSimilarity):
        print("Tiempo para calcular 'similarity': {}".format(datetime.timedelta(seconds=simProcEnd-simProcStart)))
    if willIdentify:
        print("Tiempo para procesar svmImgs: {}".format(datetime.timedelta(seconds=svmImgsProcEnd-svmImgsProcStart)))
        print("Tiempo para procesar fGs (2): {}".format(datetime.timedelta(seconds=fgsProc2End-fgsProc2Start)))
    else:
        print("Tiempo para procesar fGs (1): {}".format(datetime.timedelta(seconds=fgsProc1End-fgsProc1Start)))
    print("Tiempo total del script: {}".format(datetime.timedelta(seconds=endTime-startTime)))
