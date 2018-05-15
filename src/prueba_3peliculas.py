# -*- coding: utf-8 -*-
#!/usr/bin/python

#código usado para probar la performance de clasificación sobre 3 peliculas (Blood Diamond, The Wolf of Wall Street y The Revenant). En los 3 casos el entrenamiento de la svm se hizo con datos de Blood Diamond únicamente. Faltan imports y algún que otro detalle de código el cuál cambió luego de las pruebas, pero la base está.

from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV


def tune_svm_3movies():
    cantTrain = 1500
    trainIxsLeo = random.sample(xrange(0,len(facesLeoBloodFiltered)),cantTrain)
    trainIxsNoLeo = random.sample(xrange(0,len(facesNoLeo)),cantTrain)

    embsLeoTrain = np.array([np.array(f.get_emb()) for i,f in enumerate(facesLeoBloodFiltered) if i in trainIxsLeo])
    embsNoLeoTrain = np.array([np.array(f.get_emb()) for i,f in enumerate(facesNoLeo) if i in trainIxsNoLeo])
    meanPerFeat = np.mean(embsLeoTrain,axis=0,keepdims=True)
    centeredData = embsLeoTrain - meanPerFeat
    sqSum = np.sum(centeredData*centeredData,axis=1)
    sqrtSqSumMean = np.sqrt(np.mean(sqSum))
    myGamma = 1. / (2.*sqrtSqSumMean**2)
    print("Gamma calculado: '{}'".format(myGamma))
    X_train = np.vstack([embsLeoTrain,embsNoLeoTrain])
    y_train = np.hstack([np.zeros(embsLeoTrain.shape[0]),np.ones(embsNoLeoTrain.shape[0])])

    embsLeoBloodTest = np.array([np.array(f.get_emb()) for i,f in enumerate(facesLeoBloodFiltered) if not(i in trainIxsLeo)])
    embsLeoWolfTest = np.array([np.array(f.get_emb()) for f in facesLeoWolf])
    embsLeoRevenantTest = np.vstack([np.array([np.array(f.get_emb()) for f in facesLeoRevenant]),np.array([np.array(f.get_emb()) for f in facesLeoRevenantFiltered])])
    embsLeoTest = np.vstack([embsLeoBloodTest,embsLeoWolfTest,embsLeoRevenantTest])
    embsNoLeoTest = np.array([np.array(f.get_emb()) for i,f in enumerate(facesNoLeo) if not(i in trainIxsNoLeo)])
    X_test = np.vstack([embsLeoTest,embsNoLeoTest])
    y_test = np.hstack([np.zeros(embsLeoTest.shape[0]),np.ones(embsNoLeoTest.shape[0])])

    # Set the parameters by cross-validation
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': ['auto',myGamma], 'C': [0.1, 1, 10, 100, 1000]}]

    scores = ['precision', 'recall']
    print('======================================================')
    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)

        clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5,
                           scoring='%s_macro' % score)
        clf.fit(X_train, y_train)

        print("Best parameters set found on development set:")
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.4f (+/-%0.04f) for %r"
                  % (mean, std * 2, params))

        print("Detailed classification report:")
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print('======================================================')
        print("best est: {}".format(clf.best_estimator_))

def calculate_precision_recall_f1score_and_plot_3movies(embsLeoTest,testSetName):
    for exNum in range(0,execTimes):
        trainIxsLeo = random.sample(xrange(0,len(facesLeoBloodFiltered)),cantTrain)
        trainIxsNoLeo = random.sample(xrange(0,len(facesNoLeo)),cantTrain)

        sys.stdout.write("\rRealizando ejecución #{}".format(exNum))
        sys.stdout.flush()
        embsLeoTrain = np.array([np.array(f.get_emb()) for i,f in enumerate(facesLeoBloodFiltered) if i in trainIxsLeo])
        embsNoLeoTrain = np.array([np.array(f.get_emb()) for i,f in enumerate(facesNoLeo) if i in trainIxsNoLeo])

        if testSetName == 'embsLeoBloodTest':
            embsLeoTest = np.array([np.array(f.get_emb()) for i,f in enumerate(facesLeoBloodFiltered) if not(i in trainIxsLeo)])
        #~ embsLeoBloodTest = np.array([np.array(f.get_emb()) for i,f in enumerate(facesLeoBloodFiltered) if not(i in trainIxsLeo)])
        #~ embsLeoWolfTest = np.array([np.array(f.get_emb()) for f in facesLeoWolf])
        #~ embsLeoRevenantTest = np.vstack([np.array([np.array(f.get_emb()) for f in facesLeoRevenant]),np.array([np.array(f.get_emb()) for f in facesLeoRevenantFiltered])])

        #~ embsLeoTest = np.vstack([embsLeoBloodTest,embsLeoWolfTest,embsLeoRevenantTest])
        embsNoLeoTest = np.array([np.array(f.get_emb()) for i,f in enumerate(facesNoLeo) if not(i in trainIxsNoLeo)])

        X_test = np.vstack([embsLeoTest,embsNoLeoTest])
        y_test = np.hstack([np.zeros(embsLeoTest.shape[0]),np.ones(embsNoLeoTest.shape[0])])

        meanPerFeat = np.mean(embsLeoTrain,axis=0,keepdims=True)
        centeredData = embsLeoTrain - meanPerFeat
        sqSum = np.sum(centeredData*centeredData,axis=1)
        sqrtSqSumMean = np.sqrt(np.mean(sqSum))
        myGamma = 1. / (2.*sqrtSqSumMean**2)

        svm = SVC(C=10, gamma=myGamma, kernel='rbf')

        cantDataList = [1,5,10,20,30,40,50,75,100,150,200,300,350,400]+range(500,len(embsLeoTrain),100)
        target_names = ['class Leo', 'class NoLeo']
        prfs = [] #precision_recall_fscore_support
        trueAndPred = []
        for cantData in cantDataList:
            X_train = np.vstack([embsLeoTrain[:cantData],embsNoLeoTrain])
            y_train = np.hstack([np.zeros(cantData),np.ones(embsNoLeoTrain.shape[0])])
            #print('======================================================')
            #print("Probando con {} datos con 'clase Leo':".format(cantData))
            svm.fit(X_train, y_train)
            y_true, y_pred = y_test, svm.predict(X_test)
            prfs.append(precision_recall_fscore_support(y_true, y_pred))
            trueAndPred.append((y_true,y_pred))
            #print()
            #print("Detailed classification report:")
            #print(classification_report(y_true, y_pred,target_names=target_names))
            #print('======================================================')
        execVals.append((prfs,trueAndPred))

    prfsExecCant = np.array([ex[0] for ex in execVals]) #tengo las 20 ejecuciones, y en cada indice los 14 valores segun cantData, donde cada valor es una 4-tuple con un un array de shape (2,) donde la primer componente es el valor en Leo y la 2da en NoLeo.
    prfsCantExec = np.swapaxes(prfsExecCant,0,1) #reacomodamos para tener los 14 valores segun cantData y luego en cada indice las 20 ejecuciones.
    prfsCant = map(lambda i: np.mean(prfsCantExec[i],axis=0),range(0,len(cantDataList))) #promediamos las execTimes ejecuciones

    #plotting:
    fig = plt.figure(3,figsize=(8,8))
    plt.suptitle("Test Set: {}".format(testSetName))

    plt.subplot(3,1,1)
    plt.title('precision')
    pLeo = [x[0][0] for x in prfsCant]
    pNoLeo = [x[0][1] for x in prfsCant]
    plt.plot(cantDataList,pLeo,':b.',cantDataList,pNoLeo,':c.')
    plt.grid(True)
    plt.legend(['Leo','NoLeo'],loc='best',ncol=2,fontsize=9,shadow=True, fancybox=True)

    plt.subplot(3,1,2)
    plt.title('recall')
    rLeo = [x[1][0] for x in prfsCant]
    rNoLeo = [x[1][1] for x in prfsCant]
    plt.plot(cantDataList,rLeo,':r.',cantDataList,rNoLeo,':m.')
    plt.grid(True)
    plt.legend(['Leo','NoLeo'],loc='best',ncol=2,fontsize=9,shadow=True, fancybox=True)

    plt.subplot(3,1,3)
    plt.title('f1-score')
    fLeo = [x[2][0] for x in prfsCant]
    fNoLeo = [x[2][1] for x in prfsCant]
    plt.plot(cantDataList,fLeo,':g.',cantDataList,fNoLeo,':y.')
    plt.grid(True)
    plt.legend(['Leo','NoLeo'],loc='best',ncol=2,fontsize=9,shadow=True, fancybox=True)

    fig.savefig('./plots/p_r_f1_scores_seed{}_{}.png'.format(seedN,testSetName), bbox_inches='tight',dpi=300)
    fig.clf()

def classify_3movies():
    load = False

    filesLeoBlood = listFiles('./test_sets/seguidas_3442')
    filesLeoWolf = listFiles('./test_sets/wolf_727')
    filesLeoRevenant = listFiles('./test_sets/revenant_868')
    filesLFW = listFiles('./test_sets/lfw')

    with open('./pickled_objs/lfwImgs_detThr{}.obj'.format(DETECTION_THR), 'rb') as fp:
            imgsLFW = cPickle.load(fp)
            fp.close()
    with open('./pickled_objs/svm_leoZoo&Lfw.obj','rb') as fp:
            svm = cPickle.load(fp)
            fp.close()

    if load:
        with open('./pickled_objs/imgsLeoBlood.obj', 'rb') as fp:
            imgsLeoBlood = cPickle.load(fp)
            fp.close()
        with open('./pickled_objs/imgsLeoWolf.obj', 'rb') as fp:
            imgsLeoWolf = cPickle.load(fp)
            fp.close()
        with open('./pickled_objs/imgsLeoRevenant.obj', 'rb') as fp:
            imgsLeoRevenant = cPickle.load(fp)
            fp.close()

    else:
        sys.stdout.write("Creando contenedores de imagenes...")
        sys.stdout.flush()
        imgsLeoBlood = [myImage(fIx,filesLeoBlood[fIx]) for fIx in range(0,len(filesLeoBlood))]
        imgsLeoWolf = [myImage(fIx,filesLeoWolf[fIx]) for fIx in range(0,len(filesLeoWolf))]
        imgsLeoRevenant = [myImage(fIx,filesLeoRevenant[fIx]) for fIx in range(0,len(filesLeoRevenant))]
        print("OK")

        stEmbs = time.time()
        sys.stdout.write("Procesando imágenes...")
        sys.stdout.flush()
        imgsLeoBlood = pool.map(pool_process_img, imgsLeoBlood)
        imgsLeoWolf = pool.map(pool_process_img, imgsLeoWolf)
        imgsLeoRevenant = pool.map(pool_process_img, imgsLeoRevenant)
        print("OK. tiempo: {}.".format(datetime.timedelta(seconds=time.time()-stEmbs)))

        with open(pathCheck("./pickled_objs/")+"imgsLeoBlood.obj", 'wb') as fp:
            cPickle.dump(imgsLeoBlood, fp)
            fp.close()
        with open(pathCheck("./pickled_objs/")+"imgsLeoWolf.obj", 'wb') as fp:
            cPickle.dump(imgsLeoWolf, fp)
            fp.close()
        with open(pathCheck("./pickled_objs/")+"imgsLeoRevenant.obj", 'wb') as fp:
            cPickle.dump(imgsLeoRevenant, fp)
            fp.close()

    #detect Leo(s)
    facesLeoBlood = flattenList([img.get_faces() for img in imgsLeoBlood])
    facesLeoWolf1Det = flattenList([img.get_faces() for img in imgsLeoWolf if img.nDetected<2])
    facesLeoRevenant1Det = flattenList([img.get_faces() for img in imgsLeoRevenant if img.nDetected<2])

    #Hacemos un filtrado porque en facesLeoBlood hay caras que no son de leo, lo mismo pasa en Revenant donde se obtuvo más de 1 detección. Las caras de Wolf son todas de leo, pues se seleccionaron los frames a mano.
    facesLeoRevenantFiltered = flattenList([img.get_faces() for img in imgsLeoRevenant if img.nDetected>=2])
    facesLeoRevenantFiltered = [face for face in facesLeoRevenantFiltered if svm.predict(np.array(face.get_emb()).reshape(1,-1))[0] == 0]
    facesNoLeo = flattenList([img.get_faces() for img in imgsLFW])
    facesLeoBloodFiltered = [face for face in facesLeoBlood if svm.predict(np.array(face.get_emb()).reshape(1,-1))[0] == 0]

    cantTrain = 1500 #int(round(len(filesLeo)*0.8))
    execVals = []
    execTimes = 20
    seedN = 2019
    random.seed(seedN) #for reproducibility
    #~ tune_svm_3movies()

    embsLeoWolfTest = np.array([np.array(f.get_emb()) for f in facesLeoWolf1Det])
    embsLeoRevenantTest = np.vstack([np.array([np.array(f.get_emb()) for f in facesLeoRevenant1Det]),np.array([np.array(f.get_emb()) for f in facesLeoRevenantFiltered])])

    calculate_precision_recall_f1score_and_plot_3movies(None, 'embsLeoBloodTest')
    calculate_precision_recall_f1score_and_plot_3movies(embsLeoWolfTest, 'embsLeoWolfTest')
    calculate_precision_recall_f1score_and_plot_3movies(embsLeoRevenantTest, 'embsLeoRevenantTest')


#=======================================================================
#Prueba de clasificación "de bola" según distancia, sin svm.

def myPredict(test, cen, rad, k):
    pred = []
    for t in test:
        if euclidean_distances([t,cen])[0][1] < k * rad:
            pred.append(0)
        else:
            pred.append(1)
    return pred

def calculate_precision_recall_f1score_and_plot_3movies_noSVM(embsLeoTest,testSetName,k):
    for exNum in range(0,execTimes):
        trainIxsLeo = random.sample(xrange(0,len(facesLeoBloodFiltered)),cantTrain)
        trainIxsNoLeo = random.sample(xrange(0,len(facesNoLeo)),cantTrain)

        print("Realizando ejecución #{}:".format(exNum))
        embsLeoTrain = np.array([np.array(f.get_emb()) for i,f in enumerate(facesLeoBloodFiltered) if i in trainIxsLeo])
        embsNoLeoTrain = np.array([np.array(f.get_emb()) for i,f in enumerate(facesNoLeo) if i in trainIxsNoLeo])

        if testSetName == 'embsLeoBloodTest':
            embsLeoTest = np.array([np.array(f.get_emb()) for i,f in enumerate(facesLeoBloodFiltered) if not(i in trainIxsLeo)])
        embsNoLeoTest = np.array([np.array(f.get_emb()) for i,f in enumerate(facesNoLeo) if not(i in trainIxsNoLeo)])

        X_test = np.vstack([embsLeoTest,embsNoLeoTest])
        y_test = np.hstack([np.zeros(embsLeoTest.shape[0]),np.ones(embsNoLeoTest.shape[0])])

        cantDataList = [5,10,20,30,40,50,75,100,150,200,300,350,400]+range(500,len(embsLeoTrain),100)
        target_names = ['class Leo', 'class NoLeo']
        prfs = [] #precision_recall_fscore_support
        trueAndPred = []
        for cantData in cantDataList:
            X_train = np.vstack([embsLeoTrain[:cantData],embsNoLeoTrain])
            y_train = np.hstack([np.zeros(cantData),np.ones(embsNoLeoTrain.shape[0])])

            meanPerFeat = np.mean(embsLeoTrain[:cantData],axis=0,keepdims=True)
            centeredData = embsLeoTrain[:cantData] - meanPerFeat
            sqSum = np.sum(centeredData*centeredData,axis=1)
            sqrtSqSumMean = np.sqrt(np.mean(sqSum))
            #~ myGamma = 1. / (2.*sqrtSqSumMean**2)
            print("\tCant. data: {} - sqrtSqSumMean: {} - k = {}".format(cantData,sqrtSqSumMean,k))

            y_true, y_pred = y_test, myPredict(X_test, meanPerFeat.reshape(128,), sqrtSqSumMean, k)
            prfs.append(precision_recall_fscore_support(y_true, y_pred))
            trueAndPred.append((y_true,y_pred))
        execVals.append((prfs,trueAndPred))

    prfsExecCant = np.array([ex[0] for ex in execVals]) #tengo las exNum ejecuciones, y en cada indice los valores segun cantData, donde cada valor es una 4-tuple con un un array de shape (2,) donde la primer componente es el valor en Leo y la 2da en NoLeo.
    prfsCantExec = np.swapaxes(prfsExecCant,0,1) #reacomodamos para tener los 14 valores segun cantData y luego en cada indice las 20 ejecuciones.
    prfsCant = map(lambda i: np.mean(prfsCantExec[i],axis=0),range(0,len(cantDataList))) #promediamos las execTimes ejecuciones

    #plotting:
    fig = plt.figure(3,figsize=(8,8))
    plt.suptitle("Test Set: {}, k = {}".format(testSetName,k))

    plt.subplot(3,1,1)
    plt.title('precision')
    pLeo = [x[0][0] for x in prfsCant]
    pNoLeo = [x[0][1] for x in prfsCant]
    plt.plot(cantDataList,pLeo,':b.',cantDataList,pNoLeo,':c.')
    plt.grid(True)
    plt.legend(['Leo','NoLeo'],loc='best',ncol=2,fontsize=9,shadow=True, fancybox=True)

    plt.subplot(3,1,2)
    plt.title('recall')
    rLeo = [x[1][0] for x in prfsCant]
    rNoLeo = [x[1][1] for x in prfsCant]
    plt.plot(cantDataList,rLeo,':r.',cantDataList,rNoLeo,':m.')
    plt.grid(True)
    plt.legend(['Leo','NoLeo'],loc='best',ncol=2,fontsize=9,shadow=True, fancybox=True)

    plt.subplot(3,1,3)
    plt.title('f1-score')
    fLeo = [x[2][0] for x in prfsCant]
    fNoLeo = [x[2][1] for x in prfsCant]
    plt.plot(cantDataList,fLeo,':g.',cantDataList,fNoLeo,':y.')
    plt.grid(True)
    plt.legend(['Leo','NoLeo'],loc='best',ncol=2,fontsize=9,shadow=True, fancybox=True)

    fig.savefig('./plots/centroide/p_r_f1_scores_seed{}_k{}_{}.png'.format(seedN,k,testSetName), bbox_inches='tight',dpi=300)
    fig.clf()

def classify_3movies_noSVM():
    load = True

    filesLeoBlood = listFiles('./test_sets/test_set_seguidas_3442')
    filesLeoWolf = listFiles('./test_sets/test_set_wolf_727')
    filesLeoRevenant = listFiles('./test_sets/test_set_revenant_868')
    filesNoLeo = listFiles('./test_sets/lfw')

    if load:
        with open('./pickled_objs/imgsLeoBlood.obj', 'rb') as fp:
            imgsLeoBlood = cPickle.load(fp)
            timesLeoBlood = cPickle.load(fp)
            fp.close()
        with open('./pickled_objs/imgsLeoWolf.obj', 'rb') as fp:
            imgsLeoWolf = cPickle.load(fp)
            timesLeoWolf = cPickle.load(fp)
            fp.close()
        with open('./pickled_objs/imgsLeoRevenant.obj', 'rb') as fp:
            imgsLeoRevenant = cPickle.load(fp)
            timesLeoRevenant = cPickle.load(fp)
            fp.close()
        with open('./pickled_objs/imgsNoLeo_lfw.obj', 'rb') as fp:
            imgsNoLeo = cPickle.load(fp)
            timesNoLeo = cPickle.load(fp)
            fp.close()
    else:
        sys.stdout.write("Creando contenedores de imagenes...")
        sys.stdout.flush()
        imgsLeoBlood = [myImage(fIx,filesLeoBlood[fIx]) for fIx in range(0,len(filesLeoBlood))]
        imgsLeoWolf = [myImage(fIx,filesLeoWolf[fIx]) for fIx in range(0,len(filesLeoWolf))]
        imgsLeoRevenant = [myImage(fIx,filesLeoRevenant[fIx]) for fIx in range(0,len(filesLeoRevenant))]
        imgsNoLeo = [myImage(fIx,filesNoLeo[fIx]) for fIx in range(0,len(filesNoLeo))]
        print("OK")

        bar = progressbar.ProgressBar()
        print("Procesando Leo Blood...")
        timesLeoBlood = []
        for img in bar(imgsLeoBlood):
            timesLeoBlood.append(img.process(upscale=0))

        bar = progressbar.ProgressBar()
        print("Procesando Leo Wolf...")
        timesLeoWolf = []
        for img in bar(imgsLeoWolf):
            timesLeoWolf.append(img.process(upscale=0))

        bar = progressbar.ProgressBar()
        print("Procesando Leo Revenant...")
        timesLeoRevenant = []
        for img in bar(imgsLeoRevenant):
            timesLeoRevenant.append(img.process(upscale=0))

        bar = progressbar.ProgressBar()
        print("Procesando No Leo...")
        timesNoLeo = []
        for img in bar(imgsNoLeo):
            timesNoLeo.append(img.process(upscale=0))

        with open(pathCheck("./pickled_objs/")+"imgsLeoBlood.obj", 'wb') as fp:
            cPickle.dump(imgsLeoBlood, fp)
            cPickle.dump(timesLeoBlood, fp)
            fp.close()
        with open(pathCheck("./pickled_objs/")+"imgsLeoWolf.obj", 'wb') as fp:
            cPickle.dump(imgsLeoWolf, fp)
            cPickle.dump(timesLeoWolf, fp)
            fp.close()
        with open(pathCheck("./pickled_objs/")+"imgsLeoRevenant.obj", 'wb') as fp:
            cPickle.dump(imgsLeoRevenant, fp)
            cPickle.dump(timesLeoRevenant, fp)
            fp.close()
        with open(pathCheck("./pickled_objs/")+"imgsNoLeo_lfw.obj", 'wb') as fp:
            cPickle.dump(imgsNoLeo, fp)
            cPickle.dump(timesNoLeo, fp)
            fp.close()

    #load svm-zoo
    with open('./pickled_objs/svm_leoZoo&Lfw.obj', 'rb') as fp:
        svm = cPickle.load(fp)
        fp.close()

    facesLeoBlood = flattenList([img.get_faces() for img in imgsLeoBlood])
    facesLeoWolf = flattenList([img.get_faces() for img in imgsLeoWolf if img.nDetected<2])
    facesLeoRevenant = flattenList([img.get_faces() for img in imgsLeoRevenant if img.nDetected<2])

    facesLeoRevenantFiltered = flattenList([img.get_faces() for img in imgsLeoRevenant if img.nDetected>=2])
    facesLeoRevenantFiltered = [face for face in facesLeoRevenantFiltered if svm.predict(np.array(face.get_emb()).reshape(1,-1))[0] == 0]
    facesNoLeo = flattenList([img.get_faces() for img in imgsNoLeo])
    facesLeoBloodFiltered = [face for face in facesLeoBlood if svm.predict(np.array(face.get_emb()).reshape(1,-1))[0] == 0]

    cantTrain = 1500
    execVals = []
    execTimes = 20
    seedN = 2018
    random.seed(seedN) #for reproducibility

    embsLeoWolfTest = np.array([np.array(f.get_emb()) for f in facesLeoWolf])
    embsLeoRevenantTest = np.vstack([np.array([np.array(f.get_emb()) for f in facesLeoRevenant]),np.array([np.array(f.get_emb()) for f in facesLeoRevenantFiltered])])

    ks = np.arange(1,2,0.1)
    for k in ks:
        calculate_precision_recall_f1score_and_plot_3movies_noSVM(None, 'embsLeoBloodTest',k)
        calculate_precision_recall_f1score_and_plot_3movies_noSVM(embsLeoWolfTest, 'embsLeoWolfTest',k)
        calculate_precision_recall_f1score_and_plot_3movies_noSVM(embsLeoRevenantTest, 'embsLeoRevenantTest',k)
