# -*- coding: utf-8 -*-
#!/usr/bin/python

from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV

def tune_svm_zoo(myGamma, X_train, y_train):
    # Set the parameters by cross-validation
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': ['auto',1,10,100,1e-1,1e-2,1e-3,1e-4,myGamma],
                         'C': [1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

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
        falsePos=np.array([i for i,p in enumerate(y_pred) if p==0 and y_test[i]==1])-123
        rectsNoLeoDet = pool.map(detect_and_list,enumerate(filesNoLeo))
        rectsNoLeoDet = [r for r in rectsNoLeoDet if r] #saco los None
        for fp in falsePos:
            filepath=filesNoLeo[rectsNoLeoDet[fp][0].get_frameIx()]
            folder=os.path.split(filepath)[0]
            cantFiles = len(os.listdir(folder))
            print("NoLeo con clase: {} - fp: {} - file: {} - cantFiles: {}".format(clf.predict(embsNoLeoTest[fp].reshape((1,128))),fp,filepath,cantFiles))
            img=cv2.imread(filepath)
            cv2.imwrite(pathCheck("./NoLeosFalsePositives/")+os.path.basename(filepath),img)

def calculate_precision_recall_f1score_and_plot_zoo():
    for exNum in range(0,execTimes):
        trainIxsLeo = random.sample(xrange(0,len(filesLeo)),cantTrain)
        trainIxsNoLeo = random.sample(xrange(0,len(filesNoLeo)),cantTrain)

        sys.stdout.write("\rRealizando ejecución #{}".format(exNum))
        sys.stdout.flush()
        embsLeoTrain = [e for i,e in enumerate(embsLeo) if i in trainIxsLeo]
        embsLeoTrain = np.array([e for e in embsLeoTrain if e])
        embsNoLeoTrain = [e for i,e in enumerate(embsNoLeo) if i in trainIxsNoLeo]
        embsNoLeoTrain = np.array([e for e in embsNoLeoTrain if e])

        embsLeoTest = [e for i,e in enumerate(embsLeo) if not(i in trainIxsLeo)]
        embsLeoTest = np.array([e for e in embsLeoTest if e])
        embsNoLeoTest = copy.deepcopy(embsNoLeo)
        embsNoLeoTest = np.array([e for i,e in enumerate(embsNoLeoTest) if e if not(i in trainIxsNoLeo)])
        X_test = np.vstack([embsLeoTest,embsNoLeoTest])
        y_test = np.hstack([np.zeros(embsLeoTest.shape[0]),np.ones(embsNoLeoTest.shape[0])])

        meanPerFeat = np.mean(embsLeoTrain,axis=0,keepdims=True)
        centeredData = embsLeoTrain - meanPerFeat
        sqSum = np.sum(centeredData*centeredData,axis=1)
        sqrtSqSumMean = np.sqrt(np.mean(sqSum))
        myGamma = 1. / (2.*sqrtSqSumMean**2)

        svm = SVC(C=10, cache_size=200, class_weight=None, coef0=0.0,
                  decision_function_shape=None, degree=3, gamma=3.7145418027139296,
                  kernel='rbf', max_iter=-1, probability=False, random_state=None,
                  shrinking=True, tol=0.001, verbose=False)

        cantDataList = [1,5,10,20,30,40,50,75,100,150,200,300,350,len(embsLeoTrain)] #len(embsLeoTrain)=488
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
            #print("pred sobre los 123 Leo: {}".format(y_pred[:123]))
            #print('======================================================')
        execVals.append((prfs,trueAndPred))

    prfsExecCant = np.array([ex[0] for ex in execVals]) #tengo las 20 ejecuciones, y en cada indice los 14 valores segun cantData, donde cada valor es una 4-tuple con un un array de shape (2,) donde la primer componente es el valor en Leo y la 2da en NoLeo.
    prfsCantExec = np.swapaxes(prfsExecCant,0,1) #reacomodamos para tener los 14 valores segun cantData y luego en cada indice las 20 ejecuciones.
    prfsCant = map(lambda i: np.mean(prfsCantExec[i],axis=0),range(0,len(cantDataList))) #promediamos las execTimes ejecuciones

    #plotting:
    fig = plt.figure(3,figsize=(8,8))
    plt.suptitle("Evolucion de precision-recall-f1Score en base a la cantidad de datos de Leo para Train")

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

    fig.savefig('./p_r_f1_scores_seed{}.png'.format(seedN), bbox_inches='tight',dpi=300)
    fig.clf()
