# -*- coding: utf-8 -*-
#!/usr/bin/pyth

from sklearn.decomposition import PCA as sklearnPCA

def pca_over_embs(imgsRelevant,imgsNotRelevant):
    facesRelevant = flattenList([im.get_faces() for im in imgsRelevant])
    facesNotRelevant = flattenList([im.get_faces() for im in imgsNotRelevant])

    embsRelevant= [f.get_emb() for f in facesRelevant]
    embsNotRelevant = [f.get_emb() for f in facesNotRelevant]

    sklPCA = sklearnPCA(n_components=2)
    sklPCA = sklPCA.fit(embsRelevant+embsNotRelevant)
    embsRelevantPCA = sklPCA.transform(embsRelevant)
    embsNotRelevantPCA = sklPCA.transform(embsNotRelevant)

    plt.figure(2,figsize=(10,10))
    plt.plot([e[0] for e in embsRelevantPCA],[e[1] for e in embsRelevantPCA],'o',markersize=1.5,color='green',alpha=0.5,aa=True,label='Leo')
    plt.plot([e[0] for e in embsNotRelevantPCA],[e[1] for e in embsNotRelevantPCA],'o',markersize=1.3,color='red',alpha=0.5,aa=True,label='NoLeo')

    plt.title('embedding 128D PCA to 2dim')
    plt.xlabel('Comp1')
    plt.ylabel('Comp2')
    plt.legend()
    plt.savefig('./PCA.png', bbox_inches='tight', dpi = 800)
    plt.clf()

def calculate_distances(embsRelevant, embsNotRelevant):
    distsLeo = euclidean_distances(embsRelevant, embsRelevant)
    distsLeoTriang = [(distsLeo[i][j]) for i in range(0,len(distsLeo)) for j in range(0,len(distsLeo)) if j>i]
    distsWithNoLeo = euclidean_distances(embsRelevant, embsNotRelevant)
    return (distsLeoTriang, distsWithNoLeo)
