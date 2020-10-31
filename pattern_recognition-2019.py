import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_validate
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as shc
import pandas as pd
Cm = []  # ta dianusmata opou tha metrithoun oi apostaseis tous me alla shmeia kai tha apofasizetai
#gia kathe shmeio an anhkei se mia ap autes tis omades h allou

n_clusters = []  #lista me ton arithmo twn clusters pou exoume meta thn ektelesh
#tou algorithmou bsas gia kathe epanalhpsh sto range tou theta.
#ta apotelsmata tha mpoun s ena plot kai o arithmos clusters pou tha emfanistei perissoteres
#fores einai kai pithanoteros arithmos clusters


def load_data(f_name):#pairnei to arxeio csv kai to metatrepei se array
    data = np.loadtxt(f_name, usecols=range(0,3))# ftiaxnei to pinaka me ta data
    np.random.shuffle(data)# anakateuei ta data gia na nai dikaio
    return data

def adjust_data(n,data):#me th sunarthsh auth tha prosarmosoume ta dedomena wste na apanthsoume
    #sto erwthma ths dhmotikothtas
    global total
    r_t=[None]*1682#lista me sunoliko arithmo ratings kathe tainias
    r_a=[None]*1682#lista me athroisma ratings kathe tainias
    for i in range(0,1682):
        k=0
        m=data[i][2]
        if(r_t[int(data[i][1])]==None):
            for j in range(0,n):
                if(data[i][1]==data[j][1]):
                    k=k+1
                    m=m+data[j][2]
            r_t[int(data[i][1])]=k
            r_a[int(data[i][1])]=m
    r=[]#lista me average ratings kathe tainias
    r1=[]#lista me sunoliko arithmo ratings, xanaftiaxnoume th lista giati kapoies tainies
    #den exoun ratings
    for i in range(0,1682):
        if(r_t[i]!= None and r_t[i]>=100):
            r.append(r_a[i]/r_t[i])
            r1.append(r_t[i])
    total=np.column_stack((r,r1))#2d array me athroisma ratings kai average kai to xoume thesei global
    #wste na mporei na xrhsimopoihthei kai se alles sunarthseis
    return total

def theta_range_calc(theta_min, theta_max, theta_step):#upologizoume to theta range(katofli anomoiothtas)
    theta_range = np.arange(theta_min, theta_max, theta_step)#bazoume to katofli se pinaka
    return theta_range

def plot_alg(x,y,labels,centroids_exists,file_plot):#h sunarthsh gia thn emfanish tou plot
    #an theloume na deixoume centroids tote bazoume 1, diaforetika 0
    colors=10*["g.","r.","c.","b.","k."]
    if(centroids_exists==0):
        for i in range(len(y)):
            plt.plot(x[i], y[i],colors[labels],markersize=4)
    else:
        for i in range(len(x)):
            plt.plot(x[i][0], x[i][1], colors[labels[i]],markersize=4)
        plt.scatter(y[:,0],y[:,1], marker='x', s=350,linewidths=5)
        
    plt.savefig('plots/'+file_plot)
    plt.close()
    #plt.show()
    

    #onoma arxeiou #megisth kai elaxisth timh anomoiothtas #megistos arithmos clusters
def bsas(f_name, theta_min, theta_max, theta_step, q=-1):

    #global total
    data = load_data(f_name=f_name)#fwrtonoume ta dedomena

    theta_range = theta_range_calc(theta_min=theta_min, theta_max=theta_max, theta_step=theta_step)
    #upologizoume katofli anomoiothtas
    
    n = data.shape[0]  # sunolikos arithmos dianusmatwn gia omadopoihsh
    
    adjust_data(n,data)#o pinakas me ta kainourgia dedomena

    for Theta in theta_range:
        cluster = 1  # arithmos clusters, to arxikopoioume me 1 se kathe epanalhpsh wste kathe
        #fora pou ekteleitai o algorithmos na epistrefei diaforetiko arithmo omadwn analoga me
        #to theta
        Cm[:] = []  # gia ton idio logo kanoume clear th lista me ta dianusmata
        Cm.append(total[0])  # arxikopoioume th lista me to prwto cluster pou einai kai to
        #prwto dianusma apo ta dedomena
        
        # h omadopoihsh twn upoloipwn dianusmatwn
        for i in range(1, len(total)):
            d_min = np.linalg.norm(Cm[0]-total[i])#h elaxisth apostash metaxu tou cluster kai tou dianusmatos
           
            for k in range(1, cluster):#gia ta mexri ekeinh th stigmh clusters pou exoun brethei
                distance = np.linalg.norm(Cm[k]-total[i])#h apostash enos tuxaiou shmeiou me to cluster
                if (d_min > distance):#an h apostash einai mikroterh apo thn elaxisth tote pernei th thesh ths
                    d_min = distance
            if (q == -1):#q=-1 shmainei xwris orio clusters
                if (d_min > Theta):  #an h minimum apostash einai megaluterh apo to katofli anomoiothtas
                    cluster = cluster + 1  # tote ftiaxnete kainourgio cluster
                    Cm.append(total[i])  #prosthetoume to dianusma sth lista
                
            else:#antistoixa otan exoume thesh oria gia ton arithmo twn clusters
                if (d_min > Theta) and (m < q):
                    cluster = cluster + 1
                    Cm.append(total[i])

        n_clusters.append(cluster)  #prosthetoume twn arithmo clusters pou brhkame sto telos ths epanalhpshs
        #kai epanalambanoume th diadikasia analoga me to theta range pou orisame

    #print n_clusters
    #print Cm

    plot_alg(theta_range,n_clusters,0,0,'bsas')

def hierarchical(total,num):#hierarchical clustering
    #se periptwsh pou de thelete na apothikeutei to grafhma ston antistoixo fakelo
    #sto desktop allaxte ta paths sto savefig
    Y=total
    
    plt.figure(figsize=(10, 7))  
    dend = shc.dendrogram(shc.linkage(Y, method='ward'))
    plt.savefig('plots/hierarchical dendrogram ward')
    plt.close()
    cluster = AgglomerativeClustering(n_clusters=num, affinity='euclidean', linkage='ward')  
    cluster.fit_predict(Y)
    plt.figure(figsize=(10, 7))  
    plt.scatter(Y[:,0], Y[:,1], c=cluster.labels_, cmap='rainbow')  
    plt.savefig('plots/hierarchical ward')
    plt.close()
def cal_nclusters():#fernei ton arithmo twn clusters, bash tou parapanw theorhmatos oti dld o arithmos
    #me th megaluterh suxnothta epilegetai
    max=0
    max_num=n_clusters[0]
    for x in n_clusters:
        i=0
        for y in n_clusters:
           if(x==y):
               i=i+1
        if(i>max):
            max=i
            max_num=x
    return max_num
def kmeans(total,ideal_n_clusters):#algorithmos kmeans
    global Y
    
    Y=total
    clf = KMeans(n_clusters=ideal_n_clusters)  
    clf.fit(Y)
    centroids = clf.cluster_centers_
    labels=clf.labels_
    
    plot_alg(total,centroids,labels,1,'kmeans')


bsas('u.data',30,200,1,-1)
num=cal_nclusters()
kmeans(total,num)
hierarchical(total,num)
