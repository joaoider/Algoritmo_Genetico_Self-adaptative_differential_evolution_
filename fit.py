# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 16:28:05 2020

@author: Joao
"""
import numpy as np
from matplotlib import pyplot as plt
import multiprocessing
from multiprocessing import Pool
# from uncertainties import ufloat
# from uncertainties.umath import *
import time

""" ################################# Parâmetros ###############################################"""

n_runs = 40        # número de vezes que o programa irá rodar
popsize = 100      # número de possíveis soluções
gmax = 50000   # número de iterações

# dados do dispositivo
A = 0.0259
L = 0.144
n = 8.*127.  # número de junções
Z = 0.0026      # Figura de mérito teórica

name = '5'

# Modificar para cada arquivo

if name == '1':
    # ajustar para cada medida
    th = [89., 91.]  
    tc = [37.,39.]

# Search Space           
# [alpha (Seebeck Coefficient), R (internal resistance), Temperatura_hot, Temperature_Cold, F (mutation factor), CR (crossover rate)]
lbound = np.array([0.0, 0.0, th[0], tc[0],0.0, 0.0])
ubound = np.array([5., 5., th[1], tc[1], 2.0, 1.0]) 

# carrega dados experimentais do arquivo txt
data = np.loadtxt(name+'.txt', comments='#')
voltage  = data[:,0]   # coluna 1 - voltage
current  = data[:,1]   # coluna 2 - corrente

# calcula potência e resistência de carga a partir dos dados experimentais
power = voltage * current
Load_resistance = voltage/(n*current)

""" ########### Equação modelo a ser ajustada ######### """

# Equações a serem ajustadas do artigo 'TEG using effective material properties' 
def IVW(par): # par = np.array([alpha, R, Th, Tc F, CR]) 
    # return I (Eq.34),  Vn (Eq.35),  I*V = Wn (Eq.36)    
    I = par[0]*(par[2]-par[3])/(Load_resistance+par[1]) 
    V = n*par[0]*(par[2]-par[3])*(Load_resistance/par[1])/((Load_resistance/par[1])+1.)
    return I, V, I*V

""" ###########  Definição de Funções para Evolução diferencial adaptativa ####### """

################################  INICIALIZAÇÃO  ###################################
# Gera a população aleatória inicial X.
# Eq.8 do artigo.
def Pop():
    return lbound + (ubound-lbound)*np.random.rand(popsize, len(lbound))

################################   MUTAÇÃO   ################################
# Gera o vetor doador V.
# Eq.9 do artigo.
def Donor(X, F, Xbest, Xr1, Xr2):
    return X + F * (Xbest - X) + F * (Xr1 - Xr2)

################################   CROSSOVER   ###################################
# Cria um vetor U a partir do vetor inicial X e do vetor doador V.
# Eq.10 do artigo.
def Crossover(x, donor, sol):
    for j in range(len(lbound)):
        random_number = np.random.rand()
        if random_number <= x[-1] or j == np.random.choice(len(lbound)):
            sol[j] = donor[j]
        elif random_number > x[j]:
            sol[j] = x[j]
    return sol

# Pega os valores fora do search space (limites) e coloca dentro dos limites novamente.
def Penalty(sol):
    for j in range(len(lbound)):
        if sol[j] > ubound[j]:
            sol[j] = lbound[j] + np.random.rand()*(ubound[j]-lbound[j])
        if sol[j] < lbound[j]:
            sol[j] = lbound[j] + np.random.rand()*(ubound[j]-lbound[j])
    return sol

################################   SELEÇÃO   ###################################
# selecionando os valores com menor função objetiva
def Selection(sol, x):
    if objective(sol) < objective(x):
        x = sol
    else:
        x = x
    return x

# Função Objetiva (onde se analisa os melhores resultados)
# Eq. 11 do artigo.
def objective(sol):
    I, V, W = IVW(sol)
    return rmse(current, I) + rmse(voltage, V) + rmse(power, W)

# Root Mean Square Error (Eq.3 do artigo)
def rmse(predictions, targets):
    return (np.sqrt((1./np.float(np.size(targets)))*np.sum(np.power(((targets-predictions)/predictions),2))))

# Control Parameters atualization 
# Auto-adapta os valores de F e CR
def control(pop):
    if np.random.rand() < 0.1:
        pop[-2] = 0.1 + np.random.rand()*0.9 # Auto-adaptando F (Eq.12)
    if np.random.rand() < 0.1:    
        pop[-1] = np.random.rand() # Auto-adaptando CR (Eq.13)
    return pop

# Função que roda o programa inteiro
def main(count):
    a = open(str('results_')+str(name)+"_"+str(count)+".txt", "w")
    np.random.seed()
    g = 0
    pop = Pop()    # Gera a população inicial.
    score = np.zeros((popsize, 1))   
    donor = np.zeros((popsize, len(lbound)))
    trial = np.zeros((popsize, len(lbound)))
    for i in range(popsize):
        score[i] = objective(pop[i,:])
    best_index = np.argmin(score)
    g+=1
    plt.ion()
    while g <= gmax:
        for i in range(popsize):
            random_index = np.random.choice(np.delete(np.arange(popsize),i),2)
            # Eq.9 do artigo, criando vetor doador. (pop[i,-2] é o fator de mutação F)
            # donor[i,:] = Donor(pop[i,:], pop[i,-2], pop[best_index,:], pop[random_index[0],:], pop[random_index[1],:])
            donor[i,:] = pop[i,:] + pop[i,-2]*(pop[best_index,:]-pop[i,:])+pop[i,-2]*(pop[random_index[0],:]-pop[random_index[1],:])
            trial[i,:] = Crossover(pop[i,:], donor[i,:], trial[i,:])
            trial[i,:] = Penalty(trial[i,:])
            pop[i,:] = Selection(trial[i,:], pop[i,:])
            pop[i,:] = control(pop[i,:])
            score[i] = objective(pop[i,:])
        best_index = np.argmin(score)
        #print(g,score[best_index],pop[best_index,:])
        a.write("%d %.5f %.5f %.5f %.5f %.5f %.5f %.5f \n" % (g, score[best_index], pop[best_index,0], pop[best_index,1], pop[best_index,2], pop[best_index,3], pop[best_index,4], pop[best_index,5]))   
        # print("%d \n" % (g))
        g+=1
    a.close()
    return 0

""" ############## Função para gerar os gráficos ############### """

# Função para gerar gráfico comparando valores experimentais com o FIT
def grafico(x, y, yfit, title, xlabel, ylabel):
    plt.clf()
    plt.plot(x, y, 'k*', linewidth = 2, label=title)
    plt.plot(x, yfit, 'r--', linewidth = 2, label='Fit')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    #plt.show()
    plt.savefig(name+'_'+title+'_fit.png', dpi=300)

# Função para gerar gráfico comparando valores experimentais com o MODELO
def graficomodelo(xmod, ymod, xexp, yexp, title, xlabel, ylabel):
    plt.clf()
    plt.plot(xexp, yexp, 'k*', linewidth = 2,label=title)
    plt.plot(xmod, ymod, 'b--', linewidth = 2, label='Model')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    #plt.show()
    plt.savefig(name+'_'+title+'_model.png', dpi=300)
    
""" ################################ Rodando Programa ################################# """
starttime = time.time()
if __name__ == '__main__': 
    print("Processadores que serão usados = %d " % int(multiprocessing.cpu_count()))
    filenames = [i for i in range(n_runs)]
    p = Pool(int(multiprocessing.cpu_count()))
    p.map(main, filenames)
    
# calculando tempo para rodar programa
endtime = time.time()
tempo = (endtime - starttime)/3600
print(tempo, 'horas para rodar este programa')