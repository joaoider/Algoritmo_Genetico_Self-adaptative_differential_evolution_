# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 13:42:13 2020

@author: Joao
"""
import numpy as np
from uncertainties import ufloat
# from uncertainties.umath import *
from matplotlib import pyplot as plt

""" #################################### PARÂMETROS ######################################## """

time = 20.87554525739617   # tempo de duração do programa. copiar o resultado do programa anterior, FIT

n_runs = 40      # número de vezes que o programa irá rodar
popsize = 100        # tamanho da população aleatório inicial
gmax = 50000      # número de iterações

# dados do dispositivo
A = 0.0259
L = 0.144 
n = 8.*127.      # número de junções pn
Z = 0.0026      # figura de mérito da literatura

# Nome do arquivo com Medidas Experimentais
name = '1'

# Carrega o arquivo com os dados experimentais
data = np.loadtxt(name+'.txt', comments='#')
voltage  = data[:,0]   
current  = data[:,1]

# Calcula potência e resistência de carga com os dados experimentais
power = voltage * current
Load_resistance = voltage/(n*current)

# Equações a serem ajustadas do artigo 'TEG using effective material properties'
def IVW(par): # par = np.array([alpha, R, Th, Tc F, CR]) 
    # return I (Eq.34),  Vn (Eq.35),  Wn (Eq.36)    
    I = par[0]*(par[2]-par[3])/(Load_resistance+par[1]) 
    V = n*par[0]*(par[2]-par[3])*(Load_resistance/par[1])/((Load_resistance/par[1])+1.)
    return I, V, I*V

# Corrente de curto circuito 
def Isc(R, alpha, th1, tc1, vn):
    return (1./R)*(alpha*(th1-tc1)-vn/n)

# Tensão de circuito aberto
def Voc(R, alpha, th1, tc1, Ix):
    return n*(alpha*(th1-tc1)-R*Ix)

# Tensão de circuito aberto 2
def Voc2(par, Ix):
    return n*(par[0]*(par[2]-par[3])-par[1]*Ix)

# Calcula as equaçõçes referentes aos Modelos para I, V, W e R_L
# Calculo para qual das 40 runs?
def IVWR_mod(out):
    V_mod = np.linspace(0., Voc2(out,0.), 1000)
    R_L_mod = V_mod/((n*out[0]*(out[2]-out[3])/out[1])-V_mod/out[1])
    I_mod = out[0]*(out[2]-out[3])/(R_L_mod+out[1])
    W_mod = I_mod*V_mod
    return I_mod, V_mod, W_mod, R_L_mod

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

out = np.zeros((n_runs, 6)) # não entendo função zeros np

for j in range(n_runs):
    data = np.loadtxt(str('results_')+str(name)+"_"+str(j)+".txt")
    out[j,:]  = data[gmax-1, 2:8]  

I_fit, V_fit, W_fit = IVW(out[0,:])
I_mod, V_mod, W_mod, R_mod = IVWR_mod(out[0,:])

# Calcula os valores com os erros? não entendi.
alpha = ufloat(np.mean(out[:,0]), np.std(out[:,0]))
R = ufloat(np.mean(out[:,1]), np.std(out[:,1]))
th = ufloat(np.mean(out[:,2]), np.std(out[:,2]))
tc = ufloat(np.mean(out[:,3]), np.std(out[:,3]))
F = ufloat(np.mean(out[:,4]), np.std(out[:,4]))
CR = ufloat(np.mean(out[:,5]), np.std(out[:,5]))

index2 = np.argwhere(W_fit==np.max(W_fit))  # não entendo função argwhere
eta_c = 1. - tc/th   # eta de Carnot
rho = 4.*A/L*W_fit[index2[0,0]]/(n*(Isc(R,alpha,th,tc, 0.))**2)   # O que calcula???
T_medio = (th-tc)/2.   # temperatura média
eta_max = ((th-tc)/th)*((1.+Z*T_medio)**0.5-1.)/((1.+Z*T_medio)**0.5+tc/th) # eta máximo
Z_eff = (1./(T_medio))*((((1.+(eta_max*tc/(eta_c*th)))/(1.-eta_max/eta_c))**2)-1.) # figura de mérito efetiva

# Escreve os dados no arquivo
b = open(str('results_')+str(name)+"_"+"all_values.txt", "w")
b.write('runs = ' + str(n_runs)+'\n')
b.write('popsize = ' + str(popsize)+'\n')
b.write('gmax = ' + str(gmax)+'\n')
b.write('time(horas) = ' + str(time)+'\n')
b.write('F = ' + str(F)+'\n')
b.write('CR = ' + str(CR)+'\n')
b.write('Alpha = ' + str(alpha)+'\n')
b.write('R = '+str(R)+'\n')
b.write('T_h = '+str(th)+'\n')
b.write('T_c = '+str(tc)+'\n')
b.write('rho = '+str(rho)+'\n')
b.write('A corrente de curto circuito é: '+str(Isc(R,alpha,th,tc, 0.))+ '\n')  
b.write('A tensão de circuito aberto é: ' +str(Voc(R,alpha, th,tc,0.))+'\n')  
b.write('A potência máxima é %.5f W \n' %W_fit[index2[0,0]]) 
b.write('A tensão na potência máxima é %.5f V \n' %V_fit[index2[0,0]]) 
b.write('A corrente na potência máxima é %.5f A \n' %I_fit[index2[0,0]]) 
b.write('Effective Figure of Merit is: '+ str(Z_eff)+ '\n' )
b.write('Eta_c = ' + str(eta_c)+ '\n')
b.write('Eta_max = ' +str(eta_max)+ '\n')
b.close()

# Salva dados no arquivo
x = np.transpose(np.array([I_mod, V_mod, W_mod]))
np.savetxt(name+'_out_to_plot_final.dat', x, fmt='%1.4e',delimiter=' ', header='Imod Vmod Wmod')