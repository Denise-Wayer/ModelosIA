from tkinter import Y
import pandas as pd
import matplotlib.pyplot as plot
import seaborn as sea
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

def leituraTabela():
    #lê os dados da tabela 
    global tabela 
    tabela = pd.read_csv("advertising.csv")
    print(tabela.corr())#calcula a correlação entre as linhas e colunas da tabela 
    sea.heatmap(tabela.corr(), cmap='mako', annot=True)#gera o grafico
    plot.show()#exibe o grafico
    tabela

def treinamentoIA(tabela):
    #y quem voce quer prever(vendas) e x é quem voce vai usar para fazer a previsão neste caso os meios de anuncio
    x=tabela[["TV","Radio","Jornal"]]
    y=tabela["Vendas"]

    x_treino, x_teste, y_treino, y_teste = train_test_split(x,y, test_size=0.3, random_state=1)
    modelo_regressaolinear = LinearRegression()
    modelo_arvoredecisao = RandomForestRegressor()

    #treina as 2 i.a
    modelo_regressaolinear.fit(x_treino, y_treino)
    modelo_arvoredecisao.fit(x_treino, y_treino)
    #fazer previsao dos testes
    previsao_regressaolinear=modelo_regressaolinear.predict(x_teste)
    previsao_arvoredecisao = modelo_arvoredecisao.predict(x_teste)
    #vai fazer o calculo de rquadrado e trazer o valor dos testes comparando os modelos de arvore de decidao e 
    print(r2_score(y_teste, previsao_regressaolinear))
    print(r2_score(y_teste, previsao_arvoredecisao))
    
leituraTabela() 
treinamentoIA(tabela)
   