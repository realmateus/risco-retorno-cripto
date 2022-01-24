import pandas as pd
import numpy as np
import matplotlib as fig
from datetime import date
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import plotly.express as px
import scipy.optimize as solver
import seaborn as sns
%matplotlib inline

plt.style.use('seaborn')

pd.core.common.is_list_like = pd.api.types.is_list_like

mudanca_proporcional=pd.read_excel('moedas.xlsx',sheet_name='NORMALIZAR')
mudanca_proporcional.head(30)

moedas = ["ADA", "BNB", "BTC", "DOGE", "DOT", "ETH","XRP"]

fig=px.line(mudanca_proporcional,x='Date',y=mudanca_proporcional.columns[1:8])

fig.update_layout(
    title="VARIAÇÃO MOEDAS",
    xaxis_title="DATA",
    yaxis_title="VARIAÇÃO",
    legend_title="CRIPTOS",
)

fig.show()

n=len(mudanca_proporcional)
prec=mudanca_proporcional.drop(['Date'],axis=1)
ri=prec/prec.shift(1)-1
mi=ri.mean().values*234
sigma=ri.cov()*234
print('MATRIZ DE COVARIÂNCIA DA CARTEIRA')
print(sigma)

ax = sns.heatmap(sigma,annot=True,fmt=".3f")

vet_R=[]
vet_Vol=[]
for i in range(100000):
    w=np.random.random(len(moedas))
    w=w/np.sum(w)
    retorno=np.sum(w*mi)
    risco=np.sqrt(np.dot(w.T,np.dot(sigma,w)))
    vet_R.append(retorno)
    vet_Vol.append(risco)

fig2=px.scatter(x=vet_Vol,y=vet_R,color=vet_R)
fig2.show()

def f_obj(peso):
    return np.sqrt(np.dot(peso.T,np.dot(sigma,peso)))

x0=np.array([1.0/(len(moedas)) for x in range(len(moedas))])

bounds=tuple((0,1) for x in range(len(moedas)))

faixa_ret=np.arange(-0.05,0.5, .001)

risk=[]

for i in faixa_ret:
    constraints=[{'type':'eq','fun':lambda x:np.sum(x)-1},
    {'type':'eq','fun':lambda x:np.sum(x*mi)-i}]

    outcome=solver.minimize(f_obj,x0,constraints=constraints,bounds=bounds,method='SLSQP')

    risk.append(outcome.fun)

fig3=px.line(x=risk,y=faixa_ret)
fig3.show()

print(outcome['x'])

def estatistisca_port(peso):
    peso=np.array(peso)
    ret_ot=np.sum(peso*mi)
    risco_ot=np.sqrt(np.dot(peso.T,np.dot(sigma,peso)))
    return np.array([ret_ot,risco_ot])

for i in faixa_ret:
    constraints=[{'type':'eq','fun': lambda x:sum(x)-1}]
    outcome=solver.minimize(f_obj,x0,constraints=constraints,bounds=bounds,method='SLSQP')
    risk.append(outcome.fun)

ret_ot,vol_ot=estatistisca_port(outcome['x'])
print('RETORNO ÓTIMO:',str((ret_ot*100).round(3))+'%')
print('VOLATILIDADE ÓTIMA:',str((vol_ot*100).round(3))+'%')

import plotly.graph_objects as go
fig3 = go.Figure(data=fig2.data + fig3.data)
fig3.update_layout(title="RISCO ESPERADO X RETORNO ESPERADO", xaxis_title="RISOC", yaxis_title="RETORNO")
fig3.show()
