import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
d=pd.read_csv("previsao_especialidade.csv",dtype={"cnesexecutante":str}); d["dia"]=pd.to_datetime(d.dia)
c=d[(d.cnesexecutante=="2237571")&(d.especialidademae=="ODONTO ENDODONTIA")]
w=c.set_index("dia")[["y","esperado","base_esp","ag_esp"]].resample("W").sum()
w=w[w.ag_esp>0]  # semanas com agenda
wp=lambda m:(w.y-w[m]).abs().sum()/w.y.sum()
print("semanas com agenda:",len(w),"| faltas",int(w.y.sum()),"| V1 %.0f | base %.0f"%(w.esperado.sum(),w.base_esp.sum()))
print("WAPE semanal V1 %.3f | base %.3f"%(wp("esperado"),wp("base_esp")))
print("corr V1 %.3f | base %.3f"%(np.corrcoef(w.y,w.esperado)[0,1],np.corrcoef(w.y,w.base_esp)[0,1]))
print("dentro de +-25%% do observado: V1 %.0f%% | base %.0f%%"%(100*((w.esperado/w.y.replace(0,np.nan)-1).abs()<=.25).mean(),100*((w.base_esp/w.y.replace(0,np.nan)-1).abs()<=.25).mean()))
w.round(2).to_csv("endodontia_2237571_semanal.csv")
INK,SEC,MUT,GRID,BG="#0b0b0b","#52514e","#898781","#e1e0d9","#fcfcfb"
plt.rcParams.update({"font.size":10,"axes.facecolor":BG,"figure.facecolor":BG,"axes.edgecolor":GRID,"xtick.color":MUT,"ytick.color":MUT,"text.color":INK})
fig,(a,b)=plt.subplots(2,1,figsize=(13,8),gridspec_kw={"height_ratios":[2,1]})
a.plot(w.index,w.y,color=INK,lw=2,marker="o",ms=3,label="Observado")
a.plot(w.index,w.base_esp,color=MUT,lw=1.4,ls=(0,(4,2)),label="Baseline (taxa 28d da especialidade)")
a.plot(w.index,w.esperado,color="#2a78d6",lw=1.9,label="Modelo por paciente (V1)")
a.set_ylabel("faltas / semana"); a.legend(frameon=False,labelcolor=SEC,ncol=3,loc="upper right")
a.set_title(f"Odonto endodontia, CNES 2237571 — faltas por semana (WAPE V1 {wp('esperado'):.2f} | base {wp('base_esp'):.2f})",loc="left",fontsize=12)
r=w.esperado-w.y
b.bar(w.index,r,width=5,color=np.where(r>=0,"#2a78d6","#eb6834"))
b.axhline(0,color=MUT,lw=.8); b.set_ylabel("V1 − observado")
b.set_title("Erro semanal do V1 (azul = superestimou, laranja = subestimou)",loc="left",fontsize=10,color=SEC)
for ax in (a,b):
    ax.grid(axis="y",color=GRID,lw=.7); ax.set_axisbelow(True)
    for s in ("top","right"): ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig("grafico_endodontia_semanal.png",dpi=130,facecolor=BG)
