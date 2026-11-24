import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
d=pd.read_csv("previsao_especialidade.csv",dtype={"cnesexecutante":str}); d["dia"]=pd.to_datetime(d.dia)
c=d[d.cnesexecutante=="2237571"]
top=c.groupby("especialidademae").y.sum().sort_values(ascending=False).index[:8]
INK,SEC,MUT,GRID,BG="#0b0b0b","#52514e","#898781","#e1e0d9","#fcfcfb"
plt.rcParams.update({"font.size":9,"axes.facecolor":BG,"figure.facecolor":BG,"axes.edgecolor":GRID,"xtick.color":MUT,"ytick.color":MUT,"text.color":INK})
fig,axs=plt.subplots(4,2,figsize=(14,12),sharex=True)
for ax,e in zip(axs.ravel(),top):
    x=c[c.especialidademae==e].set_index("dia")[["y","esperado","base_esp"]].resample("MS").sum()
    x=x.iloc[:-1] if x.index[-1].month==3 and x.index[-1].year==2026 and False else x
    ax.plot(x.index,x.y,color=INK,lw=2,label="Observado")
    ax.plot(x.index,x.base_esp,color="#898781",lw=1.3,ls=(0,(4,2)),label="Baseline (taxa 28d da especialidade)")
    ax.plot(x.index,x.esperado,color="#2a78d6",lw=1.8,label="Modelo por paciente (V1)")
    w=lambda m:(c[c.especialidademae==e].y-c[c.especialidademae==e][m]).abs().sum()/c[c.especialidademae==e].y.sum()
    ax.set_title(f"{e}  ·  WAPE diário V1 {w('esperado'):.2f} | base {w('base_esp'):.2f}",loc="left",fontsize=10)
    ax.grid(axis="y",color=GRID,lw=.7); ax.set_axisbelow(True)
    for s in ("top","right"): ax.spines[s].set_visible(False)
axs[0,0].legend(frameon=False,fontsize=8,labelcolor=SEC,loc="upper left")
fig.suptitle("CNES 2237571 — faltas por mês, por especialidade (8 com mais faltas), 2023-2026",x=0.07,ha="left",fontsize=13)
fig.tight_layout(rect=(0,0,1,0.97)); fig.savefig("grafico_especialidade_2237571.png",dpi=130,facecolor=BG)
