import json
def R(id,x,y,w,h,t,bg,fs=22):
    return {"type":"rectangle","id":id,"x":x,"y":y,"width":w,"height":h,"backgroundColor":bg,"fillStyle":"hachure","roughness":2,"roundness":{"type":3},"label":{"text":t,"fontSize":fs,"fontFamily":1}}
def T(id,x,y,t,fs,col="#1e1e1e"):
    return {"type":"text","id":id,"x":x,"y":y,"text":t,"fontSize":fs,"fontFamily":1,"strokeColor":col,"roughness":2}
def A(id,x,y,pts,label=None):
    a={"type":"arrow","id":id,"x":x,"y":y,"width":pts[-1][0],"height":pts[-1][1],"points":pts,"endArrowhead":"arrow","roughness":2}
    if label:a["label"]={"text":label,"fontSize":18,"fontFamily":1}
    return a
d1=[T("t",300,30,"流程①：从原始数据到全球归因地图（1982–2022）",40),
R("a",40,150,300,150,"ERA5 逐日 0.25°\n→ 逐月 0.5°\nVPD · SM · Tas · SWR","#a5d8ff"),
R("b",40,370,300,150,"PKU GIMMS-MODIS NDVI\n半月 → 逐月 0.5°\n剔除云/雪 QC","#b2f2bb"),
R("c",40,590,300,130,"MCD12C1 土地覆盖\n合并为 9 类植被","#ffec99"),
R("d",440,300,330,170,"STL 去季节/去趋势\n→ 距平残差\n仅保留生长季月份","#ffd8a8"),
A("e1",342,225,[[0,0],[96,120]]),A("e2",342,445,[[0,0],[96,-40]]),A("e3",342,655,[[0,0],[96,-190]],"掩膜"),
R("f",870,150,360,190,"逐格点 AutoML（FLAML）\nRF / XGBoost 自动选模调参\n3×3 邻域样本，80/20 划分","#d0bfff"),
R("g",870,430,360,170,"TreeSHAP |φ|\n→ 相对重要性 RI (%)\n全时段 / 1982–2001 / 2002–2022","#ffc9c9"),
A("e4",772,370,[[0,0],[96,-120]]),A("e5",1050,342,[[0,0],[0,86]]),
R("h",1320,150,360,170,"R² ≥ 0.2 筛选\n33,988 个格点\n面积加权统计","#99e9f2"),
R("i",1320,420,360,230,"三张地图\nRI_SM / RI_VPD\nD = RI_VPD − RI_SM\nδRI 与 ΔD 时段变化","#e9ecef"),
A("e6",1232,515,[[0,0],[86,-280]]),A("e7",1500,322,[[0,0],[0,96]]),
T("n1",440,560,"结果：SM 主导 42.3% 面积；VPD 重要性上升 71.8%",24,"#c2255c"),
T("n2",440,600,"ΔD>0（向 VPD 控制转移）62.9% 面积",24,"#c2255c"),
T("n3",300,790,"每个 0.5° 格点各训一个模型：144 进程 × 3 节点并行",28,"#1971c2")]
d2=[T("t",320,30,"流程②：AutoML + SHAP 两级归因解耦",40),
T("h1",80,110,"第一级：格点内——谁在驱动 NDVI？",26,"#1971c2"),
R("a",40,170,280,130,"X = SM, VPD,\nTas, SWR 距平","#a5d8ff"),
R("b",400,170,300,130,"FLAML AutoML\nrf ↔ xgboost\n时间预算 8 s/格点","#d0bfff"),
R("c",780,170,300,130,"TreeSHAP\n每个月每个因子 φ","#ffc9c9"),
R("d",1160,170,400,130,"RI = mean|φ| / Σ ×100\nD = RI_VPD − RI_SM\n前后时段 → δRI, ΔD","#ffec99"),
A("e1",322,235,[[0,0],[76,0]],"y=NDVI"),A("e2",702,235,[[0,0],[76,0]]),A("e3",1082,235,[[0,0],[76,0]]),
T("h2",80,380,"第二级：格点间——为什么这里会转向？",26,"#1971c2"),
R("f",40,440,420,150,"背景气候特征\nVPD/SM 均值、变异系数 CV\n及其时段变化 Δ","#b2f2bb"),
R("g",540,440,340,150,"0.5° → 2.5° 面积加权\n1,761 个块\n再跑一次 AutoML","#d0bfff"),
R("h",960,440,320,150,"全局 TreeSHAP\n重要性 + 依赖曲线","#ffc9c9"),
A("e4",462,515,[[0,0],[76,0]]),A("e5",882,515,[[0,0],[76,0]]),A("e6",1360,302,[[0,0],[-200,136]],"目标 y"),
R("i",1340,440,300,150,"SHAP 图三种风格\nstyle / alt / final","#99e9f2"),A("e7",1282,515,[[0,0],[56,0]]),
T("n1",60,650,"δRI_VPD：ΔVPD 均值贡献 28.1%（R²=0.55）",24,"#c2255c"),
T("n2",60,690,"ΔD：ΔVPD 21.3% · ΔCV_SM 17.7% · ΔSM 17.7%（R²=0.59）",24,"#c2255c"),
T("n3",220,780,"「解耦」= 先量化 SM vs VPD 的相对控制，再解释控制权为何转移",28,"#1971c2")]
json.dump({"d1":d1,"d2":d2},open("skel.json","w"),ensure_ascii=False)
