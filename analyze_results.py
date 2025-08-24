#!/usr/bin/env python3
"""
analyze_results.py

Analisa o CSV gerado pelos experimentos (output do compara_todos_automatizado)
Gera:
 - resumo CSV (melhor configuração por modelo)
 - gráficos (boxplot por modelo, barra de melhores, heatmap normalizacao x modelo,
   boxplot/violin por ativacao para MLPs, scatter accuracy x tempo)
 - imprime top-K configurações globais

Uso:
  python analyze_results.py resultados_todos.csv --outdir analysis_out --topk 10

Dependências:
  pandas, numpy, matplotlib, seaborn
"""

import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
import os
from statsmodels.formula.api import ols
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

sns.set_theme(style='whitegrid', context='talk')

INPUT = 'resultados_todos_auto-sem-pca.csv'
OUTDIR = 'analysis_report-sem-pca-final'
os.makedirs(OUTDIR, exist_ok=True)

df = pd.read_csv(INPUT)
# normalize column names (as before)
df.columns = [c.strip() for c in df.columns]
df = df.rename(columns={c: c.strip().lower().replace(' ','_') for c in df.columns})

# ensure numeric
numcols = ['numero_treinamentos','tempo_execucao_em_segundos','media','minimo','maximo','mediana','desvio_padrao','eta','epocas']
for c in numcols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

# --- Best config por modelo
best_idx = df.groupby('modelo')['media'].idxmax()
best_by_model = df.loc[best_idx].reset_index(drop=True)
best_by_model.to_csv(os.path.join(OUTDIR,'best_config_by_model.csv'), index=False)

# --- Plot A: barras com erro
plt.figure(figsize=(9,5))
order = best_by_model.sort_values('media', ascending=False)['modelo']
sns.barplot(data=best_by_model, x='modelo', y='media', order=order, errorbar=None)
for i,row in enumerate(best_by_model.sort_values('media', ascending=False).itertuples()):
    plt.errorbar(i, row.media, yerr=row.desvio_padrao, fmt='none', ecolor='k', capsize=4)
plt.ylabel('Acurácia média (%)'); plt.title('Melhor configuração por modelo'); plt.xticks(rotation=45)
plt.tight_layout(); plt.savefig(os.path.join(OUTDIR,'fig_best_bar.png'), dpi=200); plt.close()

# --- Plot B: boxplots das execuções das melhores configs
keys = ['modelo','normalizacao','funcao_ativacao','optimizador','eta']
# build mask
mask = pd.Series(False, index=df.index)
for _, rc in best_by_model[keys].fillna('nan').iterrows():
    cond = True
    for k in keys:
        cond &= (df[k].fillna('nan') == rc[k])
    mask |= cond
subset = df[mask]
if not subset.empty:
    plt.figure(figsize=(10,6))
    sns.boxplot(x='modelo', y='media', data=subset, order=order)
    plt.title('Boxplot das execuções: melhores configs por modelo'); plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR,'fig_box_best_configs.png'), dpi=200); plt.close()

# --- Plot C: Pareto
plt.figure(figsize=(8,6))
plt.scatter(best_by_model['tempo_execucao_em_segundos'], best_by_model['media'], s=100)
for _, r in best_by_model.iterrows():
    plt.text(r.tempo_execucao_em_segundos*1.01, r.media, r.modelo)
plt.xlabel('Tempo (s)'); plt.ylabel('Acurácia média (%)'); plt.title('Pareto Acurácia x Tempo')
plt.grid(True); plt.tight_layout(); plt.savefig(os.path.join(OUTDIR,'fig_pareto.png'), dpi=200); plt.close()

# --- Plot D: ativações (MLPs)
mlp_sub = df[df['modelo'].isin(['mlp1h','mlp2h'])]
if ('funcao_ativacao' in df.columns) and (not mlp_sub.empty):
    plt.figure(figsize=(10,6))
    sns.violinplot(x='funcao_ativacao', y='media', hue='modelo', data=mlp_sub, split=True)
    plt.title('Ativações (MLP1H vs MLP2H)'); plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR,'fig_activation_mlp.png'), dpi=200); plt.close()

# --- Heatmap normalizacao x modelo
if 'normalizacao' in df.columns:
    pivot = df.pivot_table(index='modelo', columns='normalizacao', values='media', aggfunc='mean')
    plt.figure(figsize=(max(6,pivot.shape[1]*1.2), max(4,pivot.shape[0]*0.6)))
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='viridis')
    plt.title('Acurácia média por Modelo x Normalização'); plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR,'fig_heatmap_norm_model.png'), dpi=200); plt.close()

# --- Estatística: ANOVA/Tukey se aplicável (testa diferenças entre modelos)
if df['media'].notna().sum() > 10:
    try:
        model = ols('media ~ C(modelo)', data=df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        anova_table.to_csv(os.path.join(OUTDIR,'anova_modelo.csv'))
        tukey = pairwise_tukeyhsd(df['media'], df['modelo'])
        with open(os.path.join(OUTDIR,'tukey_modelo.txt'),'w') as f:
            f.write(str(tukey))
    except Exception as e:
        print('ANOVA/Tukey falhou:', e)

# --- Se tiver dados pareados (arquivo separado), rodar testes pareados (exemplo)
# [IMPLEMENTAÇÃO OPCIONAL: exige que você gere per-run CSVs com identicadores de run/seed]

print("Análise completa. Saída em:", OUTDIR)