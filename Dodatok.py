# -*- coding: utf-8 -*-
"""Системний аналіз поширення дезінформації у соціальних мережах."""

import os, warnings, json
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import differential_evolution
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import (RandomForestClassifier,
                              GradientBoostingClassifier)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
    recall_score, f1_score, roc_curve, auc)

# ---- 1. Завантаження датасету ----
fake = pd.read_csv("Fake.csv"); fake["label"] = 1
true = pd.read_csv("True.csv"); true["label"] = 0
df = pd.concat([fake, true], ignore_index=True)
df["date"] = pd.to_datetime(df["date"].astype(str).str.strip(),
    format="%B %d, %Y", errors="coerce")
df = df.dropna(subset=["date", "text"]).reset_index(drop=True)

# ---- 2. Базова модель (три компартменти) ----
def base_model(y, t, beta, gamma):
    S, I, R = y; N = S + I + R
    dS = -beta * S * I / N
    dI =  beta * S * I / N - gamma * I
    dR =  gamma * I
    return [dS, dI, dR]

# ---- 3. Розширена модель (чотири компартменти) ----
def extended_model(y, t, beta, b, rho, eps, p, l, gamma_i=0.05):
    S, E, I, Z = y; N = S + E + I + Z
    dS = -beta*S*I/N - b*S*Z/N
    dE = (1-p)*beta*S*I/N + (1-l)*b*S*Z/N - rho*E*I/N - eps*E
    dI = p*beta*S*I/N + rho*E*I/N + eps*E - gamma_i*I
    dZ = l*b*S*Z/N + gamma_i*I
    return [dS, dE, dI, dZ]

# ---- 4. Порівняння моделей класифікації ----
sample = df.sample(20000, random_state=42)
Xtr, Xte, ytr, yte = train_test_split(sample["text"],
    sample["label"], test_size=0.25, stratify=sample["label"],
    random_state=42)
vec = TfidfVectorizer(max_features=8000, ngram_range=(1,2),
    stop_words="english", min_df=5, max_df=0.9)
Xtr_v = vec.fit_transform(Xtr); Xte_v = vec.transform(Xte)

models = {
    "Logistic Regression": LogisticRegression(C=4.0,
         max_iter=1000, solver="liblinear"),
    "Naive Bayes":         MultinomialNB(alpha=0.1),
    "Random Forest":       RandomForestClassifier(
         n_estimators=120, max_depth=25, n_jobs=-1, random_state=42),
    "Gradient Boosting":   GradientBoostingClassifier(
         n_estimators=80, max_depth=4, random_state=42),
}
for name, mdl in models.items():
    mdl.fit(Xtr_v, ytr)
    pred = mdl.predict(Xte_v)
    print(name, "F1 =", f1_score(yte, pred))

# ---- 5. Калібрування на реальних даних ----
fake_daily = df[df.label==1].groupby(
    pd.Grouper(key="date", freq="D")).size()
idx = pd.date_range(fake_daily.index.min(),
    fake_daily.index.max(), freq="D")
fake_daily = fake_daily.reindex(idx, fill_value=0)
win_end = fake_daily.rolling(90).sum().idxmax()
start = win_end - pd.Timedelta(days=89)
obs = fake_daily.loc[start:win_end].values.astype(float)
obs_s = pd.Series(obs).rolling(7, min_periods=1,
    center=True).mean().values
obs_c = np.cumsum(obs_s); N = max(obs_c[-1]*2.5, 500)
t = np.arange(len(obs_c), dtype=float)

def loss(params):
    b, g = params
    sol = odeint(base_model, [N-1, 1, 0], t, args=(b, g))
    return np.mean((sol[:,1] + sol[:,2] - obs_c) ** 2)

res = differential_evolution(loss,
    [(0.01, 2.0), (0.005, 1.0)], seed=42, maxiter=120)
beta_est, gamma_est = res.x; R0 = beta_est / gamma_est
print(f"beta = {beta_est:.3f}, gamma = {gamma_est:.3f}, R0 = {R0:.2f}")

# ---- 6. Сценарне прогнозування ----
scenarios = {
    "Базовий":         dict(beta=0.45, b=0.10, rho=0.25,
                              eps=0.020, p=0.60, l=0.35, gamma_i=0.05),
    "Фактчекінг":      dict(beta=0.35, b=0.20, rho=0.18,
                              eps=0.012, p=0.50, l=0.55, gamma_i=0.12),
    "Вірусний вкид":   dict(beta=0.70, b=0.05, rho=0.35,
                              eps=0.040, p=0.75, l=0.20, gamma_i=0.03),
    "Медіа-грамотність":dict(beta=0.28, b=0.25, rho=0.15,
                              eps=0.010, p=0.40, l=0.65, gamma_i=0.15),
}
N = 10000; t = np.linspace(0, 80, 400)
y0 = [N-15, 0, 10, 5]
for name, p in scenarios.items():
    sol = odeint(extended_model, y0, t,
          args=(p["beta"], p["b"], p["rho"], p["eps"],
                p["p"], p["l"], p["gamma_i"]))
    print(name, "I_max =", sol[:,2].max())

# ---- 7. Аналіз чутливості ----
betas = np.linspace(0.1, 0.8, 36)
gammas = np.linspace(0.02, 0.5, 36)
R0_grid = np.zeros((36, 36))
Imax_grid = np.zeros_like(R0_grid)
for i, g in enumerate(gammas):
    for j, b in enumerate(betas):
        R0_grid[i,j] = b / g
        sol = odeint(base_model, [N-10, 10, 0], t, args=(b, g))
        Imax_grid[i,j] = sol[:,1].max()

print("Готово.")
