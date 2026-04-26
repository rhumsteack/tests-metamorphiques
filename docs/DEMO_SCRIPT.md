# Demo 10 Minutes (Novices)

## Objectif demo

Montrer la logique `source -> follow-up -> relation attendue` sur MNIST **avec visualisation des images**, puis montrer comment les MRs **détectent les bugs**.

## Etapes

### 1. Présentation théorique (2 min)
Lire `docs/INTRO_NOVICES.md`, couvrir:
- Problème de l'oracle en ML
- Logique source -> transform -> relation attendue
- Exemples concrets

### 2. Démo normale (5 min)
Affiche les MRs qui **passent** grace à un pipeline correct:

```powershell
.\.venv\Scripts\python.exe scripts/demo_enhanced.py
```

Montre **7 scénarios** côte à côte (images visibles):
- 5 MRs normales (Identity, Double flip, Rotation, Brightness, Blur) → **PASS ✓**
- 2 cas challenge (Rotation extrême, Blur agressif) → **PASS ✓** (modèle robuste)

Chaque fenêtre affiche:
- Image source et transformée côte à côte
- Prédiction et confiance
- Verdict (PASS ou FAIL)

### 3. Démo "Bug Detection" (3 min bonus)
Montre comment les MRs **détectent les anomalies**:

```powershell
.\.venv\Scripts\python.exe scripts/demo_with_bug.py
```

Affiche:
1. **Baseline**: MR2 normal → PASS ✓
2. **Bug 1**: Inversion de couleurs → **FAIL ✗** (anomalie détectée!)
3. **Bug 2**: Darkening extrême → PASS ✓ (modèle robuste ce cas)
4. **Bug 3**: Blur extrême → PASS ✓ (idem)

Le Bug 1 montre clairement: **quand une anomalie casse la relation MR, on la détecte automatiquement**.

### 4. Exécution des tests automatisés (optionnel, 5 min bonus)
Pour montrer l'automatisation CI:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/metamorphic -m metamorphic -v
```

## Conseils presentation

- **Commencer par la démo visuelle normale** (images parlent mieux que mots)
- Pointer chaque transformation appliquée
- Souligner: "On teste une **relation**, pas une valeur exacte"
- **Montrer la démo bug-detection** juste après → passe d'un coup la notion _pourquoi c'est utile_
- Terminer par: "Ça marche sur vrais modèles (MobileNet, détection, segmentation...)"

## Timing suggeré

```
Intro théorique ........... 2 min
Demo normale .............. 5 min  (5 démos + 2 challenges)
Demo bug-detection ........ 3 min  (1 baseline + 3 bugs)
Q&A + reflex .............. 5 min
────────────────────────────────
Total ..................... 15 min (format atelier rapide)
```
