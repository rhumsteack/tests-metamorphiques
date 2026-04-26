# Scripts de Démonstration

Ce répertoire contient 3 scripts pour **présenter le test metamorphique** à des public de niveaux différents.

## 1. `demo_enhanced.py` - Démo Normal + Challenges (Pour novices)

**Objectif**: Montrer les MRs qui **passent** dans un pipeline correct, + quelques cas extrêmes.

**Contenu**:
- 5 MRs normales (Identity, Double Flip, Rotation ±12°, Brightness +15%, Blur léger)
- 2 cas challenge (Rotation extreme ±45°, Blur agressif)
- **Tous les cas affichent des images côte-à-côte**
- Verdict: PASS ou FAIL avec couleur

**Utilisation**:
```powershell
.\.venv\Scripts\python.exe scripts/demo_enhanced.py
```

**Durée**: 5-7 minutes (7 fenêtres d'images)

**Bon pour**: Introduction e novices, montrer qu'un modèle robuste passe toutes les transformations normales et même extrêmes.

---

## 2. `demo_with_bug.py` - Bug Detection (Pour montrer l'utilité)

**Objectif**: Montrer comment les MRs **détectent les anomalies** dans le pipeline.

**Contenu**:
- 1 baseline: MR2 normal → PASS ✓
- 3 cas intentionnellement buggés:
  - Bug 1: Inversion de couleurs → **FAIL ✗** (anomalie détectée!)
  - Bug 2: Darkening extrême → PASS ✓ (modèle robuste)
  - Bug 3: Blur extrême → PASS ✓ (idem)
- **Chaque cas affiche images côte-à-côte avec status PASS/FAIL en couleur**

**Utilisation**:
```powershell
.\.venv\Scripts\python.exe scripts/demo_with_bug.py
```

**Durée**: 3-5 minutes (4 fenêtres d'images)

**Bon pour**: Démontrer l'**utilité pratique** - montrer qu'un bug dans le pipeline est detecté automatiquement sans avoir besoin d'une vérité absolue.

---

## 3. Tests Automatisés (Pour ingénieurs)

**Objectif**: Afficher l'exécution automatisée des MRs via PyTest.

**Utilisation**:
```powershell
# Tous les tests metamorphiques
.\.venv\Scripts\python.exe -m pytest tests/metamorphic -m metamorphic -v

# Un MR spécifique
.\.venv\Scripts\python.exe -m pytest tests/metamorphic/test_mnist_mrs.py::test_mr1_identity_logits_stable -v
```

**Durée**: < 2 secondes pour tous les 6 tests

**Bon pour**: Montrer l'automatisation en CI/CD, intégration dans pipeline de test.

---

## Format de Présentation Recommandé (15 min)

```
Intro théorique (2 min) ......... docs/INTRO_NOVICES.md
  ↓
Demo Normal (5-7 min) .......... demo_enhanced.py
  ↓
Demo Bug-Detection (3-5 min) ... demo_with_bug.py
  ↓
Tests Auto (< 1 min) ........... pytest (optionnel)
  ↓
Q&A + Reflexion (5 min)
```

---

## Notes Techniques

- **Toutes les démos utilisent le même modèle MNIST** (poids pré-entraînés, déterministe avec seed=42)
- **Chaque fenêtre afiche**:
  - 2 images côte à côte
  - Prédictions + confiance (logits)
  - Titre avec MR et verdict
  - Fond blanc (PASS) ou rose (FAIL)
- **Les démos sont déterministes** → même résultats à chaque exécution
- **Peut tourner 100% sur CPU** en < 10 secondes au total

---

## Customization

Pour **ajouter d'autres MRs** ou **bugs**:
1. Ajouter une nouvelle fonction `demo_XXX()` dans le script
2. Utiliser `get_model_and_data()` pour charger le modèle/data
3. Appeler `plot_images_side_by_side()` pour afficher

Exemple:
```python
def demo_my_custom_mr():
    model, device, images, labels = get_model_and_data()
    source = images[0:1]
    follow_up = my_transform(source)
    # ... logique de test ...
    plot_images_side_by_side(...)
```
