# Metamorphic Testing Pilot (MNIST)

Projet pilote pour apprendre le **test metamorphique** sur un modele de classification d'images (MNIST), avec une approche pedagogique adaptee aux debutants.

## Ce qu'on veut tester

En ML, il est souvent difficile d'avoir un oracle parfait pour chaque entree. Le test metamorphique contourne ce probleme en testant des **relations** entre entrees liees:

- Image source `x`
- Image transformee `T(x)`
- Relation attendue entre les sorties `f(x)` et `f(T(x))`

Exemple simple: si on applique une transformation quasi-identite (double flip horizontal), la prediction doit rester identique.

## Demarrage rapide

1. Creer un environnement Python 3.10+
2. Installer les dependances

```bash
pip install -e .[dev]
pip install matplotlib  # Pour les démos visuelles
```

3. Lancer les **démos pédagogiques** (avec visualisation d'images)

```powershell
# Démo 1: MRs normales + cas extrêmes (5 min)
.\.venv\Scripts\python.exe scripts/demo_enhanced.py

# Démo 2: Détection de bugs (3 min)
.\.venv\Scripts\python.exe scripts/demo_with_bug.py
```

4. Executer les tests unitaires

```bash
pytest tests/unit -m unit
```

5. Executer les tests metamorphiques

```bash
pytest tests/metamorphic -m metamorphic
```

## Structure

- `docs/INTRO_NOVICES.md`: presentation claire pour novices
- `docs/DEMO_SCRIPT.md`: guide complet de présentation (15 min)
- `scripts/README.md`: guide des 3 scripts de démonstration
- `scripts/demo_enhanced.py`: démo normale (MRs passantes + cas extrêmes)
- `scripts/demo_with_bug.py`: démo détection de bugs
- `configs/mnist.yaml`: seed, taille des subsets, seuils des MRs
- `src/`: dataset, transforms, inference
- `models/`: petit CNN MNIST
- `tests/metamorphic/`: MRs automatisees avec PyTest
