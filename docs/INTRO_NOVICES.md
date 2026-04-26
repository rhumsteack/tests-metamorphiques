# Introduction Novices

## 1. Pourquoi ce projet

En test classique, on compare le resultat avec une valeur attendue precise.
En ML, cette valeur attendue n'est pas toujours connue pour chaque cas. C'est le probleme de l'**absence d'oracle**.

Le test metamorphique repond a ce probleme en verifiant des **relations de coherence**.

## 2. Ce qu'on teste exactement

On ne teste pas "le modele est parfait".
On teste plutot:

- la stabilite du comportement sur des transformations controlees,
- la coherence des predictions entre entree source et entree transformee,
- la robustesse du pipeline (batch, preprocessing, inference).

## 3. Vocabulaire minimal

- `source` : entree originale `x`
- `follow-up` : entree transformee `T(x)`
- `MR` (Metamorphic Relation): relation attendue entre `f(x)` et `f(T(x))`

Exemple MR:
- Source: image d'un chiffre
- Follow-up: meme image apres double flip horizontal
- Attendu: les logits doivent rester quasi identiques

## 4. Critere de succes

Une MR est validee si la relation definie est respectee au-dessus d'un seuil.

Exemple:
- `match_rate >= 0.85` pour une transformation de luminosite

## 5. Message a retenir

Le test metamorphique n'a pas besoin d'un oracle parfait par image.
Il a besoin de **relations fiables**, explicites, et automatisees.
