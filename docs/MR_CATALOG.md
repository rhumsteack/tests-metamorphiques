# MR Catalog (MNIST Pilot)

## MR1 - Identity

- Relation: `f(x) == f(x_clone)` en logits a tolerance tres faible
- But: verifier la stabilite deterministe minimale

## MR2 - Double Flip

- Relation: `f(x) ~= f(flip(flip(x)))`
- But: verifier une transformation reversible

## MR3 - Rotate then Inverse Rotate

- Relation: `f(x) ~= f(rotate(rotate(x, +a), -a))`
- But: detecter des anomalies d'interpolation/padding

## MR4 - Brightness Stability

- Relation: top-1 majoritairement conserve sous variation de luminosite
- But: evaluer robustesse a de petites variations photometriques

## MR5 - Blur Stability

- Relation: top-1 majoritairement conserve sous flou gaussien leger
- But: verifier robustesse aux perturbations faibles

## MR6 - Single vs Batch Consistency

- Relation: logits d'un sample seul = logits du meme sample dans un batch
- But: detecter une incoherence de pipeline batch
