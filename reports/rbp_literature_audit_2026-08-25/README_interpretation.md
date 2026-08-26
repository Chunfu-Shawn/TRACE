# RBP motif-effect literature interpretation

## Scope

The supplied `rbp_translation_effect_summary.significant.pdf` contains 166 displayed RBP labels. All 166 labels were queried online against Europe PMC/PubMed using both relevance-ranked and citation-ranked searches. The two result sets were deduplicated, producing 2,952 RBP–paper candidate records.

The automated table is a literature-screening inventory, not a claim that every retrieved paper proves the model effect. Candidate evidence levels are assigned only when the RBP symbol and a mechanism term occur in the same local title/abstract context. Full-text manual verification is still required before a reference is used in the manuscript.

## Interpretation of pre-mRNA evidence

Pre-mRNA splicing, intron retention and nuclear export are biologically relevant to translation because incompletely processed or nuclear-retained RNA is less available to cytoplasmic ribosomes. However, an RBP motif mutation in the current model does not simulate splicing. It only shows that the trained sequence model uses that motif as a predictor of the downstream CDS signal.

Therefore, splicing evidence should be described as support for an **RNA-maturation proxy hypothesis**, not as proof that the RBP directly changes translation. The hypothesis is strongest for factors that mechanistically connect splicing to export, surveillance or translation:

- **PABPN1** restricts export and translation of incompletely spliced RNA and participates in intron-retention surveillance (DOI: 10.1261/rna.079294.122).
- **YTHDC1** regulates mRNA splicing and m6A-dependent nuclear export (DOIs: 10.1016/j.molcel.2016.01.012; 10.7554/eLife.31311).
- **EIF4A3 and RBM8A** are exon-junction-complex components that connect prior splicing to export, nonsense-mediated decay and translation (DOIs: 10.1261/rna.5230104; 10.1101/gad.1389006).
- **SFPQ and NONO** are paraspeckle-associated factors linked to nuclear RNA organization and retention (DOIs: 10.1083/jcb.200906113; 10.1101/gr.087775.108).

## Recommended concordant main-text RBP-region pairs

Do not display every region for a selected RBP. Retain only the RBP-region pairs whose model direction agrees with the corresponding literature mechanism:

1. Direct translation or RNA-abundance controls: **YBX3–3UTR (+), SRSF9–CDS (+), SRSF1–CDS (+), MEX3C–3UTR (-), PUM2–3UTR (-), RBM4–5UTR (-)**.
2. RNA maturation, export and nuclear-retention controls: **PABPN1–3UTR (-), YTHDC1–CDS (+), SFPQ–3UTR (-), NONO–5UTR (-)**.

The first group has direct or comparatively close translation/stability evidence. The second group is mechanistically concordant but remains indirect evidence for an RNA-maturation proxy. EIF4A3 and RBM8A are excluded from the concordant main-text panel because EJC occupancy is primarily position/splicing-dependent and their model effects are regionally mixed.

## Important confounders

- CDS motif effects can be driven by codon identity, amino-acid changes, RNA structure or periodicity rather than RBP binding. Splicing-related CDS claims require synonymous motif disruption or an intron-containing minigene.
- lncRNA is not synonymous with unspliced or nuclear RNA. Many lncRNAs are capped, spliced and polyadenylated, and a subset is cytoplasmic. Nuclear/cytoplasmic fractionation and intron-retention annotations are needed to test the proposed mechanism.
- The 166 labels are not 166 independent motif signals. Ten duplicated effect-profile groups contain 23 labels. Examples include `CSDC2/CARHSP1`, `RBM3/CIRBP`, `EIF4A3/DDX19B`, `SAMD4A/SAMD4B`, and several Ensembl-placeholder labels. These should be collapsed by PWM or motif cluster before manuscript-level counting.
- Fourteen labels had no symbol-specific PubMed hit in the screened records. Most are Ensembl placeholders, read-through names or poorly characterized paralogs and should not be interpreted as negative biological evidence.
