# Gender Gaps in AI Risk Perception — Python Extension of Borwein et al. (2026)

An independent replication and NLP extension of:

> Borwein, Magistro, Alvarez, Bonikowski & Loewen (2026). "Explaining Women's Skepticism Toward Artificial Intelligence." *PNAS Nexus*, 5, pgaf399.

---

## Two headline findings

**1. Epistemic asymmetry**
Women are 11.3 percentage points more likely than men to express uncertainty about AI's *benefits* (OR = 1.91, p < .0001), but only 4.6 percentage points more likely on AI's *risks* (OR = 1.54, p < .001). The benefit gap is **2.4× larger** than the risk gap. Women are not uniformly more uncertain — they are specifically more uncertain about what AI will do *for* them, not what it will do *to* them.

**2. Language attenuates the gender gap**
On a matched sample of N = 3,049, adding language features (BERTopic topic probabilities + VADER sentiment + uncertainty score) to the risk perception model reduces the gender coefficient from β = +0.298 (p = 0.019) to β = +0.181 (p = 0.214) — a **39.3% attenuation**. Gender becomes statistically non-significant once language is controlled. Once you know how someone talks about AI, knowing their gender adds nothing significant to predicting their risk score.

---

## Data

All data comes from the original paper's public release on Harvard Dataverse:

**DOI: https://doi.org/10.7910/DVN/LNFLY5**

Download the following files and place them in the same directory as the notebooks:

| File | Description |
|---|---|
| `AI.csv` | Main survey dataset (N = 3,049, US + Canada) |
| `dat_benefits_text.csv` | Open-ended responses: "What are the biggest benefits of AI?" |
| `dat_risks_text.csv` | Open-ended responses: "What are the biggest risks of AI?" |

The data files are not included in this repository. They belong to the original authors and are available under the terms of the Harvard Dataverse release.

---

## Notebooks — run in this order

### Phase 1 — Replication

#### `01_EDA_AI.ipynb`
Exploratory data analysis. Establishes the raw gender gap (women = 4.83, men = 4.43 on the 0–10 risk scale), the 2×2 interaction between gender and risk orientation, and subgroup t-tests. Confirms the paper's descriptive findings in Python before any modelling.

#### `02_OLS_Risk_Perception.ipynb`
Three-model weighted least squares regression predicting `risks_AI_avg`. Replicates the paper's Table 2. Key result: baseline gender coefficient β = 0.416 (p < .001) in Model 1. Includes VIF checks (all < 5), Breusch-Pagan heteroskedasticity test, HC3 robust standard errors, and a gender × risk-orientation interaction model. Model 2 adds `objective_threat` (automation exposure, N = 1,600). Model 3 restricts to manipulation-check passers.

#### `03_Ordinal_Logit_Support.ipynb`
Ordered logistic regression predicting `support_company` (1–5 Likert scale). Replicates the paper's support models. Key result: gender OR = 0.788 (p = .001) — women are 21% less likely than men to be in a higher support category after controlling for risk perception. Includes McFadden R², proportional odds assumption check, and predicted probability plots.

---

### Phase 2 — Extension into text analysis

#### `04_STM_Replication.ipynb`
Replicates the paper's Structural Topic Model (STM) using Python's LDA and NMF at K = 15 topics on both open-ended datasets. Produces the equivalent of the paper's Figure 5 — gender × topic prevalence dot plot. Confirms the paper's finding that "don't know" topics are more prevalent in women's responses. Adds bigram analysis showing women more frequently use "lose jobs", "taking jobs", "job loss"; men more frequently use "decision making", "self aware". First appearance of the epistemic asymmetry signal.

#### `05_Epistemic_Gap_Analysis.ipynb`
Isolates epistemic uncertainty as a first-class binary variable rather than one topic among fifteen. Classifies every response as `uncertain`, `substantive`, `too_short`, or `missing`. Computes uncertainty rates, odds ratios, chi-square tests, and Cohen's h separately for the benefits and risks questions. Establishes the 2.4× asymmetry finding. Builds three mediator variables: `benefit_uncertainty`, `risk_uncertainty`, and `unc_index`.

#### `06_Sentiment_Framing.ipynb`
Primary frame classifier assigning each substantive response to its dominant thematic category (job displacement, governance/bias, privacy/security, existential, misinformation, social/human loss). Log-odds analysis of term prevalence by gender. Key results: men use significantly more governance/accountability framing (p = .009) and efficiency/productivity framing (p = .002). Women use more concrete, personal-referent language (concreteness score: W = 0.115 vs M = 0.098, p = .009). Sentiment valence shows no significant gender difference — the gap is in framing, not emotional intensity.

#### `07_VADER_Sentiment-v2.ipynb`
Replaces the hand-built lexicon scorer with VADER (Valence Aware Dictionary and sEntiment Reasoner). Scores all four VADER components (compound, neg, pos, neu) by gender. Runs within-frame sentiment comparisons. Compares VADER directly against the hand lexicon on the same data. Key results: compound score shows no significant gender difference on either risks (p = .116) or benefits (p = .290). Both methods reach identical conclusions. The null on sentiment is the finding — gender differences in AI attitudes are about content and framing, not emotional tone.

---

### Phase 3 — The bridge

#### `08_BERTopic_Mediation_v2.ipynb`
The centrepiece of the project. Builds an end-to-end NLP pipeline:
1. Loads and merges all three datasets on respondent ID
2. Generates 768-dimensional RoBERTa-base embeddings via mean pooling (HuggingFace + PyTorch)
3. Fits BERTopic (UMAP + HDBSCAN) to discover topics organically — 40 topics versus the paper's fixed K = 15
4. Scores VADER sentiment and a hedge-term uncertainty count
5. Merges all text features into the main survey dataset at the respondent level
6. Fits a Random Forest predicting `risks_AI_avg` — text features account for the majority of feature importance; sentiment alone outranks gender
7. Runs OLS predicting `support_company` (R² = 0.146, risk perception β = −0.122***)
8. Runs Baron-Kenny mediation (Text → Risk → Support)
9. **Cell 39 — the core result:** clean matched attenuation test on N = 3,049. Gender β drops from +0.298 (p = .019) to +0.181 (p = .214) when language features are added. Attenuation = 39.3%. Gender becomes non-significant.

---

### Phase 4 — Hardening

#### `Epistemic_Gap_Improvements_v2.ipynb`
Addresses four methodological critiques raised in peer review:

| Issue | Finding | Impact |
|---|---|---|
| Lexicon bias: "no benefit" coded as uncertain | Split into `TRUE_UNCERTAIN` vs `SUBSTANTIVE_NEGATIVE` | Gap changes by −0.13pp. Finding robust. |
| Hedge language bias: "I think" scored as uncertain | Separated `epistemic_score` from `hedge_score` | Hedge gender gap not significant (p = .19). Not a speech style artefact. |
| Missing data: `objective_threat` missing 47.5% | MICE imputation via `IterativeImputer` (Random Forest, 10 iterations) | Gender β: 0.366 listwise → 0.330 imputed. Mild selection bias, conclusions unchanged. |
| Asymmetric index: equal-weights benefit and risk uncertainty | Added `unc_index_weighted` (benefit × 2.4, risk × 1.0) | Available as alternative summary statistic. |

Outputs `epistemic_features_v2.csv` with all improved variables for use in downstream notebooks.

#### `Mediation_Analysis_Epistemic_Uncertainty_v2.ipynb`
Formally tests whether the binary epistemic uncertainty variable alone mediates the gender gap — the narrow version of the language-mediates-gender hypothesis.

**Results:**
- Path a (female → benefit_uncertainty) in text CSVs: a = +0.113, p < .001 ✓
- Path a (female → benefit_uncertainty) in merged dataset: β = +0.006, p = .708 ✗
- Baron-Kenny Step 2 fails. Path a collapses to near-zero in the analytical sample.
- Bootstrap ACME: +0.0001, 95% CI [−0.004, +0.005] — not significant.

**Conclusion:** The binary "said don't know" variable does not mediate the gender gap on its own. The mechanism is the full semantic structure of language — which topics respondents raise, how they frame them, how their responses cluster in embedding space. The 39.3% attenuation in `08_BERTopic_Mediation_v2.ipynb` requires the full BERTopic representation, not the simple uncertainty flag.

---

## Citation

If you use this code, please cite the original paper:

```
Borwein, S., Magistro, B., Alvarez, R., Bonikowski, B., & Loewen, P. (2026).
Explaining Women's Skepticism Toward Artificial Intelligence.
PNAS Nexus, 5, pgaf399. https://doi.org/10.7910/DVN/LNFLY5
```

---

## Notes

- All notebooks use HC3 heteroskedasticity-robust standard errors unless otherwise noted
- The paper's baseline models use survey weights (WLS). The BERTopic and epistemic notebooks use unweighted OLS on the full sample — this produces a smaller baseline gender coefficient (~0.298) than the paper's weighted estimate (0.416), both of which are correct for their respective specifications
- `objective_threat` (TLRA automation exposure score) is missing for 47.5% of respondents. Notebooks that include it run on N ≈ 1,600; notebooks that exclude it run on N ≈ 3,049
- The 59% attenuation figure cited in early outreach was a cross-notebook comparison mixing different samples and covariate sets. The correct internally-valid estimate is **39.3%** from Cell 39 of `08_BERTopic_Mediation_v2.ipynb`
