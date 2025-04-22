# Does an Early Lead Decide the Game?  
### A 2022 League of Legends Professional Match Analysis 📈🎮  
### Name: Thamim Hussain
### Email: thamimh@umich.edu

---

## 1 · Introduction  

**"League of Legends" (LoL)** is the world's most watched esport, drawing **30M+** peak viewers to its 2023 World Final.  
The dataset from **[Oracle's Elixir](https://oracleselixir.com/)** records every professional game: gold, experience, kills, objectives – all time-stamped.  

> **Central question**  
> *If a team holds an assist **and** experience lead at 10 minutes, how likely are they to win the entire match?*  

**Why you should care:**  
- **Strategic insight**: Coaches decide whether to play aggressively for early leads or draft for late-game scaling.  
- **Broadcast context**: Casters can instantly gauge win probabilities for viewers.  

| Item                      | Value                          |
|---------------------------|--------------------------------|
| Raw rows loaded           | **≈ 150K** player+team rows   |
| Rows after filtering      | **21312**         |
| Season analysed           | 2022 (patch 12.01)            |
| Relevant columns          | `goldat10`, `opp_goldat10`, `xpat10`, `opp_xpat10`, `golddiffat10`, `xpdiffat10`, `killsat10`, `opp_killsat10`, `assistsat10`, `opp_assistsat10`, `firstblood`, `result` |  

> **Column descriptions**  
> - **goldat10 / opp_goldat10**: Total team gold at 10:00 (ms).  
> - **xpat10 / opp_xpat10**: Cumulative experience points at 10:00.  
> - **golddiffat10 / xpdiffat10**: *Our* minus *their* resource counts (positive = advantage).  
> - **kills/assists at10**: Scoreboard differentials.  
> - **firstblood**: Whether this team secured first kill.  
> - **result**: `True` = win, `False` = loss.  

---

## 2 · Data Cleaning & Exploratory Data Analysis (EDA)  

### 2.1 Cleaning Steps 🔧  
1. **Team-row isolation**: Kept rows where `position == 'team'` *or* `participantid ∈ {100, 200}`; removes per-player duplicates.  
2. **Type coercion**: Converted early-game numeric columns to `float`; invalid strings → `NaN`.  
3. **Derived diffs**: If missing, created `golddiffat10 = goldat10 − opp_goldat10`, same for XP.  
4. **Feature engineering**: Added `killsdiff10`, `assistsdiff10` for modelling; cast `firstblood → bool`.  
5. **Row drop**: Removed teams with missing gold/XP diff (≈ **<<<N_DROPPED>>>** rows, < x% of total).  

**Head of the cleaned dataframe (5 rows)**  

| gameid                | side   | result   | golddiffat10 | xpdiffat10 | killsdiff10 | assistsdiff10 |
|-----------------------|--------|----------|-------------:|-----------:|------------:|--------------:|
| ESPORTSTMNT01_2690210 | Blue   | False    |         1523 |        137 |           3 |             5 |
| ESPORTSTMNT01_2690210 | Red    | True     |        -1523 |       -137 |          -3 |            -5 |
| ESPORTSTMNT01_2690219 | Blue   | False    |        -1619 |      -1586 |          -2 |            -2 |
| ESPORTSTMNT01_2690219 | Red    | True     |         1619 |       1586 |           2 |             2 |
| ESPORTSTMNT01_2690227 | Blue   | True     |         -103 |        813 |          -1 |            -1 |

---

### 2.2 Univariate Analysis  

<iframe src="assets/gold_diff_hist.html" width="800" height="410" frameborder="0"></iframe>  
Gold diff @ 10 min is nearly normal with μ ≈ 0 and σ ≈ 1.8k gold; extreme early snowballs (>5k) are rare.  

---

### 2.3 Bivariate Analysis  

<iframe src="assets/gold_xp_scatter.html" width="900" height="450" frameborder="0"></iframe>  
The diagonal trend shows gold and XP leads move together. Orange (wins) dominate the upper-right quadrant, supporting our hypothesis.  

---

### 2.4 Interesting Aggregates  

**Win-rate by binned gold lead:**  

| gold_bin          | Win Rate |
|-------------------|----------|
| (-1885, -942]     | 0.25     |
| (0, 942]          | 0.60     |
| (2828, 3771]      | 0.92     |
| ...               | ...      |

<iframe src="assets/winrate_by_gold_bin.html" width="800" height="350" frameborder="0"></iframe>  
Crossing **+1k gold** flips the matchup above **60%** win chance; beyond **+5k**, victories are almost guaranteed.  

---

### 2.5 Imputation  
We did not impute: missing early-game stats only occur in “remake” or prematurely terminated games and were dropped to avoid skewing results.  

---
## 3 · Framing the Prediction Problem  

- **Task type**: Binary classification (win vs loss).  
- **Response variable**: `result`, as it is the official match outcome recorded by tournament servers.  
- **Feature timing**: Only metrics known at 10 min (no mid- or late-game leaks).  
- **Evaluation metric**: ROC AUC – robust to class imbalance and grades ranking quality of predicted probabilities.  

---

## 4 · Baseline Model 🛫  

* **Algorithm**: Logistic Regression (interpretable linear classifier).  
* **Features (total = 2)**  

| Name              | Type          | Encoding         | Rationale                                  |
|-------------------|---------------|------------------|--------------------------------------------|
| `assistsdiffat10` | Quantitative  | Standard-scaled  | Measures supportive pressure and skirmish tempo. |
| `xpdiffat10`      | Quantitative  | Standard-scaled  | Captures lane dominance and level advantages. |

*There are 0 ordinal and 0 nominal predictors, so no one-hot encoding was required.*  

* **Model pipeline**: `StandardScaler ➜ LogisticRegression(max_iter=1000, C=1.0)`  
* **Train / test split**: 75/25 stratified by outcome.  

| Metric      | Value   |
|-------------|---------|
| **Accuracy**| 0.686   |
| **ROC AUC** | 0.750   |

**Is the model "good"?**  
A naïve coin-flip would score 0.50 accuracy and 0.50 AUC. Hitting 0.75 AUC means the model ranks a random win higher than a random loss 3 times out of 4 – respectable for only two features, but still misclassifies ~31% of held-out games. The linear boundary also assumes additive effects; real games likely exhibit non-linear interactions, so we expect headroom for improvement.  

---

## 5 · Final Model 🚀  

| Added feature     | Type          | Why it should help **before** seeing results            |
|--------------------|---------------|---------------------------------------------------------|
| `golddiffat10`     | Quantitative  | Gold directly buys combat stats; even a small 200g lead can secure first dragon or Herald. |
| `killsdiff10`      | Quantitative  | A kill grants 300g + tempo (death timer, map pressure) not fully reflected by assists alone. |
| `firstblood`       | Nominal (binary) | First Blood awards extra 400g and often snowballs lanes; encoded as 0/1. |

Features are scaled through `StandardScaler` (numeric) while `firstblood` is already binary (no further encoding).  

* **Algorithm**: Gradient Boosting Classifier (200 shallow trees, depth 2). Tree ensembles capture threshold effects (e.g., a 1k gold swing matters more than a 100g swing) and non-linear interactions.  
* **Hyper-parameter search**: GridSearchCV (5-fold) over `n_estimators {200,400}`, `learning_rate {0.03,0.05,0.08}`, `max_depth {2,3}`. Best = **200 trees, lr 0.03, depth 2**.  

| Model     | Accuracy | ROC AUC |
|-----------|----------|---------|
| Baseline  | 0.686    | 0.750   |
| **Final** | 0.706    | 0.777   |

**Why performance improved (data-generating perspective)**  
Gold is the universal currency for item power spikes; including `golddiffat10` lets the model separate teams that traded kills for farm from those that did not. `killsdiff10` captures raw elimination dominance, orthogonal to assists. `firstblood` encodes an early discrete snowball event worth ~400g that can’t be inferred solely from continuous differentials.  

Tree ensembles exploit interactions like "XP lead **and** firstblood" being disproportionately strong, something the linear baseline cannot model. The resulting **+0.027 AUC** and **+0.02 accuracy** confirm these early-game signals sharpen win-probability estimates.  

--- 