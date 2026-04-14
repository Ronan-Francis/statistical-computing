# Statistical Computing Quiz Reference (Weeks 5-9)

## 0. BEFORE YOU START: Sanity Check
1.  **Is the response a count?**
    * Check `summary(df$y)`. If decimals exist, use Linear Model or Gamma.
2.  **Are categories treated as factors?**
    * Run `class(df$var)`. If it is "character," use `df$var <- as.factor(df$var)`.
3.  **Check for Zeros in Gamma:**
    * If $y$ contains zeros, add a constant: `df$y_adj <- df$y + 0.1`.
4.  **Check Overdispersion:**
    * For Poisson: `deviance(model) / df.residual(model)`. If $> 1.2$, switch to `glm.nb`.

---

## 1. DECISION LOGIC: Which Model to Use

**Is y a count (0, 1, 2...)?**
* **Mean ≈ Variance:** Poisson
* **Variance > Mean (Overdispersed):** Negative Binomial

**Is y continuous and positive?**
* **Symmetric:** Linear Model (`lm`)
* **Right-skewed:** Gamma GLM

**Are there multiple response variables (e.g., eruptions AND waiting)?**
* **Multivariate Mixture:** Use `mvnormalmixEM`.

**Hidden Structures?**
* **Unlabeled Groups (Histogram):** Normal Mixture (`normalmixEM`).
* **Unlabeled Slopes (Scatterplot):** Mixture of Regressions (`regmixEM`).

---

## 2. GLM Fitting and Extensions

### Weighted Models (Important for Insurance/Rates)
If you have a "Holders" or "Exposure" variable, you can use it as an offset or as weights.
```r
# Gamma with Weights (e.g., size of the group)
fit_gamma_w <- glm(y ~ x, family = Gamma(link = "log"), data = df, weights = exposure)
```

### Coefficient Interpretation (Log-Link)
**Rule:** `exp(coef(fit))` for Poisson, NegBin, Gamma, and Logistic.
* **Unit Increase:** `exp(beta) = 1.10` → 10% increase per 1 unit of X.
* **Large Step (e.g., 100 units):** `exp(coef(fit)["x"] * 100)`

---

## 3. Model Comparison (AIC/BIC)

### The $k=1$ Baseline
When testing mixtures, always calculate the $k=1$ (standard normal) case for your comparison table.
```r
# k = 1 (Standard Normal)
loglik1 <- sum(dnorm(data, mean(data), sd(data), log = TRUE))
aic_k1  <- -2 * loglik1 + 2 * (3 * 1 - 1) 
```
* **Lower AIC/BIC wins.** BIC is stricter for complex models.

---

## 4. Density Estimation (KDE)

### ggplot2 Syntax
If the quiz asks for a "modern" plot:
```r
ggplot(df, aes(x = var)) +
  geom_histogram(aes(y = after_stat(density)), bins = 15) +
  geom_density(kernel = "gaussian", bw = "ucv", color = "blue") # Cross-validation
```

### Visualizing Large Data
If $N > 5000$, standard scatterplots are unreadable. Use **Hexbins**.
```r
library(hexbin)
hexbinplot(y ~ x, data = FC, main = "2D Histogram")
```

---

## 5. Mixture Models

### Multivariate Mixtures (e.g., Old Faithful)
Used when you have two variables that cluster together.
```r
library(mixtools)
fit_mv <- mvnormalmixEM(df_2d, k = 2)
plot(fit_mv, whichplots = 2) # Density contours
```

### Manual BIC for Mixtures
* **1D Normal Mixture:** $p = 3k - 1$
* **2D Multivariate Mixture:** $p = 6k - 1$
* **Regression Mixture:** $p = (d + 2)k - 1$ (where $d$ is num of predictors)

---

## 6. Regression Diagnostics & Filtering

### High Leverage Points
Points with high `hatvalues` pull the line toward them. You may need to filter them to see the "true" trend.
```r
lev <- hatvalues(model)
high_lev <- lev > (2 * mean(lev)) # Or a fixed threshold like 0.08

# Refit without high leverage points
model_clean <- lm(y ~ x, data = df[!high_lev, ])
```

---

## 7. Custom MLE & Optimization

### Simple Parameter Estimation (`optim`)
For finding a single rate (like a Poisson $\lambda$) without a full GLM:
```r
loglik_pois <- function(lambda) { -sum(dpois(data, lambda, log = TRUE)) }

optim(par = mean(data), fn = loglik_pois, method = "Brent", lower = 0, upper = 1000)
```

---

## 8. Quick Interpretation Templates

* **Confidence Intervals:** "The 95% CI for $\beta_1$ is $[a, b]$. Since this interval does not contain the null value (e.g., 0 or 0.02), we reject $H_0$ at the 5% level."
* **High Leverage:** "Observation X has a leverage of [val], which exceeds the threshold. Removing it changed the slope from [old] to [new], suggesting it was an influential outlier."
* **KDE Bandwidth:** "The `ucv` (cross-validation) bandwidth is smaller than `nrd0`, revealing a multimodal structure that the rule-of-thumb bandwidth smoothed over."
* **Mixture Assignment:** "Based on posterior probabilities, [N] observations were assigned to Component 1, which aligns with the [Factor] group in the original data."
