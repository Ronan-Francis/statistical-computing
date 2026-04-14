# Statistical Computing Quiz Reference (Weeks 5-9)

---

## 0. BEFORE YOU START: Sanity Check
1. **Is the response a count?** Check `summary(df$y)`. If there are decimals, it's not a count—use a Linear Model or Gamma.
2. **Are categories treated as factors?** Run `class(df$var)`. If it says "character" or "numeric" (but means categories), use `as.factor()`.
3. **Check for Zeros in Gamma:** If $y$ contains zeros, `glm(..., family=Gamma)` will crash. Add a constant (e.g., `y + 0.1`).
4. **Check Overdispersion:** Always check `Residual Deviance / DF` for Poisson models. If $> 1.2$, switch to `glm.nb`.

---

## 1. DECISION LOGIC: Which Model to Use

***Look at the response variable (y) first.***

**Is y a count (0, 1, 2, 3...)?**
* **Residual Deviance ≈ Degrees of Freedom → POISSON**
```r
model_pois <- glm(y ~ x, data = df, family = "poisson")
summary(model_pois) # Check: Residual deviance / df.residual
```
* **Residual Deviance >> Degrees of Freedom → NEGATIVE BINOMIAL**
```r
library(MASS)
model_nb <- glm.nb(y ~ x, data = df)
summary(model_nb)
```

**Is y continuous and strictly positive (e.g. rainfall, hospital stay)?**
* **Symmetric → LINEAR MODEL (lm)**
```r
hist(df$y) # Check for bell shape
model_lm <- lm(y ~ x, data = df)
```
* **Right-skewed (long tail to the right) → GAMMA GLM**
```r
# If zeros exist, use: df$y_adj <- df$y + 0.1
model_gamma <- glm(y ~ x, data = df, family = Gamma(link = "log"))
```

**Does the histogram show multiple humps (modes) with no group labels?**
* **MIXTURE MODEL**
```r
library(mixtools)
mix_model <- normalmixEM(df$y, k = 2) # k = number of humps
plot(mix_model, which = 2)
```

**Does a scatterplot show distinct lines/slopes with no group labels?**
* **MIXTURE OF REGRESSIONS**
```r
library(mixtools)
reg_mix <- regmixEM(y ~ x, data = df, k = 2)
summary(reg_mix)
```

---

## 2. GLM Fitting and Interpretation

### Fitting Models
```r
# POISSON / NEG BINOMIAL (Counts)
fit_pois <- glm(y ~ x1, family = poisson, data = df)
fit_nb   <- glm.nb(y ~ x1, data = df)

# OFFSET (Rates: e.g., claims per policyholder)
fit_pois_offset <- glm(y ~ x1 + offset(log(exposure)), family = poisson, data = df)

# GAMMA (Positive, right-skewed)
df$y_adj <- df$y + 0.1  # Fix for zeros
fit_gamma <- glm(y_adj ~ x1, family = Gamma(link = "log"), data = df)

# LOGISTIC (Binary outcome 0/1)
fit_logit <- glm(y ~ x1, family = binomial, data = df)
```

### Coefficient Interpretation (Log-Link)
**Rule:** Always `exp()` coefficients for Poisson, NegBin, Gamma, and Logistic models.
```r
exp(coef(fit_model))
exp(coef(fit_model)["variable_name"])
```
* **exp(beta) = 1.15:** "A one-unit increase in X multiplies the response by 1.15 (a 15% increase)."
* **exp(beta) = 0.85:** "A one-unit increase in X multiplies the response by 0.85 (a 15% decrease)."

---

## 3. Model Comparison

### AIC / BIC
```r
AIC(model1, model2) # Lower = better
BIC(model1, model2) # Lower = better (stricter penalty for complexity)
```
*(Note: Do not directly compare AIC of a Gamma model to a Poisson/NegBin model due to different distributional assumptions and data modifications).*

### Likelihood Ratio Test (Nested Models)
```r
# Base R Anova
anova(simpler_model, fuller_model, test = "Chisq")
# Low p-value (< 0.05) = fuller model is significantly better

# Manual LRT
lrt_stat <- -2 * (logLik(simpler_model) - logLik(fuller_model))
lrt_pval <- pchisq(lrt_stat, df = df_difference, lower.tail = FALSE)
```

---

## 4. KDE and Nonparametric Classification

### KDE Bandwidths & Kernels
* **`bw = "nrd0"` (Default):** Smooth, good for general shape, but can "oversmooth" and hide bimodality.
* **`bw = "ucv"` (Cross-validation):** Jagged/noisier, but highly accurate for revealing true underlying "humps" or modes.
* **Gaussian Kernel:** Smooth curves.
* **Rectangular Kernel:** Produces "staircase" or histogram-style blocky curves.

### Full Classifier Template
$P(C_1 | x) = \frac{f_1(x)P(C_1)}{f_1(x)P(C_1) + f_0(x)P(C_0)}$

```r
lo <- min(df$x); hi <- max(df$x)

# 1. Estimate densities
f0 <- density(df$x[df$group == "no"],  from = lo, to = hi)
f1 <- density(df$x[df$group == "yes"], from = lo, to = hi)

# 2. Prior probability of class 1
p_c1 <- mean(df$group == "yes")

# 3. Plot posterior probability curve
plot(f1$x, f1$y * p_c1 / (f1$y * p_c1 + f0$y * (1 - p_c1)),
     type = "l", xlab = "x", ylab = "P(Class=1 | x)")
abline(h = 0.5, lty = 3)

# 4. Predict probability for a specific new value
val_f0 <- approx(f0$x, f0$y, xout = x_new)$y
val_f1 <- approx(f1$x, f1$y, xout = x_new)$y
prob_c1 <- (val_f1 * p_c1) / (val_f1 * p_c1 + val_f0 * (1 - p_c1))
```

---

## 5. Mixture Models

### 1D Normal Mixture
```r
library(mixtools)
fit_mix <- normalmixEM(df$y, k = 2)

# Parameters
fit_mix$lambda  # Mixing proportions (must sum to 1)
fit_mix$mu      # Component means
fit_mix$sigma   # Component standard deviations
fit_mix$loglik  # Log-likelihood

plot(fit_mix, whichplots = 2)
```

### Mixture of Regressions (regmixEM)
```r
fit_reg <- regmixEM(y = df$y, x = df$x, k = 2, arbvar = FALSE)

# Assign points to most likely component
assigned <- ifelse(fit_reg$posterior[, 1] > 0.5, "Comp 1", "Comp 2")
```

### Manual AIC/BIC Calculation (mixtools doesn't provide this)
```r
k <- 2         # number of components
n <- nrow(df)  # or length(x)

# PARAMETER COUNTS (p):
# 1D Normal Mixture: p = 3*k - 1
# 2D Bivariate Mixture: p = 6*k - 1
# Regression Mixture: p = (d + 2)*k - 1  (d = num predictors. If just y~x, p=3k-1)

p <- 3 * k - 1 

aic_val <- -2 * fit_mix$loglik + 2 * p
bic_val <- -2 * fit_mix$loglik + log(n) * p
```

---

## 6. Linear Regression & Interactions

### Fitting & Assumptions
```r
fit_lm <- lm(y ~ x, data = df)
fit_lm_noint <- lm(y ~ x - 1, data = df) # No intercept

# Assumptions check
plot(fit_lm) # Check Residuals vs Fitted & Normal Q-Q
hatvalues(fit_lm) > 0.08  # Identifies influential points (leverage)
```

### Manual Slopes from Interaction Model
```r
fit_interact <- lm(y ~ x * group, data = df)
cc <- coef(fit_interact)

# Base Group:
# Intercept = cc[1], Slope = cc[3]
abline(cc[1], cc[3], col = "red")

# Other Group:
# Intercept = cc[1] + cc[2], Slope = cc[3] + cc[4]
abline(cc[1] + cc[2], cc[3] + cc[4], col = "blue")
```

### Hypothesis Test on a Coefficient
```r
n          <- fit_lm$df.residual
theta_null <- 0.02          # H0 value
theta_hat  <- coef(fit_lm)["x"]
SE         <- summary(fit_lm)$coefficients["x", 2]

t_stat <- (theta_hat - theta_null) / SE
pval   <- 2 * pt(abs(t_stat), df = n, lower.tail = FALSE) # Two-sided
```

---

## 7. Custom Maximum Likelihood (MLE)

```r
library(stats4)
obs <- c(...)

# 1. Exponential
nll_exp <- function(lambda) { -sum(dexp(obs, rate = lambda, log = TRUE)) }
fit_mle <- mle(nll_exp, start = list(lambda = 1/mean(obs)))

# 2. Normal
nll_norm <- function(mu, sigma) { -sum(dnorm(obs, mean = mu, sd = sigma, log = TRUE)) }
fit_mle <- mle(nll_norm, start = list(mu = mean(obs), sigma = sd(obs)))

# 3. Poisson with Predictors
nll_pois <- function(beta0, beta1) {
  lambda <- exp(beta0 + beta1 * x)
  -sum(dpois(y, lambda, log = TRUE))
}
fit_mle <- mle(nll_pois, start = list(beta0 = 0, beta1 = 1))

# 4. Negative Binomial
nll_nb <- function(beta0, beta1, theta) {
  mu <- exp(beta0 + beta1 * x)
  -sum(dnbinom(y, mu = mu, size = exp(theta), log = TRUE))
}
fit_mle <- mle(nll_nb, start = list(beta0 = 0, beta1 = 1, theta = 2))
# Note: theta is on log scale; exp(coef(fit_mle)["theta"]) to compare to glm.nb

# 5. Normal Mixture
negloglik_mix <- function(lambda, mu1, s1, mu2, s2) {
  -sum(log(lambda * dnorm(obs, mu1, s1) + (1 - lambda) * dnorm(obs, mu2, s2)))
}
fit_mix_mle <- mle(negloglik_mix,
  start = list(lambda=0.5, mu1=mean(obs)-sd(obs), s1=sd(obs), mu2=mean(obs)+sd(obs), s2=sd(obs)),
  method = "L-BFGS-B",
  lower = c(lambda = 0.01, mu1 = -Inf, s1 = 0.1, mu2 = -Inf, s2 = 0.1),
  upper = c(lambda = 0.99, mu1 = Inf,  s1 = Inf,  mu2 = Inf,  s2 = Inf))
```

---

## 8. Manual EM Algorithm (2-Component Normal)

```r
twoMix <- function(x) {
  lambda <- 0.5
  mu     <- range(x)           # Initialize at min/max
  sigma  <- rep(sd(x), 2)      # Initialize at overall SD

  repeat {
    oldp <- c(lambda, mu, sigma)

    # E-step
    gamma <- lambda * dnorm(x, mu[1], sigma[1])
    gamma <- gamma / (gamma + (1 - lambda) * dnorm(x, mu[2], sigma[2]))

    # M-step
    mu[1]    <- sum(gamma * x) / sum(gamma)
    mu[2]    <- sum((1 - gamma) * x) / sum(1 - gamma)
    sigma[1] <- sqrt(sum(gamma * (x - mu[1])^2) / sum(gamma))
    sigma[2] <- sqrt(sum((1 - gamma) * (x - mu[2])^2) / sum(1 - gamma))
    lambda   <- sum(gamma) / length(x)

    if (sum(abs(c(lambda, mu, sigma) - oldp)) < 0.001) break
  }
  return(list(lambda = lambda, mu = mu, sigma = sigma))
}
```

---

## 9. Paste-able Interpretation Templates

* **Overdispersion:** "The residual deviance is substantially larger than the degrees of freedom, indicating overdispersion. The Poisson assumption (mean = variance) is violated, so a Negative Binomial model is appropriate."
* **Log-Link Coefficients (Gamma/Poisson/NegBin):** "The exponentiated coefficient for [Variable] is [exp(beta)]. This indicates that for every one-unit increase in [Variable], the expected [Response] is multiplied by a factor of [exp(beta)], holding all else constant."
* **Offset:** "An offset of log([Exposure]) is used to model the *rate* of events rather than the raw count. This controls for differing levels of exposure between groups."
* **Gamma Zero Fix:** "A small constant was added to the response because the Gamma distribution requires strictly positive values (y > 0)."
* **LRT Comparison:** "The likelihood ratio test yields a p-value of [p-val]. Since p < 0.05, we reject the simpler model; the inclusion of [Variable] significantly improves model fit."
* **Mixture AIC/BIC:** "A [k]-component mixture model yields the lowest AIC and BIC, indicating the optimal balance between model fit and complexity."
* **KDE Bandwidth:** "The cross-validation bandwidth (`ucv`) produces a less smooth but more informative density estimate than the rule-of-thumb (`nrd0`), revealing the true underlying multimodal structure of the data."
