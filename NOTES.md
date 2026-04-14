# Statistical Computing Quiz Reference (Weeks 5-9)

---

## DECISION LOGIC: Which Model to Use

**Look at the response variable (y) first.**

```
Is y a count (0, 1, 2, 3...)?
  ├── Residual Deviance ≈ Degrees of Freedom → POISSON
  └── Residual Deviance >> Degrees of Freedom → NEGATIVE BINOMIAL

Is y continuous and strictly positive (e.g. rainfall, hospital stay)?
  ├── Symmetric → LINEAR MODEL (lm)
  └── Right-skewed (long tail to the right) → GAMMA GLM

Does the histogram show two or more humps with no group label?
  └── MIXTURE MODEL (normalmixEM or regmixEM)

Does a scatterplot show two different slopes with no group label?
  └── MIXTURE OF REGRESSIONS (regmixEM)
```

**Overdispersion check for Poisson (always run this):**
```r
summary(fit_pois)$deviance / summary(fit_pois)$df.residual
# If result > 1.2, switch to glm.nb()
```

**Copy-paste justifications:**

| Scenario | Model | Paste this text |
|---|---|---|
| Count data, deviance ≈ df | Poisson | "The response is a count of independent events. Residual deviance is close to degrees of freedom, indicating no overdispersion. Poisson is appropriate." |
| Count data, deviance >> df | Neg Binomial | "The residual deviance ([X]) is substantially larger than the degrees of freedom ([X]), indicating overdispersion. Negative Binomial is used instead of Poisson." |
| Positive, skewed continuous | Gamma | "The response is strictly positive and right-skewed. A Gamma GLM with log link is appropriate." |
| Two humps in histogram | Mixture | "The distribution appears bimodal, suggesting the presence of latent subpopulations. A Normal mixture model is fitted." |
| Need to compare models | AIC/BIC/LRT | "Lower AIC/BIC indicates better fit. The model with AIC=[X] is preferred." |

---

## 1. Custom Maximum Likelihood (MLE)

**When:** Asked to find parameters manually without using glm(). High-mark question.

### Template: Single Parameter (e.g. Exponential)
```r
library(stats4)

obs <- c(...)  # FILL IN DATA

# Negative log-likelihood
nll <- function(lambda) {
  -sum(dexp(obs, rate = lambda, log = TRUE))
}

# Fit — start with a sensible guess (1/mean for exponential)
fit_mle <- mle(nll, start = list(lambda = 1/mean(obs)))
summary(fit_mle)

# Closed-form check (exponential only): lambda = 1/mean
1 / mean(obs)
```

### Template: Two Parameters (Normal)
```r
nll_norm <- function(mu, sigma) {
  -sum(dnorm(obs, mean = mu, sd = sigma, log = TRUE))
}

fit_mle <- mle(nll_norm, start = list(mu = mean(obs), sigma = sd(obs)))
summary(fit_mle)
```

### Template: Poisson with Predictors
```r
nll_pois <- function(beta0, beta1) {
  lambda <- exp(beta0 + beta1 * x_variable)
  -sum(dpois(y_variable, lambda, log = TRUE))
}

fit_mle <- mle(nll_pois, start = list(beta0 = 0, beta1 = 1))
summary(fit_mle)
```

### Template: Negative Binomial with Predictors
```r
nll_nb <- function(beta0, beta1, theta) {
  mu <- exp(beta0 + beta1 * x_variable)
  -sum(dnbinom(y_variable, mu = mu, size = exp(theta), log = TRUE))
}

fit_mle_nb <- mle(nll_nb, start = list(beta0 = 0, beta1 = 1, theta = 2))
summary(fit_mle_nb)

# theta is on log scale, so exponentiate to compare with glm.nb output:
exp(coef(fit_mle_nb)["theta"])
```

### Template: Normal Mixture MLE
```r
negloglik_mix <- function(lambda, mu1, s1, mu2, s2) {
  -sum(log(lambda * dnorm(obs, mu1, s1) +
           (1 - lambda) * dnorm(obs, mu2, s2)))
}

fit_mix_mle <- mle(negloglik_mix,
  start = list(lambda = 0.5, mu1 = mean(obs) - sd(obs),
               s1 = sd(obs), mu2 = mean(obs) + sd(obs), s2 = sd(obs)),
  method = "L-BFGS-B",
  lower = c(lambda = 0.01, mu1 = -Inf, s1 = 0.1, mu2 = -Inf, s2 = 0.1),
  upper = c(lambda = 0.99, mu1 = Inf,  s1 = Inf,  mu2 = Inf,  s2 = Inf))

summary(fit_mix_mle)
```

**Distribution functions to swap in:**
- Exponential: `dexp(x, rate, log = TRUE)`
- Normal: `dnorm(x, mean, sd, log = TRUE)`
- Poisson: `dpois(x, lambda, log = TRUE)`
- Negative Binomial: `dnbinom(x, mu = mu, size = size, log = TRUE)`

---

## 2. GLM Fitting and Interpretation

### Fitting

```r
# POISSON
fit_pois <- glm(y ~ x1 + x2, family = poisson, data = df)

# With OFFSET (use for rates: claims per policyholder, events per time unit)
fit_pois_offset <- glm(y ~ x1 + x2 + offset(log(exposure_var)),
                       family = poisson, data = df)

# NEGATIVE BINOMIAL (overdispersed counts)
library(MASS)
fit_nb <- glm.nb(y ~ x1 + x2, data = df)

# Negative Binomial with offset
fit_nb_offset <- glm.nb(y ~ x1 + x2 + offset(log(exposure_var)), data = df)

# GAMMA (positive, right-skewed continuous)
fit_gamma <- glm(y ~ x1 + x2, family = Gamma(link = "log"), data = df)

# LINEAR MODEL
fit_lm <- lm(y ~ x1 + x2, data = df)

# No intercept (e.g. Hubble constant)
fit_lm_noint <- lm(y ~ x - 1, data = df)

# Interaction model (separate slopes per group)
fit_interact <- lm(y ~ x * group, data = df)

# LOGISTIC REGRESSION
fit_logit <- glm(y ~ x1 + x2, family = binomial, data = df)
```

### Interpretation of Log-Link Coefficients (Poisson, NegBin, Gamma, Logistic)

```r
# Always exponentiate log-link coefficients
exp(coef(fit_model))

# For a specific variable:
exp(coef(fit_model)["variable_name"])
```

**Paste these interpretation templates:**

- **If exp(beta) = 1.12:** "For a one-unit increase in [Variable], the expected [Response] is multiplied by 1.12, a 12% increase."
- **If exp(beta) = 0.88:** "For a one-unit increase in [Variable], the expected [Response] is multiplied by 0.88, a 12% decrease."
- **If exp(beta) = 3.9 (logistic):** "The odds of [Outcome] are 3.9 times higher for [Group] compared to the baseline."
- **Intercept (Poisson/Gamma):** "At baseline conditions (all predictors = 0 or reference level), the expected [Response] is exp([intercept value])."

### Overdispersion Check

```r
summary(fit_pois)$deviance / summary(fit_pois)$df.residual
# > 1.2 means overdispersion → use glm.nb()
```

**Paste:** "The ratio of residual deviance to degrees of freedom is [X], which is substantially greater than 1, indicating overdispersion. The Poisson assumption that mean equals variance is violated. A Negative Binomial model is used instead."

### Checking Categorical Variables are Treated Correctly

```r
# Check if R sees a variable as a factor
class(df$variable)  # should say "factor"

# Convert to factor
df$variable <- as.factor(df$variable)

# Convert with labels
df$season <- factor(df$season, levels = c(1,2,3,4),
                    labels = c("Spring","Summer","Autumn","Winter"))
```

---

## 3. Model Comparison

### AIC and BIC
```r
AIC(model1, model2)
BIC(model1, model2)
# Lower = better
```

### Likelihood Ratio Test (Nested Models)
```r
anova(simpler_model, fuller_model, test = "Chisq")
# Low p-value (< 0.05) = fuller model is significantly better

# Manual LRT
lrt_stat <- -2 * (logLik(simpler_model) - logLik(fuller_model))
lrt_pval  <- pchisq(lrt_stat, df = df_difference, lower.tail = FALSE)
```

**Paste:** "The likelihood ratio test gives a p-value of [X]. Since p [< / >] 0.05, we [reject / fail to reject] the simpler model. The [fuller / simpler] model is preferred."

**Note on Gamma AIC:** Do not directly compare AIC of a Gamma model to a Poisson/NegBin model — the distributional assumptions and response variable scale differ.

---

## 4. KDE and Nonparametric Classification

### KDE

```r
# Default Gaussian, rule-of-thumb bandwidth
d <- density(df$variable)
plot(d)

# Rule-of-thumb bandwidth explicitly
d_rot <- density(df$variable, bw = "nrd0")

# Cross-validation bandwidth (smaller, reveals more structure)
d_ucv <- density(df$variable, bw = "ucv")

# Print the chosen bandwidth
cat("Bandwidth:", d_ucv$bw)

# Specific kernel
d_rect <- density(df$variable, kernel = "rectangular", bw = 5)
d_epan <- density(df$variable, kernel = "epanechnikov", bw = 5)

# 2D density
library(MASS)
dens_2d <- kde2d(x, y)
contour(dens_2d)
image(dens_2d)
```

**Bandwidth decision:** Small bandwidth = jagged/noisy. Large bandwidth = oversmoothed, hides structure. Cross-validation (ucv) is generally the most informative for revealing modes.

### Full Classifier Template
```r
# Predict P(Class = 1 | x) using Bayes rule + density

lo <- min(df$variable)
hi <- max(df$variable)

# Estimate density for each group
f0 <- density(df$variable[df$group == "no"],  from = lo, to = hi)
f1 <- density(df$variable[df$group == "yes"], from = lo, to = hi)

# Prior probability of class 1
p_c1 <- mean(df$group == "yes")

# Plot the posterior probability curve
plot(f1$x, f1$y * p_c1 / (f1$y * p_c1 + f0$y * (1 - p_c1)),
     type = "l", xlab = "variable", ylab = "P(Class=1 | x)")
abline(h = 0.5, lty = 3)

# Get probability at a specific value x_new
val_f0 <- approx(f0$x, f0$y, xout = x_new)$y
val_f1 <- approx(f1$x, f1$y, xout = x_new)$y
prob_c1 <- (val_f1 * p_c1) / (val_f1 * p_c1 + val_f0 * (1 - p_c1))
print(prob_c1)
```

**Formula:**

P(Class=1 | x) = [ f1(x) * P(C1) ] / [ f1(x) * P(C1) + f0(x) * P(C0) ]

---

## 5. Normal Mixture Models

### 1D Mixture (normalmixEM)
```r
library(mixtools)

fit_mix <- normalmixEM(df$variable, k = 2)  # k = number of components

# Extract parameters
fit_mix$lambda  # Mixing proportions (must sum to 1)
fit_mix$mu      # Component means
fit_mix$sigma   # Component standard deviations
fit_mix$loglik  # Log-likelihood (needed for AIC/BIC)

# Plot
plot(fit_mix, whichplots = 2)

# Overlay on histogram manually
hist(df$variable, breaks = 20, probability = TRUE,
     col = "lightblue", main = "Mixture fit")
curve(fit_mix$lambda[1] * dnorm(x, fit_mix$mu[1], fit_mix$sigma[1]) +
      fit_mix$lambda[2] * dnorm(x, fit_mix$mu[2], fit_mix$sigma[2]),
      add = TRUE, col = "red", lwd = 2)
```

### Mixture of Regressions (regmixEM)
```r
# Used when scatterplot shows two slopes but no group label
fit_reg <- regmixEM(y = df$y, x = df$x, k = 2, arbvar = FALSE)
summary(fit_reg)

# Plot the two regression lines
plot(df$x, df$y, main = "Mixture of Regressions")
abline(fit_reg$beta[, 1], col = "red",  lwd = 2)
abline(fit_reg$beta[, 2], col = "blue", lwd = 2)

# Posterior probabilities: which component does each point belong to?
fit_reg$posterior

# Assign each observation to its most likely component
assigned <- ifelse(fit_reg$posterior[, 1] > 0.5, "Component 1", "Component 2")
table(assigned)  # or table(assigned, df$true_group) if you know the truth
```

### Multivariate Mixture (mvnormalmixEM)
```r
# For 2D data (two columns)
X <- df[, c("var1", "var2")]
m2 <- mvnormalmixEM(X, k = 2)
m3 <- mvnormalmixEM(X, k = 3)

plot(m2, whichplots = 2)
plot(m3, whichplots = 2)
```

### Manual AIC/BIC for Mixture Models

mixtools does NOT provide AIC/BIC — you must compute manually.

```r
k <- 2         # number of components
n <- nrow(df)  # or length(x) for 1D

# Parameter counts:
# 1D Normal mixture:      p = 3k - 1
# 2D Bivariate mixture:   p = 6k - 1

p <- 3 * k - 1   # Change to 6*k-1 for 2D

aic_val <- -2 * fit_mix$loglik + 2 * p
bic_val <- -2 * fit_mix$loglik + log(n) * p

cat("AIC:", aic_val, "\nBIC:", bic_val)
```

### Comparing Multiple Mixture Components
```r
# Fit several models
m2 <- normalmixEM(x, k = 2)
m3 <- normalmixEM(x, k = 3)
m4 <- normalmixEM(x, k = 4)

n  <- length(x)
p2 <- 3*2 - 1
p3 <- 3*3 - 1
p4 <- 3*4 - 1

results <- data.frame(
  Model  = c("k=2", "k=3", "k=4"),
  LogLik = round(c(m2$loglik, m3$loglik, m4$loglik), 2),
  Params = c(p2, p3, p4),
  AIC    = round(c(-2*m2$loglik + 2*p2, -2*m3$loglik + 2*p3, -2*m4$loglik + 2*p4), 2),
  BIC    = round(c(-2*m2$loglik + log(n)*p2, -2*m3$loglik + log(n)*p3, -2*m4$loglik + log(n)*p4), 2)
)
print(results)
# Choose the row with the lowest AIC or BIC
```

**Paste:** "A [k]-component mixture model is chosen because it yields the lowest AIC ([X]) and BIC ([X]), indicating the best balance between model fit and complexity."

---

## 6. Manual EM Algorithm

```r
# For a two-component Normal mixture — manual EM loop
twoMix <- function(x) {
  lambda <- 0.5
  mu     <- range(x)           # initialise means at min and max
  sigma  <- rep(sd(x), 2)      # initialise both SDs at overall SD

  repeat {
    oldp <- c(lambda, mu, sigma)

    # E-step: compute posterior probabilities (gamma)
    gamma <- lambda * dnorm(x, mu[1], sigma[1])
    gamma <- gamma / (gamma + (1 - lambda) * dnorm(x, mu[2], sigma[2]))

    # M-step: update parameters using weighted means
    mu[1]    <- sum(gamma * x) / sum(gamma)
    mu[2]    <- sum((1 - gamma) * x) / sum(1 - gamma)
    sigma[1] <- sqrt(sum(gamma * (x - mu[1])^2) / sum(gamma))
    sigma[2] <- sqrt(sum((1 - gamma) * (x - mu[2])^2) / sum(1 - gamma))
    lambda   <- sum(gamma) / length(x)

    if (sum(abs(c(lambda, mu, sigma) - oldp)) < 0.001) break
  }
  return(list(lambda = lambda, mu = mu, sigma = sigma))
}

result <- twoMix(df$variable)
result$lambda
result$mu
result$sigma
```

---

## 7. Linear Regression (Week 6 Specifics)

```r
# Standard
fit_lm <- lm(y ~ x, data = df)
summary(fit_lm)

# No intercept (physical relationship through origin)
fit_lm_noint <- lm(y ~ x - 1, data = df)

# Interaction model (separate slopes per group)
fit_interact <- lm(y ~ x * group, data = df)
# Coefficients: cc[1] = baseline intercept, cc[2] = group offset,
#               cc[3] = baseline slope, cc[4] = slope difference
cc <- coef(fit_interact)
abline(cc[1],          cc[3],          col = "red")   # group 1
abline(cc[1] + cc[2],  cc[3] + cc[4],  col = "blue")  # group 2

# Assumptions check
plot(fit_lm)
# Check: Residuals vs Fitted (random cloud = good), Normal Q-Q (points on line = good)

# Confidence interval
confint(fit_lm)

# Influential points
hatvalues(fit_lm) > 0.08   # TRUE = high leverage
```

### Hypothesis Test on a Coefficient
```r
n          <- fit_lm$df.residual
theta_null <- 0.02                                   # your H0 value
theta_hat  <- coef(fit_lm)["x"]
SE         <- summary(fit_lm)$coefficients["x", 2]

t_stat <- (theta_hat - theta_null) / SE

# Two-sided
pval <- 2 * pt(abs(t_stat), df = n, lower.tail = FALSE)

# One-sided (less than)
pval <- pt(t_stat, df = n, lower.tail = TRUE)

pval
```

---

## 8. Logistic Regression

```r
fit_logit <- glm(y ~ x1 + x2, family = binomial, data = df)
summary(fit_logit)

# Interpret as odds ratios
exp(coef(fit_logit))

# Change reference category
df$variable <- relevel(df$variable, ref = "NewBaseline")
```

**Paste:** "The exponentiated coefficient for [Variable] is [X], meaning the odds of [Outcome] are [X] times higher for [Group] compared to the reference group."

---

## 9. Packages Quick Reference

```r
library(MASS)      # glm.nb(), datasets (Insurance, galaxies, whiteside, Pima.tr)
library(stats4)    # mle()
library(mixtools)  # normalmixEM(), regmixEM(), mvnormalmixEM()
library(ggplot2)   # plotting
```

---

## 10. Interpretation Templates (Copy-Paste)

**Gamma / Poisson / NegBin coefficient:**
"The exponentiated coefficient for [Variable] is [exp(beta)]. This indicates that for every one-unit increase in [Variable], the expected [Response] changes by a factor of [exp(beta)], equivalent to a [abs(exp(beta)-1)*100]% [increase/decrease], holding all other variables constant."

**Overdispersion:**
"The residual deviance ([X]) is substantially larger than the degrees of freedom ([X]), indicating overdispersion. The Poisson assumption (mean = variance) is violated, so a Negative Binomial model is used."

**Offset:**
"An offset of log(Holders) is included to model the claim rate per policyholder, rather than the raw count of claims. Without this, groups with more policyholders would appear to have more claims simply due to exposure."

**Model comparison (AIC/BIC):**
"The [model name] has the lowest AIC ([X]) and BIC ([X]), indicating the best trade-off between fit and complexity. It is preferred over [other model] (AIC = [X])."

**LRT:**
"The likelihood ratio test gives p = [X]. Since p < 0.05, the more complex model with [added variable] is significantly better and is retained."

**Mixture model selection:**
"A [k]-component mixture was selected because it produced the lowest [AIC/BIC] value ([X]). The [k+1]-component model showed a [higher/comparable] penalty without sufficient improvement in log-likelihood."

**Posterior probability:**
"The posterior probability matrix (fit$posterior) gives the probability that each observation belongs to each component. Observation [i] has a probability of [X] of belonging to Component 1, and is therefore assigned to [Component 1/2]."

**KDE bandwidth:**
"The cross-validation bandwidth ([X]) is smaller than the rule-of-thumb bandwidth ([X]), producing a less smooth but more informative density estimate that better reveals the underlying multimodal structure."
