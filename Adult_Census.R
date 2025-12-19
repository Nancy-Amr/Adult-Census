# ============================================================
# Adult dataset — Preprocessing + Recipes (Leakage-safe)
# Models: Logistic Regression, Naive Bayes, Random Forest
# ============================================================

# Helper: run a contiguous block of lines from a file (useful in notebooks)
run_lines <- function(from, to, file = "Adult_Census.R") {
  lines <- readLines(file, warn = FALSE)
  code <- paste(lines[from:to], collapse = "\n")
  cat("\n--- Running lines", from, "to", to, "---\n")
  eval(parse(text = code), envir = .GlobalEnv)
}

# -----------------------------
# 0) Libraries + global options
# -----------------------------
suppressPackageStartupMessages({
  library(tidyverse)
  library(tidymodels)
  library(janitor)
  library(stringr)
  library(recipes)
  library(glue)
  library(dplyr)
  library(ggplot2)
})

set.seed(404) # Ensures split/fold reproducibility

# -----------------------------
# 0.1) User-configurable knobs
# -----------------------------
DATA_PATH <- "adult.csv"
TARGET_COL <- "income"
TEST_PROP <- 0.20
V_FOLDS <- 5

# -----------------------------
# 0.2) Output locations
# -----------------------------
OUTPUT_DIR <- "outputs"
dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(file.path(OUTPUT_DIR, "data"), showWarnings = FALSE, recursive = TRUE)

# Make yardstick treat the SECOND factor level as the "event"/positive class.
# We'll set income_level levels as: low_income, high_income => high_income is positive.
options(yardstick.event_first = FALSE)

# ===========
# 1) Load + deterministic cleaning (safe globally)
# ===========

raw <- readr::read_csv(DATA_PATH, show_col_types = FALSE) %>%
  janitor::clean_names()

# Convert tokens to NA + trim
raw <- raw %>%
  mutate(across(where(is.character), ~ trimws(.x))) %>%
  mutate(across(where(is.character), ~ na_if(.x, "?"))) %>%
  mutate(across(where(is.character), ~ na_if(.x, "")))

# Ensure target exists
if (!TARGET_COL %in% names(raw)) {
  stop(glue::glue(
    "TARGET_COL='{TARGET_COL}' not found. Available: {paste(names(raw), collapse=', ')}"
  ))
}

# 1.1) Duplicate diagnostics (before removal)
n_total <- nrow(raw)
n_dup_exact <- sum(duplicated(raw))
n_unique <- dplyr::n_distinct(raw)

cat("Total rows:", n_total, "\n")
cat("Unique rows:", n_unique, "\n")
cat("Exact duplicate rows:", n_dup_exact, "\n")

dup_groups <- raw %>%
  count(across(everything()), name = "freq") %>%
  filter(freq > 1)

cat("Duplicate groups:", nrow(dup_groups), "\n")
cat(
  "Max repeats of a single row:",
  ifelse(nrow(dup_groups) == 0, 1, max(dup_groups$freq)),
  "\n"
)

# 1.2) Remove exact duplicates (keeps 1 copy)
raw <- raw %>% distinct()

# -----------------------------
# 2) Target inspection + clean mapping
# -----------------------------
cat("Unique values in target:\n")
print(sort(unique(raw[[TARGET_COL]]), na.last = TRUE))

cat("\nCounts per target value:\n")
counts <- table(raw[[TARGET_COL]], useNA = "ifany")
print(counts)

cat("\nClass percentages (%):\n")
pct <- prop.table(counts) * 100
print(round(pct, 2))

# Create a binary factor target:
raw <- raw %>%
  mutate(
    income_level = dplyr::case_when(
      income == "<=50K" ~ 0,
      income == ">50K" ~ 1,
      TRUE ~ NA_real_
    ),
    income_level = factor(
      income_level,
      levels = c(0, 1),
      labels = c("low_income", "high_income")
    )
  ) %>%
  select(-income)

TARGET_COL <- "income_level"

# Drop any rows with missing target (should be none, but safe)
raw <- raw %>% filter(!is.na(.data[[TARGET_COL]]))

# Drop fnlwgt (commonly excluded)
if ("fnlwgt" %in% names(raw)) {
  raw <- raw %>% select(-fnlwgt)
}

# -----------------------------
# 3) Split + CV (leakage-safe)
# -----------------------------
split_obj <- initial_split(raw, prop = 1 - TEST_PROP, strata = all_of(TARGET_COL))
train_data <- training(split_obj)
test_data <- testing(split_obj)

# Persist split datasets for reproducibility / later reuse
readr::write_csv(train_data, file.path(OUTPUT_DIR, "data", "train_data.csv"))
readr::write_csv(test_data, file.path(OUTPUT_DIR, "data", "test_data.csv"))

folds <- vfold_cv(train_data, v = V_FOLDS, strata = all_of(TARGET_COL))

cat("\nTrain rows:", nrow(train_data), " | Test rows:", nrow(test_data), "\n")
cat("CV folds:", V_FOLDS, "\n")

# ============================================================================
# HYPOTHESIS TESTING FOR Income Level Prediction
# ============================================================================

# 1. LOAD THE DATA

# Read the CSV file
df <- readr::read_csv(file.path(OUTPUT_DIR, "data", "train_data.csv"))

# Display basic information about the dataset
cat("Dataset dimensions:", dim(df), "\n")
cat("Target variable:", unique(df$income_level), "\n\n")

# ----------------------------------------------------------------------------
# 2. DEFINE FEATURE LISTS

target <- "income_level"

# List all categorical features to test
categorical_features <- names(df)[
  sapply(df, function(x) is.factor(x) || is.character(x)) &
    names(df) != target]

# List all numerical features to test
numerical_features <- names(df)[sapply(df, is.numeric)]

# ----------------------------------------------------------------------------
# 3. CHI-SQUARE TESTS FOR CATEGORICAL FEATURES

# Function to perform chi-square test and extract results
perform_chi_square <- function(data, feature, target) {
  # Create contingency table (cross-tabulation)
  contingency_table <- table(data[[feature]], data[[target]])
  
  # Perform chi-square test
  # H0: Feature and target are independent (feature is NOT useful)
  # H1: Feature and target are dependent (feature IS useful)
  test_result <- chisq.test(contingency_table)
  
  # Return results as a data frame
  return(data.frame(
    Feature = feature,
    Test_Type = "Chi-Square",
    Statistic = round(test_result$statistic, 4),
    P_Value = round(test_result$p.value, 6),
    Significant = test_result$p.value < 0.05,
    stringsAsFactors = FALSE
  ))
}

# Initialize empty data frame to store chi-square results
chi_square_results <- data.frame()

# Loop through each categorical feature and perform chi-square test

for (feature in categorical_features) {
  tryCatch({
    # Perform the test
    result <- perform_chi_square(df, feature, "income_level")
    
    # Append to results
    chi_square_results <- rbind(chi_square_results, result)
    
    # Print immediate feedback
    cat(sprintf("%-20s | χ² = %8.2f | p-value: %.6f | %s\n", 
                feature, 
                result$Statistic,
                result$P_Value,
                ifelse(result$Significant, "✓ SIGNIFICANT", "✗ Not significant")))
    
  }, error = function(e) {
    cat(sprintf("%-20s | Error: %s\n", feature, e$message))
  })
}


# ----------------------------------------------------------------------------
# 4. T-TESTS FOR NUMERICAL FEATURES

# Function to perform t-test for numerical features
perform_t_test <- function(data, feature, target) {
  # Split data by target classes
  groups <- split(data[[feature]], data[[target]])
  
  # Perform independent samples t-test
  # H0: Mean of feature is same for both income levels
  # H1: Mean of feature differs between income levels
  test_result <- t.test(groups[[1]], groups[[2]])
  
  # Calculate mean difference
  mean_diff <- mean(groups[[1]], na.rm = TRUE) - mean(groups[[2]], na.rm = TRUE)
  
  # Return results as a data frame
  return(data.frame(
    Feature = feature,
    Test_Type = "T-Test",
    Statistic = round(test_result$statistic, 4),
    P_Value = round(test_result$p.value, 6),
    Mean_Diff = round(mean_diff, 2),
    Significant = test_result$p.value < 0.05,
    stringsAsFactors = FALSE
  ))
}

# Initialize empty data frame to store t-test results
t_test_results <- data.frame()

# Loop through each numerical feature and perform t-test

for (feature in numerical_features) {
  tryCatch({
    # Perform the test
    result <- perform_t_test(df, feature, "income_level")
    
    # Append to results
    t_test_results <- rbind(t_test_results, result)
    
    # Print immediate feedback
    cat(sprintf("%-20s | t = %8.2f | p-value: %.6f | %s\n", 
                feature, 
                result$Statistic,
                result$P_Value,
                ifelse(result$Significant, "✓ SIGNIFICANT", "✗ Not significant")))
    
  }, error = function(e) {
    cat(sprintf("%-20s | Error: %s\n", feature, e$message))
  })
}


# ----------------------------------------------------------------------------
# 5. DISPLAY SUMMARY STATISTICS

cat("SUMMARY OF HYPOTHESIS TESTS\n")
cat("=" , rep("=", 60), "\n", sep = "")
cat(sprintf("Total features tested: %d\n", nrow(all_results)))
cat(sprintf("Significant features (p < 0.05): %d\n", sum(all_results$Significant)))
cat(sprintf("Non-significant features: %d\n", sum(!all_results$Significant)))
cat("\n")

# ----------------------------------------------------------------------------
# 6. VISUALIZATION: P-VALUES BY FEATURE

# Create visualization of p-values
p_value_plot <- ggplot(all_results, aes(x = reorder(Feature, P_Value), y = P_Value)) +
  geom_bar(stat = "identity", aes(fill = Significant)) +
  geom_hline(yintercept = 0.05, linetype = "dashed", color = "red", size = 1) +
  scale_fill_manual(values = c("TRUE" = "#22c55e", "FALSE" = "#ef4444"),
                    labels = c("Significant", "Not Significant")) +
  coord_flip() +
  labs(
    title = "Hypothesis Test Results: P-Values by Feature",
    subtitle = "Red line indicates significance threshold (α = 0.05)",
    x = "Feature",
    y = "P-Value",
    fill = "Result"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    legend.position = "bottom"
  )

# Display the plot
print(p_value_plot)

# ----------------------------------------------------------------------------
# 7. EFFECT SIZE CALCULATION (For T-Tests)

# Function to calculate Cohen's d
calculate_cohens_d <- function(data, feature, target) {
  groups <- split(data[[feature]], data[[target]])
  
  mean1 <- mean(groups[[1]], na.rm = TRUE)
  mean2 <- mean(groups[[2]], na.rm = TRUE)
  
  sd1 <- sd(groups[[1]], na.rm = TRUE)
  sd2 <- sd(groups[[2]], na.rm = TRUE)
  
  n1 <- length(na.omit(groups[[1]]))
  n2 <- length(na.omit(groups[[2]]))
  
  # Pooled standard deviation
  pooled_sd <- sqrt(((n1 - 1) * sd1^2 + (n2 - 1) * sd2^2) / (n1 + n2 - 2))
  
  # Cohen's d
  cohens_d <- (mean1 - mean2) / pooled_sd
  
  # Interpret effect size
  effect_interpretation <- ifelse(abs(cohens_d) < 0.2, "Negligible",
                                  ifelse(abs(cohens_d) < 0.5, "Small",
                                         ifelse(abs(cohens_d) < 0.8, "Medium", "Large")))
  
  return(data.frame(
    Feature = feature,
    Cohens_d = round(cohens_d, 3),
    Effect_Size = effect_interpretation
  ))
}
effect_sizes <- data.frame()

cat("Effect Size Interpretation:\n
    |d| < 0.2: Negligible effect\n
    0.2 ≤ |d| < 0.5: Small effect\n
    0.5 ≤ |d| < 0.8: Medium effect\n
    |d| ≥ 0.8: Large effect\n")

# Calculate effect sizes for numerical features
for (feature in numerical_features) {
  tryCatch({
    effect <- calculate_cohens_d(df, feature, "income_level")
    effect_sizes <- rbind(effect_sizes, effect)
    cat(sprintf("%-20s | Cohen's d: %6.3f | Effect: %s\n", 
                feature, effect$Cohens_d, effect$Effect_Size))
  }, error = function(e) {
    cat(sprintf("%-20s | Error calculating effect size\n", feature))
  })
}

# ----------------------------------------------------------------------------
# 8. EFFECT SIZE CALCULATION (For CHI-SQUARE Tests)

# Function to calculate Cramér's V
calculate_cramers_v <- function(data, feature, target) {
  contingency_table <- table(data[[feature]], data[[target]])
  chi2 <- chisq.test(contingency_table)$statistic
  n <- sum(contingency_table)
  
  # Degrees of freedom for the denominator
  min_dim <- min(dim(contingency_table)) - 1
  
  v_stat <- sqrt(chi2 / (n * min_dim))
  
  # Interpret effect size
  interpretation <- ifelse(v_stat < 0.1, "Negligible",
                           ifelse(v_stat < 0.3, "Small",
                                  ifelse(v_stat < 0.5, "Medium", "Large")))
  
  return(data.frame(
    Feature = feature,
    Cramers_V = round(v_stat, 3),
    Effect_Size = interpretation
  ))
}

effect_sizes <- data.frame()

cat("Effect Size Interpretation:\n
    |d| < 0.1: Negligible effect\n
    0.1 ≤ |d| < 0.3: Small effect\n
    0.3 ≤ |d| < 0.5: Medium effect\n
    |d| ≥ 0.5: Large effect\n")

# Calculate effect sizes for categorical features
categorical_effects <- data.frame()
for (feature in categorical_features) {
  eff <- calculate_cramers_v(df, feature, "income_level")
  categorical_effects <- rbind(categorical_effects, eff)
  cat(sprintf("%-20s | Cramér's V: %6.3f | Effect: %s\n", 
              feature, eff$Cramers_V, eff$Effect_Size))
}


# ----------------------------------------------------------------------------
# 9. Visualize CHI SQUARE Results


chi_vis <- chi_square_results %>%
  arrange(desc(Statistic)) %>%
  mutate(
    P_Label = ifelse(P_Value < 0.001, "<0.001", sprintf("%.3f", P_Value)),
    Sig_Label = ifelse(Significant, "***", "")
  )

ggplot(chi_vis,
       aes(x = reorder(Feature, Statistic),
           y = Statistic,
           fill = Feature)) +       # map fill to Feature
  geom_col() +
  coord_flip() +
  labs(
    title = "Chi-Square Statistics: Categorical Features vs Income",
    x = "Feature",
    y = "Chi-Square Statistic"
  ) +
  theme_minimal() +
  theme(legend.position = "none")  # hide legend if too many features


# ----------------------------------------------------------------------------
# 10. Visualize T-TEST Results

t_vis <- t_test_results %>%
  mutate(
    Abs_T = abs(Statistic),
    P_Label = ifelse(P_Value < 0.001, "<0.001", sprintf("%.3f", P_Value)),
    Sig_Label = ifelse(Significant, "***", "")
  ) %>%
  arrange(desc(Abs_T))

ggplot(t_vis,
       aes(x = reorder(Feature, Abs_T),
           y = Abs_T,
           fill = Feature)) +
  geom_col() +
  coord_flip() +
  labs(
    title = "T-Statistics: Numerical Features vs Income",
    x = "Feature",
    y = "Absolute T-Statistic"
  ) +
  theme_minimal() +
  theme(legend.position = "none")


# ============================================================
# 4) EDA helpers (optional, TRAIN ONLY)
# ============================================================

# 4.1) Missingn summary
missing_summary <- train_data %>%
  summarise(across(everything(), ~ sum(is.na(.x)))) %>%
  pivot_longer(everything(), names_to = "column", values_to = "na_count") %>%
  mutate(
    n = nrow(train_data),
    na_pct = round(100 * na_count / n, 2)
  ) %>%
  arrange(desc(na_count))

cat("\n=== Missingness (TRAIN) ===\n")
print(missing_summary %>% filter(na_count > 0))

# 4.2) Is "unknown"/missing informative?
# (Missing here = NA, which will become "unknown" in the recipe)
missing_by_target <- function(df, col, target) {
  df %>%
    mutate(is_missing = is.na(.data[[col]])) %>%
    count(.data[[target]], is_missing, name = "n") %>%
    group_by(.data[[target]]) %>%
    mutate(pct_within_class = round(100 * n / sum(n), 2)) %>%
    ungroup()
}

chisq_missing <- function(df, col, target) {
  tab <- table(df[[target]], is.na(df[[col]]))
  # If any expected counts are too small, chi-square may warn;
  # fisher.test is safer but slower.
  test <- suppressWarnings(chisq.test(tab))
  list(tab = tab, test = test)
}

cramers_v <- function(tab) {
  # tab is a contingency table
  chi2 <- suppressWarnings(chisq.test(tab, correct = FALSE)$statistic)
  n <- sum(tab)
  k <- min(nrow(tab), ncol(tab))
  as.numeric(sqrt(chi2 / (n * (k - 1))))
}

cols_to_check <- c("workclass", "occupation", "native_country")

for (col in cols_to_check) {
  cat("\n====================================================\n")
  cat("Column:", col, "\n")
  cat("====================================================\n")

  print(missing_by_target(train_data, col, TARGET_COL))

  res <- chisq_missing(train_data, col, TARGET_COL)
  cat("\nContingency table (rows=target, cols=missing?):\n")
  print(res$tab)

  cat("\nChi-square test:\n")
  print(res$test)

  cat("\nCramer's V (effect size):\n")
  print(round(cramers_v(res$tab), 4))
}

# 4.3) Outlier check
num_cols <- train_data %>%
  select(where(is.numeric)) %>%
  names()

cat("Numeric columns:\n")
print(num_cols)

outlier_summary <- purrr::map_dfr(num_cols, function(col) {
  x <- train_data[[col]]
  x <- x[!is.na(x)]

  if (length(x) == 0) {
    return(tibble(
      column = col, n = 0, missing = sum(is.na(train_data[[col]])),
      q1 = NA_real_, q3 = NA_real_, iqr = NA_real_,
      lower = NA_real_, upper = NA_real_,
      outlier_n = NA_integer_, outlier_pct = NA_real_,
      p01 = NA_real_, p05 = NA_real_, p95 = NA_real_, p99 = NA_real_,
      min = NA_real_, max = NA_real_
    ))
  }

  q1 <- quantile(x, 0.25, na.rm = TRUE, type = 7)
  q3 <- quantile(x, 0.75, na.rm = TRUE, type = 7)
  iqr <- q3 - q1
  lower <- q1 - 1.5 * iqr
  upper <- q3 + 1.5 * iqr

  outlier_n <- sum(x < lower | x > upper)
  outlier_pct <- round(100 * outlier_n / length(x), 3)

  tibble(
    column = col,
    n = length(x),
    missing = sum(is.na(train_data[[col]])),
    q1 = as.numeric(q1), q3 = as.numeric(q3), iqr = as.numeric(iqr),
    lower = as.numeric(lower), upper = as.numeric(upper),
    outlier_n = outlier_n, outlier_pct = outlier_pct,
    p01 = as.numeric(quantile(x, 0.01, na.rm = TRUE)),
    p05 = as.numeric(quantile(x, 0.05, na.rm = TRUE)),
    p95 = as.numeric(quantile(x, 0.95, na.rm = TRUE)),
    p99 = as.numeric(quantile(x, 0.99, na.rm = TRUE)),
    min = min(x, na.rm = TRUE),
    max = max(x, na.rm = TRUE)
  )
}) %>%
  arrange(desc(outlier_pct), desc(outlier_n))

print(outlier_summary)

library(ggplot2)
library(dplyr)
library(tidyr)

num_cols <- train_data %>%
  select(where(is.numeric)) %>%
  names()

train_long <- train_data %>%
  select(all_of(num_cols)) %>%
  pivot_longer(cols = everything(), names_to = "feature", values_to = "value")

ggplot(train_long, aes(x = feature, y = value)) +
  geom_boxplot(na.rm = TRUE) +
  coord_flip() +
  labs(title = "Boxplots of Numeric Features (Train Only)", x = NULL, y = NULL)

# 4.4) Categorical EDA (TRAIN ONLY) — unique values + histograms
suppressPackageStartupMessages({
  library(tidyverse)
  library(stringr)
})

# 1) Identify categorical columns (character or factor), excluding the target
cat_cols <- train_data %>%
  select(where(~ is.character(.x) || is.factor(.x))) %>%
  select(-all_of(TARGET_COL)) %>%
  names()

cat("Categorical columns (excluding target):\n")
print(cat_cols)

# 2) Summary table: #levels, missing count/%, top level + its %
cat_summary <- purrr::map_dfr(cat_cols, function(col) {
  x <- train_data[[col]]

  # Treat NA explicitly
  na_count <- sum(is.na(x))
  n <- length(x)

  # Unique levels (excluding NA) - count
  n_levels <- dplyr::n_distinct(x, na.rm = TRUE)

  # Top frequency (excluding NA)
  tab <- table(x, useNA = "no")
  if (length(tab) == 0) {
    top_level <- NA_character_
    top_n <- 0
    top_pct <- NA_real_
  } else {
    top_level <- names(which.max(tab))
    top_n <- as.integer(max(tab))
    top_pct <- round(100 * top_n / sum(tab), 2)
  }

  tibble(
    column = col,
    n_levels = n_levels,
    na_count = na_count,
    na_pct = round(100 * na_count / n, 2),
    top_level = top_level,
    top_n = top_n,
    top_pct = top_pct
  )
}) %>% arrange(desc(n_levels), desc(na_pct))

cat("\n=== Categorical summary (TRAIN) ===\n")
print(cat_summary)

# 3) Print full frequency tables for each categorical column (optionally truncated)
#    Set TOP_K to limit printing if some columns have many levels (like native_country).
TOP_K <- 20

print_freq <- function(col, top_k = 20) {
  cat("\n----------------------------------\n")
  cat("Column:", col, "\n")
  cat("----------------------------------\n")

  x <- train_data[[col]]
  tab <- sort(table(x, useNA = "ifany"), decreasing = TRUE)
  total_non_na <- sum(tab[names(tab) != "<NA>"])

  # Print top_k counts
  print(head(tab, top_k))

  # If more levels exist, show how many are hidden
  if (length(tab) > top_k) {
    cat("... (", length(tab) - top_k, "more levels not shown)\n", sep = "")
  }

  # Print rare-level stats (excluding NA)
  if (!is.na(total_non_na) && total_non_na > 0) {
    rare_1pct <- sum(tab[names(tab) != "<NA>"] < 0.01 * total_non_na)
    rare_0_5pct <- sum(tab[names(tab) != "<NA>"] < 0.005 * total_non_na)
    cat("Rare levels (<1% of non-NA):", rare_1pct, "\n")
    cat("Very rare levels (<0.5% of non-NA):", rare_0_5pct, "\n")
  }
}

# Print tables
invisible(purrr::walk(cat_cols, ~ print_freq(.x, TOP_K)))

# 4) Histograms (bar charts) for each categorical column
#    NOTE: for high-cardinality columns, we plot top TOP_K + lump others into "OTHER"
plot_cat <- function(col, top_k = 20) {
  dfp <- train_data %>%
    mutate(val = as.character(.data[[col]])) %>%
    mutate(val = ifelse(is.na(val), "<NA>", val)) %>%
    count(val, name = "n") %>%
    arrange(desc(n))

  if (nrow(dfp) > top_k) {
    dfp <- dfp %>%
      mutate(val = ifelse(row_number() <= top_k, val, "OTHER")) %>%
      group_by(val) %>%
      summarise(n = sum(n), .groups = "drop") %>%
      arrange(desc(n))
  }

  ggplot(dfp, aes(x = reorder(val, n), y = n)) +
    geom_col() +
    coord_flip() +
    labs(
      title = paste("Categorical histogram:", col),
      x = col,
      y = "Count"
    )
}

# Plot all categorical columns (one plot per column)
# If you have many columns, you can run a subset: e.g., cat_cols[1:5]
for (col in cat_cols) {
  print(plot_cat(col, TOP_K))
}

# ============================================================
# 5) Feature engineering decisions encoded INSIDE recipes
#    (so they happen fold-safely during CV)
# ============================================================

# Education grouping (report-friendly and stable)
# We'll keep:
# - education_num (numeric)
# - education_group (categorical derived from education)
# Then we drop the original detailed education to avoid redundancy.
education_group_map <- function(education) {
  case_when(
    education %in% c("Preschool", "1st-4th", "5th-6th", "7th-8th", "9th", "10th", "11th", "12th") ~ "School",
    education %in% c("HS-grad") ~ "HighSchool",
    education %in% c("Some-college", "Assoc-acdm", "Assoc-voc") ~ "SomeCollege_Assoc",
    education %in% c("Bachelors") ~ "Bachelors",
    education %in% c("Masters", "Prof-school", "Doctorate") ~ "Graduate",
    TRUE ~ "Other"
  )
}

# ============================================================
# 6) Recipes
# ============================================================

# Common preprocessing steps shared across all models:
# - Convert strings to factors
# - Missing nominal -> "unknown" (informative for workclass/occupation)
# - Simplify native_country to US/non_US/unknown (reduce sparsity)
# - Create hours_category, capital_state, log magnitudes, education_group
# - Collapse rare factor levels -> "other" (fold-trained)
# - One-hot encode categoricals
# - Remove zero-variance columns
#
# NOTE: step_normalize is placed BEFORE step_dummy so only numeric predictors are normalized.

base_rec <- recipe(as.formula(paste(TARGET_COL, "~ .")), data = train_data) %>%
  # Feature engineering (deterministic; safe inside recipe)
  step_mutate(
    # native_country simplified: US vs non_US vs unknown
    native_country_simple = case_when(
      as.character(native_country) == "United-States" ~ "US",
      as.character(native_country) == "unknown" ~ "unknown",
      TRUE ~ "non_US"
    ),

    # Hours categories (keep numeric hours_per_week too)
    hours_category = case_when(
      hours_per_week < 35 ~ "part_time",
      hours_per_week <= 45 ~ "full_time",
      hours_per_week <= 60 ~ "overtime",
      TRUE ~ "extreme_overtime"
    ),

    # Capital state + log magnitudes
    capital_state = case_when(
      capital_gain > 0 ~ "gain",
      capital_loss > 0 ~ "loss",
      TRUE ~ "none"
    ),
    log_capital_gain = log1p(capital_gain),
    log_capital_loss = log1p(capital_loss),

    # Education grouping
    education_group = education_group_map(as.character(education))
  ) %>%
  # Ensure character predictors are treated as categorical
  step_string2factor(all_nominal_predictors()) %>%
  # Missing categoricals -> "unknown" (fold-safe; no stats learned)
  step_unknown(all_nominal_predictors(), new_level = "unknown") %>%
  # Drop raw columns after derived versions to reduce redundancy/skew
  step_rm(native_country, capital_gain, capital_loss, education) %>%
  # Rare levels -> "other" (learned per fold from training portion only)
  step_other(all_nominal_predictors(), threshold = 0.01, other = "other") %>%
  # Numeric imputation (Adult numeric usually has none, but safe & complete)
  step_impute_median(all_numeric_predictors()) %>%
  # Remove zero variance
  step_zv(all_predictors()) %>%
  # One-hot encode
  step_dummy(all_nominal_predictors(), one_hot = TRUE)

# ---- Logistic Regression recipe (z-score numeric only) ----
rec_lr <- base_rec %>%
  # Normalize numeric predictors only (before dummy => dummies unaffected)
  step_normalize(all_numeric_predictors())

# ---- Naive Bayes recipe ----
# NB sometimes benefits from scaling numeric predictors; we apply z-score to numeric only.
# (Dummies are NOT normalized because normalization happens before dummy encoding.)
rec_nb <- base_rec %>%
  step_normalize(all_numeric_predictors())

# ---- Random Forest recipe ----
# Trees do not need normalization.
rec_rf <- base_rec

# ============================================================
# 7) Save a bundle for the modeling/tuning script
# ============================================================

preprocess_bundle <- list(
  meta = list(
    data_path = DATA_PATH,
    target_col = TARGET_COL,
    test_prop = TEST_PROP,
    v_folds = V_FOLDS,
    positive_class = "high_income"
  ),
  split_obj = split_obj,
  train_data = train_data,
  test_data = test_data,
  folds = folds,
  recipes = list(
    logistic = rec_lr,
    naive_bayes = rec_nb,
    random_forest = rec_rf
  )
)

saveRDS(preprocess_bundle, file.path(OUTPUT_DIR, "preprocess_bundle.rds"))
cat("\nSaved bundle:", file.path(OUTPUT_DIR, "preprocess_bundle.rds"), "\n")

# ============================================================
# 8) Modeling + Tuning
# Models: Logistic Regression (glmnet), Naive Bayes, Random Forest (ranger)
# ============================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(tidymodels)
  library(readr)
  library(discrim)
  library(klaR)
})

set.seed(404)

# -----------------------------
# 0) Load bundle
# -----------------------------
BUNDLE_PATH <- "outputs/preprocess_bundle.rds"
bundle <- readRDS(BUNDLE_PATH)

train_data <- bundle$train_data
test_data <- bundle$test_data
folds <- bundle$folds
TARGET_COL <- bundle$meta$target_col

rec_lr <- bundle$recipes$logistic
rec_nb <- bundle$recipes$naive_bayes
rec_rf <- bundle$recipes$random_forest

OUTPUT_DIR <- "outputs"
dir.create(file.path(OUTPUT_DIR, "models"), showWarnings = FALSE, recursive = TRUE)
dir.create(file.path(OUTPUT_DIR, "reports"), showWarnings = FALSE, recursive = TRUE)

# yardstick: event = second level (high_income)
options(yardstick.event_first = FALSE)

# -----------------------------
# 1) Metrics (imbalance-aware)
# -----------------------------
metrics <- metric_set(roc_auc, pr_auc, accuracy, bal_accuracy, f_meas, precision, recall)

ctrl <- control_grid(
  save_pred = TRUE,
  save_workflow = TRUE,
  verbose = TRUE
)

# -----------------------------
# 2) Model specs + grids
# -----------------------------
# Logistic Regression (glmnet)
lr_spec <- logistic_reg(penalty = tune(), mixture = tune()) %>%
  set_engine("glmnet") %>%
  set_mode("classification")

lr_grid <- grid_regular(
  penalty(range = c(-4, 0)), # 1e-4 to 10 (log10 scale)
  mixture(range = c(0, 1)),
  levels = c(penalty = 10, mixture = 5)
)

# Naive Bayes (engine may vary; klaR works in many setups)
nb_spec <- naive_Bayes(smoothness = tune(), Laplace = tune()) %>%
  set_engine("klaR") %>%
  set_mode("classification")

nb_grid <- crossing(
  smoothness = c(0.5, 1, 2),
  Laplace    = c(0, 0.5, 1, 2)
)

# Random Forest (ranger)
rf_spec <- rand_forest(mtry = tune(), min_n = tune(), trees = 500) %>%
  set_engine("ranger", importance = "impurity") %>%
  set_mode("classification")

rf_grid <- grid_space_filling(
  mtry(range = c(2L, 30L)),
  min_n(range = c(2L, 40L)),
  size = 25
)

# -----------------------------
# 3) Workflows
# -----------------------------
wf_lr <- workflow() %>%
  add_recipe(rec_lr) %>%
  add_model(lr_spec)
wf_nb <- workflow() %>%
  add_recipe(rec_nb) %>%
  add_model(nb_spec)
wf_rf <- workflow() %>%
  add_recipe(rec_rf) %>%
  add_model(rf_spec)

# -----------------------------
# 4) Tuning (CV)
# -----------------------------
cat("\n============================\nTuning: Logistic Regression\n============================\n")
res_lr <- tune_grid(
  wf_lr,
  resamples = folds,
  grid = lr_grid,
  metrics = metrics,
  control = ctrl
)

cat("\n============================\nTuning: Naive Bayes\n============================\n")
res_nb <- tune_grid(
  wf_nb,
  resamples = folds,
  grid = nb_grid,
  metrics = metrics,
  control = ctrl
)

cat("\n============================\nTuning: Random Forest\n============================\n")
res_rf <- tune_grid(
  wf_rf,
  resamples = folds,
  grid = rf_grid,
  metrics = metrics,
  control = ctrl
)

# Save tuning results
saveRDS(res_lr, file.path(OUTPUT_DIR, "models", "tune_lr.rds"))
saveRDS(res_nb, file.path(OUTPUT_DIR, "models", "tune_nb.rds"))
saveRDS(res_rf, file.path(OUTPUT_DIR, "models", "tune_rf.rds"))

# -----------------------------
# 5) Pick best by PR AUC (primary)
# -----------------------------
best_lr <- select_best(res_lr, metric = "pr_auc")
best_nb <- select_best(res_nb, metric = "pr_auc")
best_rf <- select_best(res_rf, metric = "pr_auc")

cat("\nBest LR params (PR AUC):\n")
print(best_lr)
cat("\nBest NB params (PR AUC):\n")
print(best_nb)
cat("\nBest RF params (PR AUC):\n")
print(best_rf)

# Finalize workflows
final_wf_lr <- finalize_workflow(wf_lr, best_lr)
final_wf_nb <- finalize_workflow(wf_nb, best_nb)
final_wf_rf <- finalize_workflow(wf_rf, best_rf)

# Fit finalists on full TRAIN (still no test usage for decisions)
fit_lr <- fit(final_wf_lr, data = train_data)
fit_nb <- fit(final_wf_nb, data = train_data)
fit_rf <- fit(final_wf_rf, data = train_data)

saveRDS(fit_lr, file.path(OUTPUT_DIR, "models", "final_lr_fit.rds"))
saveRDS(fit_nb, file.path(OUTPUT_DIR, "models", "final_nb_fit.rds"))
saveRDS(fit_rf, file.path(OUTPUT_DIR, "models", "final_rf_fit.rds"))

# -----------------------------
# 6) Quick CV report tables (top configs)
# -----------------------------
top_k <- function(res, metric = "pr_auc", k = 10) {
  collect_metrics(res) %>%
    filter(.metric == metric) %>%
    arrange(desc(mean)) %>%
    slice_head(n = k)
}

write_csv(top_k(res_lr), file.path(OUTPUT_DIR, "reports", "top10_lr_pr_auc.csv"))
write_csv(top_k(res_nb), file.path(OUTPUT_DIR, "reports", "top10_nb_pr_auc.csv"))
write_csv(top_k(res_rf), file.path(OUTPUT_DIR, "reports", "top10_rf_pr_auc.csv"))

cat("\nDone.\n")
cat("Saved tuning results to outputs/models/ and top-10 tables to outputs/reports/.\n")
cat("Next step: choose thresholds using CV preds, then evaluate once on test.\n")

# ============================================================
# 09) Threshold Selection + Ensemble (Leakage-safe)
# Uses out-of-fold CV predictions from tune_*.rds
# Objective: choose threshold that maximizes F1 on CV predictions
# Then save thresholds + curves for reporting.
# ============================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(tidymodels)
})

set.seed(404)
options(yardstick.event_first = FALSE)

# -----------------------------
# 0) Load saved tuning results
# -----------------------------
res_lr <- readRDS(file.path(OUTPUT_DIR, "models", "tune_lr.rds"))
res_nb <- readRDS(file.path(OUTPUT_DIR, "models", "tune_nb.rds"))
res_rf <- readRDS(file.path(OUTPUT_DIR, "models", "tune_rf.rds"))

# Pick best config by PR AUC (same rule you used for finalists)
best_lr <- select_best(res_lr, metric = "pr_auc")
best_nb <- select_best(res_nb, metric = "pr_auc")
best_rf <- select_best(res_rf, metric = "pr_auc")

# -----------------------------
# 1) Helpers
# -----------------------------
make_class <- function(p, threshold) {
  factor(
    ifelse(p >= threshold, "high_income", "low_income"),
    levels = c("low_income", "high_income")
  )
}

# Metrics we care about during threshold selection
thresh_metrics <- metric_set(f_meas, precision, recall, bal_accuracy, accuracy)

threshold_curve <- function(df, prob_col = ".pred_high_income",
                            truth_col = "income_level",
                            thresholds = seq(0.01, 0.99, by = 0.01)) {
  map_dfr(thresholds, function(t) {
    tmp <- df %>%
      dplyr::mutate(.pred_class = make_class(.data[[prob_col]], t))

    m <- thresh_metrics(
      tmp,
      truth = !!sym(truth_col),
      estimate = .pred_class
    )

    m %>%
      dplyr::mutate(threshold = t) %>%
      dplyr::select(threshold, .metric, .estimator, .estimate)
  }) %>%
    pivot_wider(names_from = .metric, values_from = .estimate)
}

pick_best_threshold_f1 <- function(curve_df) {
  # Primary: max F1
  # Tie-breaker: higher recall, then higher balanced accuracy
  curve_df %>%
    arrange(desc(f_meas), desc(recall), desc(bal_accuracy)) %>%
    slice(1)
}

# Extract out-of-fold predictions for the best config.
# We rely on `.row` + `id` to align across models.
oof_best_preds <- function(res, best_params, model_name) {
  collect_predictions(res) %>%
    semi_join(best_params, by = names(best_params)) %>%
    dplyr::select(id, .row, income_level, .pred_high_income) %>%
    dplyr::rename(!!paste0("p_", model_name) := .pred_high_income)
}

# -----------------------------
# 2) Threshold selection per model (OOF CV)
# -----------------------------
oof_lr <- oof_best_preds(res_lr, best_lr, "lr")
oof_nb <- oof_best_preds(res_nb, best_nb, "nb")
oof_rf <- oof_best_preds(res_rf, best_rf, "rf")

curve_lr <- threshold_curve(oof_lr %>% dplyr::rename(.pred_high_income = p_lr))
curve_nb <- threshold_curve(oof_nb %>% dplyr::rename(.pred_high_income = p_nb))
curve_rf <- threshold_curve(oof_rf %>% dplyr::rename(.pred_high_income = p_rf))

bestT_lr <- pick_best_threshold_f1(curve_lr) %>% dplyr::mutate(model = "LogReg")
bestT_nb <- pick_best_threshold_f1(curve_nb) %>% dplyr::mutate(model = "NaiveBayes")
bestT_rf <- pick_best_threshold_f1(curve_rf) %>% dplyr::mutate(model = "RandomForest")

best_thresholds_single <- bind_rows(bestT_lr, bestT_nb, bestT_rf) %>%
  dplyr::select(model, threshold, f_meas, precision, recall, bal_accuracy, accuracy)

print(best_thresholds_single)

write_csv(curve_lr, file.path(OUTPUT_DIR, "reports", "threshold_curve_lr.csv"))
write_csv(curve_nb, file.path(OUTPUT_DIR, "reports", "threshold_curve_nb.csv"))
write_csv(curve_rf, file.path(OUTPUT_DIR, "reports", "threshold_curve_rf.csv"))
write_csv(best_thresholds_single, file.path(OUTPUT_DIR, "reports", "best_thresholds_single_models.csv"))

# -----------------------------
# 3) Ensemble options (soft voting)
#    Ensemble is built from OOF predictions (still leakage-safe).
# -----------------------------

# Align OOF predictions by id + row
oof_ens <- oof_lr %>%
  inner_join(oof_rf, by = c("id", ".row", "income_level")) %>%
  inner_join(oof_nb, by = c("id", ".row", "income_level")) %>%
  dplyr::mutate(
    p_avg = (p_lr + p_rf + p_nb) / 3,
    # Optional weighted average (edit weights if you want)
    p_wavg = (0.33 * p_lr + 0.34 * p_rf + 0.33 * p_nb)
  )

curve_avg <- threshold_curve(oof_ens %>% dplyr::rename(.pred_high_income = p_avg))
curve_wavg <- threshold_curve(oof_ens %>% dplyr::rename(.pred_high_income = p_wavg))

bestT_avg <- pick_best_threshold_f1(curve_avg) %>% dplyr::mutate(model = "Ensemble_Avg")
bestT_wavg <- pick_best_threshold_f1(curve_wavg) %>% dplyr::mutate(model = "Ensemble_Weighted")

best_thresholds_ens <- bind_rows(bestT_avg, bestT_wavg) %>%
  dplyr::select(model, threshold, f_meas, precision, recall, bal_accuracy, accuracy)

print(best_thresholds_ens)

write_csv(curve_avg, file.path(OUTPUT_DIR, "reports", "threshold_curve_ensemble_avg.csv"))
write_csv(curve_wavg, file.path(OUTPUT_DIR, "reports", "threshold_curve_ensemble_wavg.csv"))
write_csv(best_thresholds_ens, file.path(OUTPUT_DIR, "reports", "best_thresholds_ensembles.csv"))

# -----------------------------
# 4) Save a compact object for later final test evaluation
# -----------------------------
threshold_bundle <- list(
  best_params = list(lr = best_lr, nb = best_nb, rf = best_rf),
  best_thresholds_single = best_thresholds_single,
  best_thresholds_ens = best_thresholds_ens
)

saveRDS(threshold_bundle, file.path(OUTPUT_DIR, "models", "threshold_bundle.rds"))
cat("\nSaved:", file.path(OUTPUT_DIR, "models", "threshold_bundle.rds"), "\n")
cat("Next step: final evaluation on test using the chosen thresholds (no re-selection).\n")

# ============================================================
# Final TEST Evaluation (Conflict-safe + Version-robust)
# - Uses frozen thresholds from threshold_bundle.rds if present
# - Evaluates single models + ensembles on TEST
# - Writes metrics + confusion matrices to CSV
# ============================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(tidymodels)
  library(readr)
})

set.seed(404)
options(yardstick.event_first = FALSE) # positive class = "high_income"

# -----------------------------
# 0) Load TEST data + final fits
# -----------------------------
test_data <- readr::read_csv(
  file.path(OUTPUT_DIR, "data", "test_data.csv"),
  show_col_types = FALSE
)

fit_lr <- readRDS(file.path(OUTPUT_DIR, "models", "final_lr_fit.rds"))
fit_nb <- readRDS(file.path(OUTPUT_DIR, "models", "final_nb_fit.rds"))
fit_rf <- readRDS(file.path(OUTPUT_DIR, "models", "final_rf_fit.rds"))

# -----------------------------
# 1) Load frozen thresholds (recommended)
# -----------------------------
THRESH_BUNDLE_PATH <- file.path(OUTPUT_DIR, "models", "threshold_bundle.rds")

if (file.exists(THRESH_BUNDLE_PATH)) {
  th <- readRDS(THRESH_BUNDLE_PATH)

  # Expecting tables with columns: model, threshold
  t_single <- th$best_thresholds_single
  t_ens <- th$best_thresholds_ens

  getT <- function(tbl, name, fallback) {
    v <- tbl %>%
      dplyr::filter(model == name) %>%
      dplyr::pull(threshold)

    if (length(v) == 0) fallback else v[[1]]
  }

  T_LR <- getT(t_single, "LogReg", 0.59)
  T_NB <- getT(t_single, "NaiveBayes", 0.40)
  T_RF <- getT(t_single, "RandomForest", 0.55)
  T_EA <- getT(t_ens, "Ensemble_Avg", 0.48)
  T_EW <- getT(t_ens, "Ensemble_Weighted", 0.48)
} else {
  # Fallback thresholds
  T_LR <- 0.59
  T_NB <- 0.40
  T_RF <- 0.55
  T_EA <- 0.48
  T_EW <- 0.48
}

# -----------------------------
# 2) Helpers (conflict-safe + robust)
# -----------------------------
make_class <- function(p, threshold) {
  factor(
    ifelse(p >= threshold, "high_income", "low_income"),
    levels = c("low_income", "high_income")
  )
}

# Robust confusion-matrix export (works even if as_tibble(conf_mat) isn't supported)
conf_mat_df <- function(cm) {
  out <- base::as.data.frame(cm$table, stringsAsFactors = FALSE)
  names(out) <- c("truth", "estimate", "n")
  out
}

# Vector-based evaluation avoids metric_set() / tidyselect column-selection pitfalls
eval_from_probs <- function(df, model_name, prob_col, threshold) {
  # Guard common paste typo: ".pred_high_income," or "p_avg,"
  prob_col <- sub(",\\s*$", "", prob_col)

  if (!("income_level" %in% names(df))) {
    stop("Missing required column: income_level", call. = FALSE)
  }
  if (!(prob_col %in% names(df))) {
    stop(sprintf(
      "Column '%s' not found for model '%s'. Available columns: %s",
      prob_col, model_name, paste(names(df), collapse = ", ")
    ), call. = FALSE)
  }

  df2 <- df %>%
    dplyr::mutate(
      income_level = factor(income_level, levels = c("low_income", "high_income")),
      .pred_high_income = as.numeric(.data[[prob_col]]),
      .pred_class = make_class(.pred_high_income, threshold)
    )

  truth <- df2$income_level
  prob <- df2$.pred_high_income
  pred <- df2$.pred_class

  metrics_tbl <- tibble::tibble(
    model = model_name,
    threshold = threshold,
    .metric = c("roc_auc", "pr_auc", "f_meas", "precision", "recall", "bal_accuracy", "accuracy"),
    .estimate = c(
      yardstick::roc_auc_vec(truth, prob, event_level = "second"),
      yardstick::pr_auc_vec(truth, prob, event_level = "second"),
      yardstick::f_meas_vec(truth, pred, event_level = "second"),
      yardstick::precision_vec(truth, pred, event_level = "second"),
      yardstick::recall_vec(truth, pred, event_level = "second"),
      yardstick::bal_accuracy_vec(truth, pred, event_level = "second"),
      yardstick::accuracy_vec(truth, pred)
    )
  )

  cm <- yardstick::conf_mat(df2, truth = income_level, estimate = .pred_class)

  list(metrics = metrics_tbl, conf_mat = cm)
}

# -----------------------------
# 3) Predict probabilities on TEST
# -----------------------------
pred_lr <- predict(fit_lr, test_data, type = "prob") %>%
  dplyr::bind_cols(test_data %>% dplyr::select(income_level))

pred_nb <- predict(fit_nb, test_data, type = "prob") %>%
  dplyr::bind_cols(test_data %>% dplyr::select(income_level))

pred_rf <- predict(fit_rf, test_data, type = "prob") %>%
  dplyr::bind_cols(test_data %>% dplyr::select(income_level))

stopifnot(".pred_high_income" %in% names(pred_lr))
stopifnot(".pred_high_income" %in% names(pred_nb))
stopifnot(".pred_high_income" %in% names(pred_rf))

# -----------------------------
# 4) Evaluate single models on TEST using frozen thresholds
# -----------------------------
res_lr <- eval_from_probs(pred_lr, "LogReg", ".pred_high_income", T_LR)
res_nb <- eval_from_probs(pred_nb, "NaiveBayes", ".pred_high_income", T_NB)
res_rf <- eval_from_probs(pred_rf, "RandomForest", ".pred_high_income", T_RF)

# -----------------------------
# 5) Ensembles on TEST (soft voting)
# -----------------------------
ens_df <- test_data %>%
  dplyr::select(income_level) %>%
  dplyr::mutate(
    p_lr = pred_lr$.pred_high_income,
    p_nb = pred_nb$.pred_high_income,
    p_rf = pred_rf$.pred_high_income,
    p_avg = (p_lr + p_nb + p_rf) / 3,
    # same weights as CV script
    p_wavg = (0.33 * p_lr + 0.33 * p_nb + 0.34 * p_rf)
  )

res_ea <- eval_from_probs(ens_df, "Ensemble_Avg", "p_avg", T_EA)
res_ew <- eval_from_probs(ens_df, "Ensemble_Weighted", "p_wavg", T_EW)

# -----------------------------
# 6) Save report-friendly outputs
# -----------------------------
all_metrics <- dplyr::bind_rows(
  res_lr$metrics, res_nb$metrics, res_rf$metrics,
  res_ea$metrics, res_ew$metrics
) %>%
  dplyr::arrange(
    model,
    match(.metric, c("roc_auc", "pr_auc", "f_meas", "precision", "recall", "bal_accuracy", "accuracy"))
  )

readr::write_csv(all_metrics, file.path(OUTPUT_DIR, "reports", "final_test_metrics.csv"))

# Confusion matrices as CSV (robust across package versions)
readr::write_csv(conf_mat_df(res_lr$conf_mat), file.path(OUTPUT_DIR, "reports", "confmat_test_lr.csv"))
readr::write_csv(conf_mat_df(res_nb$conf_mat), file.path(OUTPUT_DIR, "reports", "confmat_test_nb.csv"))
readr::write_csv(conf_mat_df(res_rf$conf_mat), file.path(OUTPUT_DIR, "reports", "confmat_test_rf.csv"))
readr::write_csv(conf_mat_df(res_ea$conf_mat), file.path(OUTPUT_DIR, "reports", "confmat_test_ensemble_avg.csv"))
readr::write_csv(conf_mat_df(res_ew$conf_mat), file.path(OUTPUT_DIR, "reports", "confmat_test_ensemble_wavg.csv"))

# Print a compact summary to console
cat("\n================ FINAL TEST METRICS (Key Rows) ================\n")
print(
  all_metrics %>%
    dplyr::filter(.metric %in% c("roc_auc", "pr_auc", "f_meas", "precision", "recall", "bal_accuracy", "accuracy")) %>%
    tidyr::pivot_wider(names_from = .metric, values_from = .estimate) %>%
    dplyr::arrange(dplyr::desc(pr_auc))
)

cat("\nSaved:\n")
cat(" - outputs/reports/final_test_metrics.csv\n")
cat(" - outputs/reports/confmat_test_*.csv\n")
cat("Done.\n")
