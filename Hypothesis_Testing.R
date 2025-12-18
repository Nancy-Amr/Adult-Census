# ============================================================================
# HYPOTHESIS TESTING FOR Income Level Prediction
# ============================================================================

# Load required libraries
library(dplyr)      # For data manipulation
library(ggplot2)    # For visualization

# ----------------------------------------------------------------------------
# 1. LOAD THE DATA
# ----------------------------------------------------------------------------

# Read the CSV file
df <- read.csv("original_cleaned.csv", stringsAsFactors = FALSE)

# Display basic information about the dataset
cat("Dataset dimensions:", dim(df), "\n")
cat("Target variable:", unique(df$income_level), "\n\n")

# ----------------------------------------------------------------------------
# 2. DEFINE FEATURE LISTS
# ----------------------------------------------------------------------------

# List all categorical features to test
categorical_features <- c(
  "workclass", "education", "marital_status", "occupation",
  "relationship", "race", "sex", "age_group", "education_simple",
  "hours_category", "has_capital", "country_region"
)

# List all numerical features to test
numerical_features <- c(
  "age", "education_num", "capital_gain", "capital_loss", "hours_per_week"
)

# ----------------------------------------------------------------------------
# 3. CHI-SQUARE TESTS FOR CATEGORICAL FEATURES
# ----------------------------------------------------------------------------

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
cat("Performing Chi-Square Tests for Categorical Features...\n")
cat("=" , rep("=", 60), "\n", sep = "")

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

cat("\n")

# ----------------------------------------------------------------------------
# 4. T-TESTS FOR NUMERICAL FEATURES
# ----------------------------------------------------------------------------

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
cat("Performing T-Tests for Numerical Features...\n")
cat("=" , rep("=", 60), "\n", sep = "")

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

cat("\n")

# ----------------------------------------------------------------------------
# 5. DISPLAY SUMMARY STATISTICS
# ----------------------------------------------------------------------------

cat("\n")
cat("SUMMARY OF HYPOTHESIS TESTS\n")
cat("=" , rep("=", 60), "\n", sep = "")
cat(sprintf("Total features tested: %d\n", nrow(all_results)))
cat(sprintf("Significant features (p < 0.05): %d\n", sum(all_results$Significant)))
cat(sprintf("Non-significant features: %d\n", sum(!all_results$Significant)))
cat("\n")

# ----------------------------------------------------------------------------
# 6. VISUALIZATION: P-VALUES BY FEATURE
# ----------------------------------------------------------------------------

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
# ----------------------------------------------------------------------------

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
# 7. EFFECT SIZE CALCULATION (For CHI-SQUARE Tests)
# ----------------------------------------------------------------------------

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

# Example loop for your categorical features
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

