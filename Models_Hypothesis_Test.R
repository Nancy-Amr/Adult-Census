# ============================================================================
# STATISTICAL MODEL COMPARISON TESTS
# Methods: Welch's t-test (pairwise) + ANOVA (overall)
# ============================================================================

library(tidyverse)
library(tidymodels)

# Load necessary data
OUTPUT_DIR <- "outputs"

# PREPARE DATA: Extract CV Performance Metrics
# Load CV results
res_lr <- readRDS(file.path(OUTPUT_DIR, "models", "tune_lr.rds"))
res_nb <- readRDS(file.path(OUTPUT_DIR, "models", "tune_nb.rds"))
res_rf <- readRDS(file.path(OUTPUT_DIR, "models", "tune_rf.rds"))

# Get best configurations
best_lr <- select_best(res_lr, metric = "pr_auc")
best_nb <- select_best(res_nb, metric = "pr_auc")
best_rf <- select_best(res_rf, metric = "pr_auc")

# Extract fold-level metrics for best configurations
get_fold_metrics <- function(res, best_params, metric = "f_meas") {
  collect_metrics(res, summarize = FALSE) %>%
    semi_join(best_params, by = names(best_params)) %>%
    filter(.metric == metric) %>%
    select(id, .estimate) %>%
    pull(.estimate)
}

# Get F1 scores across folds for each model
f1_lr <- get_fold_metrics(res_lr, best_lr, "f_meas")
f1_nb <- get_fold_metrics(res_nb, best_nb, "f_meas")
f1_rf <- get_fold_metrics(res_rf, best_rf, "f_meas")

# Summary statistics for each model
models_summary <- data.frame(
  Model = c("Logistic Regression", "Naive Bayes", "Random Forest"),
  Mean = c(mean(f1_lr), mean(f1_nb), mean(f1_rf)),
  SD = c(sd(f1_lr), sd(f1_nb), sd(f1_rf)),
  Median = c(median(f1_lr), median(f1_nb), median(f1_rf)),
  Min = c(min(f1_lr), min(f1_nb), min(f1_rf)),
  Max = c(max(f1_lr), max(f1_nb), max(f1_rf))
)

print(models_summary)

# ----------------------------------------------------------------------------
# 1. ANOVA (Overall Comparison of All Models)
# ----------------------------------------------------------------------------

# Prepare data in long format for ANOVA
anova_data <- data.frame(
  F1_Score = c(f1_lr, f1_nb, f1_rf),
  Model = factor(rep(c("LogReg", "NaiveBayes", "RandomForest"), 
                     each = length(f1_lr)))
)

# Perform ANOVA
anova_result <- aov(F1_Score ~ Model, data = anova_data)
anova_summary <- summary(anova_result)

print(anova_summary)

# Extract ANOVA statistics
anova_table <- anova_summary[[1]]
f_statistic <- anova_table["Model", "F value"]
p_value <- anova_table["Model", "Pr(>F)"]
df_between <- anova_table["Model", "Df"]
df_within <- anova_table["Residuals", "Df"]
ss_between <- anova_table["Model", "Sum Sq"]
ss_within <- anova_table["Residuals", "Sum Sq"]

cat("ANOVA Summary:\n")
cat(sprintf("  F-statistic: %.4f\n", f_statistic))
cat(sprintf("  Degrees of freedom: Between groups = %d, Within groups = %d\n", 
            df_between, df_within))
cat(sprintf("  Sum of Squares: Between = %.6f, Within = %.6f\n", ss_between, ss_within))
cat(sprintf("  p-value: %.6f\n", p_value))
cat("\n")

if (p_value < 0.05) {
  cat("✓ ANOVA RESULT: SIGNIFICANT\n")
  cat("At least one model performs significantly differently from the others.\n\n")
  cat("Proceeding to post-hoc pairwise comparisons using Tukey HSD...\n\n")
  
  # Post-hoc Tukey HSD test for pairwise comparisons with correction
  cat(rep("=", 70), "\n", sep = "")
  cat("TUKEY HSD POST-HOC TEST: Pairwise Comparisons with Adjustment\n")
  cat(rep("=", 70), "\n", sep = "")
  cat("\n")
  
  cat("Tukey's Honest Significant Difference (HSD) test\n")
  cat("Controls family-wise error rate across multiple comparisons\n")
  cat("Adjusted significance level for multiple testing\n\n")
  
  tukey_result <- TukeyHSD(anova_result)
  
  print(tukey_result)
  cat("\n")
  
  # Extract and format Tukey results
  tukey_df <- as.data.frame(tukey_result$Model)
  tukey_df$Comparison <- rownames(tukey_df)
  tukey_df <- tukey_df %>%
    mutate(
      Significant = `p adj` < 0.05,
      Interpretation = case_when(
        `p adj` >= 0.05 ~ "No significant difference",
        diff > 0 ~ "First model significantly better",
        diff < 0 ~ "Second model significantly better"
      )
    ) %>%
    select(Comparison, diff, lwr, upr, `p adj`, Significant, Interpretation)
  
  colnames(tukey_df) <- c("Comparison", "Mean_Diff", "CI_Lower", "CI_Upper", 
                          "P_Value_Adjusted", "Significant", "Interpretation")
  
  tukey_df <- tukey_df %>%
    mutate(across(c(Mean_Diff, CI_Lower, CI_Upper, P_Value_Adjusted), 
                  ~round(.x, 6)))
  
  cat("Tukey HSD Results Summary:\n")
  print(tukey_df)
  cat("\n")
  
  # Detailed interpretation
  cat("Detailed Tukey HSD Results:\n")
  cat(rep("-", 70), "\n", sep = "")
  
  for (i in 1:nrow(tukey_df)) {
    row <- tukey_df[i, ]
    cat(sprintf("\n%s:\n", row$Comparison))
    cat(sprintf("  Mean difference: %.4f\n", row$Mean_Diff))
    cat(sprintf("  95%% CI: [%.4f, %.4f]\n", row$CI_Lower, row$CI_Upper))
    cat(sprintf("  Adjusted p-value: %.6f\n", row$P_Value_Adjusted))
    
    if (row$Significant) {
      cat(sprintf("  ✓ SIGNIFICANT: %s\n", row$Interpretation))
    } else {
      cat(sprintf("  ✗ NOT SIGNIFICANT: %s\n", row$Interpretation))
    }
  }
  
  # Save Tukey results
  write_csv(tukey_df, file.path(OUTPUT_DIR, "reports", "tukey_hsd_results.csv"))
  
} else {
  cat("✗ ANOVA RESULT: NOT SIGNIFICANT\n")
  cat("No significant differences detected among the three models.\n")
  cat("All models perform similarly based on F1 scores.\n")
}

# ANOVA summary
anova_summary_df <- data.frame(
  Test = "One-Way ANOVA",
  F_Statistic = round(f_statistic, 4),
  DF_Between = df_between,
  DF_Within = df_within,
  P_Value = round(p_value, 6),
  Significant = p_value < 0.05,
  Interpretation = ifelse(
    p_value < 0.05,
    "At least one model differs significantly",
    "No significant differences among models"
  ),
  stringsAsFactors = FALSE
)

anova_summary_df

# Boxplot for ANOVA results
ggplot(anova_data, aes(x = Model, y = F1_Score, fill = Model)) +
  geom_boxplot(alpha = 0.7) +                # Show spread of F1 scores
  geom_jitter(width = 0.1, alpha = 0.5) +    # Add individual fold points
  scale_fill_brewer(palette = "Set3") + 
  labs(
    title = "ANOVA: F1 Score Distribution Across Models",
    subtitle = "Shows variability and differences in mean F1 scores",
    x = "Model",
    y = "F1 Score"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    legend.position = "none"
  )

# ----------------------------------------------------------------------------
# 2. WELCH'S T-TEST (Pairwise Comparisons)
# ----------------------------------------------------------------------------

# Function to perform Welch's t-test
perform_welch_ttest <- function(scores1, scores2, model1_name, model2_name) {
  test_result <- t.test(scores1, scores2, var.equal = FALSE, paired = FALSE)
  
  mean_diff <- mean(scores1) - mean(scores2)
  var1 <- var(scores1)
  var2 <- var(scores2)
  
  return(data.frame(
    Model1 = model1_name,
    Model2 = model2_name,
    Mean_Model1 = round(mean(scores1), 4),
    Mean_Model2 = round(mean(scores2), 4),
    Var_Model1 = round(var1, 6),
    Var_Model2 = round(var2, 6),
    Mean_Diff = round(mean_diff, 4),
    t_Statistic = round(test_result$statistic, 4),
    DF = round(test_result$parameter, 2),
    P_Value = round(test_result$p.value, 6),
    Significant = test_result$p.value < 0.05,
    CI_Lower = round(test_result$conf.int[1], 4),
    CI_Upper = round(test_result$conf.int[2], 4),
    Interpretation = ifelse(
      test_result$p.value >= 0.05, "No significant difference",
      ifelse(mean_diff > 0, paste(model1_name, "significantly better"),
             paste(model2_name, "significantly better"))
    ),
    stringsAsFactors = FALSE
  ))
}

# Perform all pairwise Welch's t-tests
welch_results <- bind_rows(
  perform_welch_ttest(f1_lr, f1_nb, "LogReg", "NaiveBayes"),
  perform_welch_ttest(f1_lr, f1_rf, "LogReg", "RandomForest"),
  perform_welch_ttest(f1_nb, f1_rf, "NaiveBayes", "RandomForest")
)

print(welch_results %>% select(Model1, Model2, Mean_Diff, t_Statistic, DF, P_Value, Significant))

# Print detailed interpretation for each comparison
for (i in 1:nrow(welch_results)) {
  row <- welch_results[i, ]
  cat(sprintf("\n%s vs %s:\n", row$Model1, row$Model2))
  cat(sprintf("  Mean F1 scores: %.4f vs %.4f (difference: %.4f)\n", 
              row$Mean_Model1, row$Mean_Model2, row$Mean_Diff))
  cat(sprintf("  Variances: %.6f vs %.6f\n", row$Var_Model1, row$Var_Model2))
  cat(sprintf("  t-statistic: %.4f (df = %.2f)\n", row$t_Statistic, row$DF))
  cat(sprintf("  p-value: %.6f\n", row$P_Value))
  cat(sprintf("  95%% Confidence Interval: [%.4f, %.4f]\n", row$CI_Lower, row$CI_Upper))
  
  if (row$Significant) {
    cat(sprintf("  ✓ SIGNIFICANT: %s\n", row$Interpretation))
  } else {
    cat(sprintf("  ✗ NOT SIGNIFICANT: %s\n", row$Interpretation))
  }
}
welch_plot <- welch_results %>%
  mutate(
    Comparison = paste(Model1, "vs", Model2),
    Significance = ifelse(Significant, "Significant", "Not Significant")
  )

ggplot(welch_plot, aes(x = reorder(Comparison, Mean_Diff), y = Mean_Diff, fill = Significance)) +
  geom_col() +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  coord_flip() +
  scale_fill_manual(values = c("Significant" = "lightgreen", "Not Significant" = "red")) +
  labs(
    title = "Pairwise Welch's T-Test: Mean F1 Differences",
    subtitle = "Positive → first model better | Negative → second model better",
    x = "Model Comparison",
    y = "Mean F1 Difference",
    fill = "Significance"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    legend.position = "bottom"
  )
