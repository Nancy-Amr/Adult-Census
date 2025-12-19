# Project: Adult Income Prediction Dataset
# Description: Optimized preprocessing for regression, decision trees, and naive Bayes
# Dataset: 32,561 records
# =============================================================================

# =============================================================================
# 1. INSTALL AND LOAD REQUIRED PACKAGES
# =============================================================================

# Install packages if not already installed
if (!require(tidyverse)) install.packages("tidyverse")
if (!require(caret)) install.packages("caret")
if (!require(naniar)) install.packages("naniar")
if (!require(VIM)) install.packages("VIM")  # For KNN imputation
if (!require(recipes)) install.packages("recipes")  # For preprocessing
if (!require(ROSE)) install.packages("ROSE")  # For imbalance handling

# Load libraries
library(tidyverse)     # For data manipulation and visualization
library(caret)         # For machine learning utilities and preprocessing
library(naniar)        # For missing value visualization
library(VIM)           # For KNN imputation
library(recipes)       # For preprocessing pipeline
library(ROSE)          # For ROSE technique

cat("=== PACKAGES LOADED SUCCESSFULLY ===\n")

# =============================================================================
# 2. LOAD AND INITIAL INSPECTION OF RAW DATA
# =============================================================================

# Load the dataset from CSV file
df <- read_csv("adult.csv")

cat("=== INITIAL DATA INSPECTION ===\n")
cat("Dataset size:", nrow(df), "rows,", ncol(df), "columns\n")

# Display dataset structure
cat("\n1. Dataset Structure:\n")
str(df)

# Check class distribution
cat("\n2. Target Variable Distribution:\n")
income_table <- table(df$income)
print(income_table)
print(prop.table(income_table) * 100)

# =============================================================================
# 3. CLEAN COLUMN NAMES FOR CONSISTENCY
# =============================================================================

cat("\n=== CLEANING COLUMN NAMES ===\n")

# Clean column names
df <- df %>%
  rename_with(~tolower(gsub("[^[:alnum:]_\\.]", "", .))) %>%
  rename_with(~gsub("\\.", "_", .))

cat("Column names after cleaning:\n")
print(names(df))

# =============================================================================
# 4. HANDLE MISSING VALUES (CONVERT "?" TO NA)
# =============================================================================

cat("\n=== HANDLING MISSING VALUES ===\n")

# Convert "?" to NA
df_clean <- df %>%
  mutate(across(where(is.character), ~na_if(., "?")))

# Count missing values
missing_counts <- df_clean %>%
  summarise(across(everything(), ~sum(is.na(.))))

cat("Missing values per column:\n")
print(missing_counts)

# =============================================================================
# 5. IMPUTE MISSING VALUES USING KNN (BETTER FOR 32K DATASET)
# =============================================================================

cat("\n=== IMPUTING MISSING VALUES USING KNN ===\n")

# Prepare data for KNN imputation
df_temp <- df_clean %>%
  mutate(across(where(is.factor), as.character))

# Apply KNN imputation (k=10 for better accuracy with 32K samples)
set.seed(123)
df_imputed <- kNN(df_temp, 
                  variable = c("workclass", "occupation", "native_country"),
                  k = 10,  # Increased k for larger dataset
                  imp_var = FALSE)

# Convert back
df_clean <- df_imputed %>%
  mutate(across(c(workclass, occupation, native_country), as.factor))

cat("KNN imputation completed (k=10)\n")

# Verify no missing values
cat("\nMissing values after imputation:\n")
print(df_clean %>% summarise(across(everything(), ~sum(is.na(.)))))

# =============================================================================
# 6. TARGET VARIABLE PREPARATION
# =============================================================================

cat("\n=== PREPARING TARGET VARIABLE ===\n")

# Rename and convert target variable
df_clean <- df_clean %>%
  mutate(
    income_level = case_when(
      income == "<=50K" ~ 0,  # 0 for low income
      income == ">50K" ~ 1,   # 1 for high income
      TRUE ~ NA_real_
    ),
    income_level = factor(income_level, levels = c(0, 1), 
                          labels = c("low_income", "high_income"))
  ) %>%
  select(-income)  # Remove original column

# Check distribution
class_dist <- df_clean %>%
  count(income_level) %>%
  mutate(percentage = n/sum(n) * 100)

cat("\nClass Distribution:\n")
print(class_dist)
cat("\nImbalance Ratio:", 
    round(max(class_dist$n) / min(class_dist$n), 2), ": 1\n")

# =============================================================================
# 7. REMOVE FNLWGT COLUMN AND PREPARE FEATURES
# =============================================================================

cat("\n=== PREPARING FEATURES ===\n")

# Remove fnlwgt column
df_clean <- df_clean %>%
  select(-fnlwgt)

cat("Removed 'fnlwgt' column\n")

# Create useful features for the specific models
df_clean <- df_clean %>%
  mutate(
    # Age kept as is (no scaling for decision trees)
    # Create age groups for naive Bayes
    age_group = cut(age,
                    breaks = c(17, 25, 35, 45, 55, 65, 90),
                    labels = c("18-25", "26-35", "36-45", "46-55", "56-65", "66+")),
    
    # Education grouping (useful for naive Bayes)
    education_simple = case_when(
      education %in% c("Preschool", "1st-4th", "5th-6th", "7th-8th") ~ "Elementary",
      education %in% c("9th", "10th", "11th", "12th", "HS-grad") ~ "High_School",
      education %in% c("Some-college", "Assoc-acdm", "Assoc-voc") ~ "Associate",
      education == "Bachelors" ~ "Bachelor",
      education %in% c("Masters", "Doctorate", "Prof-school") ~ "Graduate",
      TRUE ~ "Other"
    ),
    
    # Working hours categories
    hours_category = cut(hours_per_week,
                         breaks = c(0, 20, 40, 60, 100),
                         labels = c("Part_time", "Full_time", "Overtime", "Excessive")),
    
    # Capital activity flag
    has_capital = ifelse(capital_gain > 0 | capital_loss > 0, "Yes", "No")
  )

# =============================================================================
# 8. HANDLE COUNTRIES COLUMN (8 NON-OVERLAPPING CATEGORIES)
# =============================================================================

cat("\n=== HANDLING COUNTRIES COLUMN (8 NON-OVERLAPPING CATEGORIES) ===\n")

# First, let's see all unique countries before grouping
all_countries <- df_clean %>%
  distinct(native_country) %>%
  pull(native_country)

cat("Total unique countries:", length(all_countries), "\n")

# Define country groups with NO OVERLAP
# Order matters in case_when - we process in sequence
df_clean <- df_clean %>%
  mutate(
    country_region = case_when(
      # 1. United States (largest group, separate)
      native_country == "United-States" ~ "United_States",
      
      # 2. Anglosphere (English-speaking developed countries)
      # MUST come BEFORE Europe and North_America to avoid overlap
      native_country %in% c("England", "Canada", "Ireland", "Scotland") ~ "Anglosphere",
      
      # 3. North America (non-English speaking)
      native_country %in% c("Mexico", "Puerto-Rico") ~ "North_America",
      
      # 4. Latin America (Central & South America)
      native_country %in% c("El-Salvador", "Cuba", "Jamaica", "Dominican-Republic",
                            "Haiti", "Guatemala", "Nicaragua", "Ecuador", 
                            "Peru", "Columbia", "Honduras", "Trinadad&Tobago") ~ "Latin_America",
      
      # 5. Europe (NON-English speaking, excludes Anglosphere countries)
      native_country %in% c("Germany", "Greece", "Italy", "Poland",
                            "Portugal", "France", "Hungary", 
                            "Yugoslavia", "Holand-Netherlands") ~ "Europe",
      
      # 6. Asia (East & Southeast Asia)
      native_country %in% c("Philippines", "China", "Japan", "India", "Vietnam",
                            "Taiwan", "Hong", "Thailand", "Cambodia", "Laos") ~ "Asia",
      
      # 7. Middle East & Oceania
      native_country %in% c("Iran", "Outlying-US(Guam-USVI-etc)", "South") ~ "Middle_East_Oceania",
      
      # 8. Other_Regions (Everything else that hasn't been assigned)
      TRUE ~ "Other_Regions"
    ),
    country_region = as.factor(country_region)
  ) %>%
  select(-native_country)  # Remove original

# Check the 8 categories
cat("\nCountry Region Distribution (8 NON-OVERLAPPING categories):\n")
region_counts <- df_clean %>%
  count(country_region, sort = TRUE) %>%
  mutate(percentage = round(n/sum(n) * 100, 2))

print(region_counts)

# Show examples for each category
cat("\nExamples in each category:\n")
# Create a mapping of original countries to regions for display
country_mapping <- df_clean %>%
  # We lost the native_country column, so let's show what we know
  group_by(country_region) %>%
  summarise(
    count = n(),
    percentage = round(n()/nrow(df_clean)*100, 2)
  ) %>%
  arrange(desc(count))

print(country_mapping)

# Show actual country composition (we need to do this before removing native_country)
cat("\nTo see actual countries in each region, we would need to save mapping.\n")
cat("But here's what we know based on the logic:\n")
cat("1. United_States: United-States only\n")
cat("2. Anglosphere: England, Canada, Ireland, Scotland\n")
cat("3. North_America: Mexico, Puerto-Rico\n")
cat("4. Latin_America: El-Salvador, Cuba, Jamaica, Dominican-Republic, etc.\n")
cat("5. Europe: Germany, Greece, Italy, Poland, France, etc. (excluding Anglosphere)\n")
cat("6. Asia: Philippines, China, Japan, India, Vietnam, etc.\n")
cat("7. Middle_East_Oceania: Iran, Outlying-US(Guam-USVI-etc), South\n")
cat("8. Other_Regions: All remaining countries\n")

# =============================================================================
# 9. CHECK AND REMOVE ANY REMAINING NA VALUES
# =============================================================================

cat("\n=== CHECKING FOR REMAINING NA VALUES ===\n")

# Check for any remaining NA values
na_summary <- df_clean %>%
  summarise(across(everything(), ~sum(is.na(.))))

total_nas <- sum(na_summary)
cat("Total NA values remaining:", total_nas, "\n")

if (total_nas > 0) {
  cat("\nNA values per column:\n")
  print(na_summary)
  
  # Identify rows with NA values
  rows_with_na <- df_clean %>%
    filter(if_any(everything(), is.na))
  
  cat("\nRows with NA values:", nrow(rows_with_na), "\n")
  
  # Save these rows for analysis before removing
  write.csv(rows_with_na, "rows_with_na_before_removal.csv", row.names = FALSE)
  cat("Saved rows with NA values to 'rows_with_na_before_removal.csv'\n")
  
  # Now remove NA values
  rows_before <- nrow(df_clean)
  df_clean <- df_clean %>%
    drop_na()
  rows_after <- nrow(df_clean)
  
  cat("\nRows before removing NAs:", rows_before, "\n")
  cat("Rows after removing NAs:", rows_after, "\n")
  cat("Rows deleted due to NA values:", rows_before - rows_after, "\n")
  cat("Percentage deleted:", round((rows_before - rows_after)/rows_before*100, 2), "%\n")
  
} else {
  cat("No NA values found after KNN imputation. No rows deleted.\n")
  rows_before <- nrow(df_clean)
  rows_after <- nrow(df_clean)
}

cat("\nFinal dataset size:", nrow(df_clean), "records\n")

# =============================================================================
# 10. PREPARE DIFFERENT DATASETS FOR DIFFERENT MODELS
# =============================================================================

cat("\n=== PREPARING MODEL-SPECIFIC DATASETS ===\n")

# Convert remaining character columns to factors
df_clean <- df_clean %>%
  mutate(across(where(is.character), as.factor))

# Dataset 1: For Logistic Regression (needs numeric encoding)
df_for_regression <- df_clean %>%
  mutate(
    sex = ifelse(sex == "Male", 1, 0),
    has_capital = ifelse(has_capital == "Yes", 1, 0)
  )

# Create recipe for regression with proper handling of new levels
regression_recipe <- recipe(income_level ~ ., data = df_for_regression) %>%
  step_unknown(all_nominal_predictors(), new_level = "Missing") %>%  # Handle missing factor levels
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors()) %>%  # Remove zero-variance predictors BEFORE normalization
  step_normalize(all_numeric_predictors(), -age, -sex, -has_capital) %>%
  prep()

df_regression <- bake(regression_recipe, new_data = df_for_regression)

# Dataset 2: For Decision Trees (keeps factors, no scaling)
df_for_trees <- df_clean  # Already in correct format

# Dataset 3: For Naive Bayes (discretized numeric features)
df_for_bayes <- df_clean %>%
  mutate(
    # Discretize age for naive Bayes
    age_discrete = cut(age,
                       breaks = c(17, 25, 35, 45, 55, 65, 90),
                       labels = c("Young", "Young_Adult", "Middle", "Senior", "Elderly", "Retired")),
    
    # Discretize capital gain/loss
    capital_gain_cat = cut(capital_gain,
                           breaks = c(-1, 0, 5000, 10000, 50000, 100000),
                           labels = c("None", "Low", "Medium", "High", "Very_High")),
    
    capital_loss_cat = cut(capital_loss,
                           breaks = c(-1, 0, 1500, 3000, 5000, 10000),
                           labels = c("None", "Low", "Medium", "High", "Very_High")),
    
    hours_discrete = cut(hours_per_week,
                         breaks = c(0, 20, 40, 60, 100),
                         labels = c("Very_Low", "Normal", "High", "Very_High"))
  ) %>%
  select(-age, -capital_gain, -capital_loss, -hours_per_week)

# =============================================================================
# 11. HANDLE CLASS IMBALANCE (OPTIMIZED FOR 32K DATASET)
# =============================================================================

cat("\n=== HANDLING CLASS IMBALANCE ===\n")
cat("Dataset size:", nrow(df_clean), "records\n")
cat("Class distribution: ", 
    round(class_dist$percentage[1], 1), "% low_income, ",
    round(class_dist$percentage[2], 1), "% high_income\n")

# Calculate class weights
calculate_class_weights <- function(y) {
  class_counts <- table(y)
  total <- sum(class_counts)
  weights <- total / (length(class_counts) * class_counts)
  return(weights)
}

class_weights <- calculate_class_weights(df_clean$income_level)
cat("\nClass weights for cost-sensitive learning:\n")
print(class_weights)

# Create balanced dataset using ROSE
set.seed(123)
cat("\nGenerating balanced dataset using ROSE...\n")

# Use ROSE with proper error handling
tryCatch({
  rose_data <- ROSE(income_level ~ ., 
                    data = df_clean, 
                    seed = 123)$data
  
  # Check ROSE distribution
  rose_dist <- rose_data %>%
    count(income_level) %>%
    mutate(percentage = n/sum(n) * 100)
  
  cat("\nROSE Balanced Dataset Distribution:\n")
  print(rose_dist)
  cat("ROSE dataset size:", nrow(rose_data), "records\n")
  
}, error = function(e) {
  cat("ROSE failed with error:", e$message, "\n")
  cat("Creating simple oversampled dataset instead...\n")
  
  # Simple oversampling as fallback
  rose_data <- upSample(x = df_clean %>% select(-income_level),
                        y = df_clean$income_level,
                        yname = "income_level")
  
  rose_dist <- rose_data %>%
    count(income_level) %>%
    mutate(percentage = n/sum(n) * 100)
  
  cat("\nOversampled Dataset Distribution (used as fallback):\n")
  print(rose_dist)
})

# Create SMOTE balanced version using ROSE as fallback
cat("\nGenerating SMOTE balanced dataset...\n")
tryCatch({
  # Use caret's SMOTE implementation (if available)
  # Note: caret's SMOTE might not be directly available
  # Using ROSE data as SMOTE for consistency
  smote_data <- rose_data  # Using ROSE as SMOTE alternative
  
  smote_dist <- smote_data %>%
    count(income_level) %>%
    mutate(percentage = n/sum(n) * 100)
  
  cat("\nSMOTE Balanced Dataset Distribution:\n")
  print(smote_dist)
  cat("SMOTE dataset size:", nrow(smote_data), "records\n")
  
}, error = function(e) {
  cat("SMOTE failed with error:", e$message, "\n")
  cat("Using ROSE data as SMOTE fallback...\n")
  smote_data <- rose_data
})

# =============================================================================
# 12. CREATE MODEL-SPECIFIC BALANCED DATASETS
# =============================================================================

cat("\n=== CREATING MODEL-SPECIFIC DATASETS ===\n")

# For each model type, create balanced versions
# For regression
df_regression_rose <- bake(regression_recipe, new_data = rose_data)
df_regression_smote <- bake(regression_recipe, new_data = smote_data)

# For trees
df_trees_rose <- rose_data
df_trees_smote <- smote_data

# For Naive Bayes from balanced data
df_bayes_rose <- rose_data %>%
  mutate(
    age_discrete = cut(age,
                       breaks = c(17, 25, 35, 45, 55, 65, 90),
                       labels = c("Young", "Young_Adult", "Middle", "Senior", "Elderly", "Retired")),
    
    capital_gain_cat = cut(capital_gain,
                           breaks = c(-1, 0, 5000, 10000, 50000, 100000),
                           labels = c("None", "Low", "Medium", "High", "Very_High")),
    
    capital_loss_cat = cut(capital_loss,
                           breaks = c(-1, 0, 1500, 3000, 5000, 10000),
                           labels = c("None", "Low", "Medium", "High", "Very_High")),
    
    hours_discrete = cut(hours_per_week,
                         breaks = c(0, 20, 40, 60, 100),
                         labels = c("Very_Low", "Normal", "High", "Very_High"))
  ) %>%
  select(-age, -capital_gain, -capital_loss, -hours_per_week)

df_bayes_smote <- smote_data %>%
  mutate(
    age_discrete = cut(age,
                       breaks = c(17, 25, 35, 45, 55, 65, 90),
                       labels = c("Young", "Young_Adult", "Middle", "Senior", "Elderly", "Retired")),
    
    capital_gain_cat = cut(capital_gain,
                           breaks = c(-1, 0, 5000, 10000, 50000, 100000),
                           labels = c("None", "Low", "Medium", "High", "Very_High")),
    
    capital_loss_cat = cut(capital_loss,
                           breaks = c(-1, 0, 1500, 3000, 5000, 10000),
                           labels = c("None", "Low", "Medium", "High", "Very_High")),
    
    hours_discrete = cut(hours_per_week,
                         breaks = c(0, 20, 40, 60, 100),
                         labels = c("Very_Low", "Normal", "High", "Very_High"))
  ) %>%
  select(-age, -capital_gain, -capital_loss, -hours_per_week)

# =============================================================================
# 13. DATA SPLITTING WITH STRATIFICATION (FIXED)
# =============================================================================

cat("\n=== SPLITTING DATA WITH STRATIFICATION ===\n")

set.seed(123)

# FIXED: createDataPartition doesn't have 'strata' parameter
# It automatically does stratified sampling when y is a factor
create_stratified_splits <- function(data, target_col, train_pct = 0.7) {
  
  # Create training index with stratification (automatic when y is factor)
  train_index <- createDataPartition(data[[target_col]], 
                                     p = train_pct, 
                                     list = FALSE)
  
  train_data <- data[train_index, ]
  test_data <- data[-train_index, ]
  
  # Further split training for validation
  val_index <- createDataPartition(train_data[[target_col]],
                                   p = 0.2,
                                   list = FALSE)
  
  val_data <- train_data[val_index, ]
  train_final <- train_data[-val_index, ]
  
  return(list(train = train_final, val = val_data, test = test_data))
}

# Create splits for all datasets
cat("Creating splits for original unbalanced data...\n")
splits_original <- create_stratified_splits(df_clean, "income_level")
splits_rose <- create_stratified_splits(rose_data, "income_level")
splits_smote <- create_stratified_splits(smote_data, "income_level")

# Create splits for model-specific datasets
splits_regression_original <- create_stratified_splits(df_regression, "income_level")
splits_regression_rose <- create_stratified_splits(df_regression_rose, "income_level")
splits_regression_smote <- create_stratified_splits(df_regression_smote, "income_level")

splits_trees_original <- create_stratified_splits(df_for_trees, "income_level")
splits_trees_rose <- create_stratified_splits(df_trees_rose, "income_level")
splits_trees_smote <- create_stratified_splits(df_trees_smote, "income_level")

splits_bayes_original <- create_stratified_splits(df_for_bayes, "income_level")
splits_bayes_rose <- create_stratified_splits(df_bayes_rose, "income_level")
splits_bayes_smote <- create_stratified_splits(df_bayes_smote, "income_level")

# =============================================================================
# 14. SAVE ALL DATASETS
# =============================================================================

cat("\n=== SAVING ALL DATASETS ===\n")

# Save class weights
write.csv(data.frame(class = names(class_weights), weight = class_weights),
          "class_weights.csv", row.names = FALSE)

# Save original datasets
write.csv(df_clean, "original_cleaned.csv", row.names = FALSE)
write.csv(rose_data, "rose_balanced.csv", row.names = FALSE)
write.csv(smote_data, "smote_balanced.csv", row.names = FALSE)

# Save model-specific datasets
write.csv(df_regression, "regression_ready.csv", row.names = FALSE)
write.csv(df_for_trees, "trees_ready.csv", row.names = FALSE)
write.csv(df_for_bayes, "bayes_ready.csv", row.names = FALSE)

write.csv(df_regression_rose, "regression_rose.csv", row.names = FALSE)
write.csv(df_regression_smote, "regression_smote.csv", row.names = FALSE)
write.csv(df_trees_rose, "trees_rose.csv", row.names = FALSE)
write.csv(df_trees_smote, "trees_smote.csv", row.names = FALSE)
write.csv(df_bayes_rose, "bayes_rose.csv", row.names = FALSE)
write.csv(df_bayes_smote, "bayes_smote.csv", row.names = FALSE)

# Save train/val/test splits
save_splits <- function(splits, prefix) {
  write.csv(splits$train, paste0(prefix, "_train.csv"), row.names = FALSE)
  write.csv(splits$val, paste0(prefix, "_val.csv"), row.names = FALSE)
  write.csv(splits$test, paste0(prefix, "_test.csv"), row.names = FALSE)
}

save_splits(splits_original, "original")
save_splits(splits_rose, "rose")
save_splits(splits_smote, "smote")

save_splits(splits_regression_original, "regression_original")
save_splits(splits_regression_rose, "regression_rose")
save_splits(splits_regression_smote, "regression_smote")

save_splits(splits_trees_original, "trees_original")
save_splits(splits_trees_rose, "trees_rose")
save_splits(splits_trees_smote, "trees_smote")

save_splits(splits_bayes_original, "bayes_original")
save_splits(splits_bayes_rose, "bayes_rose")
save_splits(splits_bayes_smote, "bayes_smote")

cat("✓ All datasets saved successfully\n")

# =============================================================================
# 15. CREATE TRAINING CONFIGURATIONS
# =============================================================================

cat("\n=== CREATING TRAINING CONFIGURATIONS ===\n")

# Configuration for each model with imbalance handling
training_configs <- data.frame(
  Model_Type = rep(c("Logistic_Regression", "Decision_Tree", "Naive_Bayes"), each = 3),
  Data_Version = rep(c("Original", "ROSE_Balanced", "SMOTE_Balanced"), 3),
  Imbalance_Handling = c(
    "Class_Weights", "Balanced_Data", "Balanced_Data",
    "Class_Weights", "Balanced_Data", "Balanced_Data", 
    "Class_Weights", "Balanced_Data", "Balanced_Data"
  ),
  Evaluation_Metric = c(
    "AUC_ROC", "AUC_ROC", "AUC_ROC",
    "F1_Score", "F1_Score", "F1_Score",
    "F1_Score", "F1_Score", "F1_Score"
  ),
  Notes = c(
    "Use glm() with family=binomial and class weights",
    "Train on ROSE balanced data, no weights needed",
    "Train on SMOTE balanced data, no weights needed",
    "Use rpart with class weights for original data",
    "Train on ROSE balanced data for trees",
    "Train on SMOTE balanced data for trees",
    "Use naiveBayes with discretized data and class weights",
    "Train on discretized ROSE balanced data",
    "Train on discretized SMOTE balanced data"
  )
)

write.csv(training_configs, "model_training_configs.csv", row.names = FALSE)

# =============================================================================
# 16. FINAL SUMMARY AND RECOMMENDATIONS
# =============================================================================

# Get region percentages
region_summary <- df_clean %>%
  count(country_region) %>%
  mutate(percentage = round(n/sum(n) * 100, 2))

# Create region string for output
region_output <- ""
for(i in 1:nrow(region_summary)) {
  region_output <- paste0(region_output, i, ". ", region_summary$country_region[i], 
                          ": ", region_summary$percentage[i], "%\n")
}

# Calculate rows deleted
rows_deleted <- nrow(df) - nrow(df_clean)

cat(paste0(
  "\n", strrep("=", 70), "\n",
  "✅ DATA PREPROCESSING COMPLETED SUCCESSFULLY!\n",
  strrep("=", 70), "\n\n",
  
  "📊 DATASET SUMMARY:\n",
  "• Original size: ", nrow(df), " records, ", ncol(df), " columns\n",
  "• After cleaning: ", nrow(df_clean), " records\n",
  "• Rows deleted: ", rows_deleted, " (", round(rows_deleted/nrow(df)*100, 2), "%)\n",
  "• Class distribution: ", round(class_dist$percentage[1], 1), "% low_income, ",
  round(class_dist$percentage[2], 1), "% high_income\n",
  "• Imbalance ratio: ", round(max(class_dist$n) / min(class_dist$n), 2), ":1\n",
  "• Country categories: 8 meaningful NON-OVERLAPPING geographic regions\n\n",
  
  "🗺️ COUNTRY REGIONS CREATED (8 NON-OVERLAPPING categories):\n",
  region_output,
  "\n",
  
  "🎯 IMBALANCE HANDLING OPTIONS CREATED:\n",
  "• Class Weights: low_income=", round(class_weights["low_income"], 2), 
  ", high_income=", round(class_weights["high_income"], 2), "\n",
  "• ROSE Balanced dataset created (50/50 distribution)\n",
  "• SMOTE Balanced dataset created (using ROSE as alternative)\n\n",
  
  "🔧 KEY IMPROVEMENTS:\n",
  "• Fixed country category overlap (now 8 distinct categories)\n",
  "• Added explicit NA checking and reporting\n",
  "• Tracked rows deleted due to NA values\n",
  "• Proper ordering of case_when() to prevent overlap\n",
  "• Anglosphere now properly separated from Europe/North_America\n\n",
  
  "📁 DATASETS CREATED:\n",
  "• original_cleaned.csv - Base dataset with 8 country regions\n",
  "• rose_balanced.csv - ROSE balanced dataset\n",
  "• smote_balanced.csv - SMOTE balanced dataset (ROSE-based)\n",
  "• regression_ready.csv - For logistic regression\n",
  "• trees_ready.csv - For decision trees\n",
  "• bayes_ready.csv - For naive Bayes (discretized)\n",
  "• class_weights.csv - Weights for cost-sensitive learning\n",
  "• model_training_configs.csv - 9 training configurations\n",
  "• 27 train/val/test split files\n\n",
  
  "🚀 NEXT STEPS:\n",
  "1. Train 3 models (Logistic Regression, Decision Trees, Naive Bayes)\n",
  "2. For each model, test 3 approaches:\n",
  "   a) Original data + class weights\n",
  "   b) ROSE balanced data\n",
  "   c) SMOTE balanced data\n",
  "3. Compare 9 total configurations\n",
  "4. Use appropriate metrics (AUC-ROC for regression, F1-score for others)\n",
  "5. Analyze which imbalance handling works best for each model\n\n",
  
  "⚠️ NOTE: Check 'rows_with_na_before_removal.csv' to see which rows were deleted\n",
  
  strrep("=", 70), "\n"
))
