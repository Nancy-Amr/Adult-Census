library(dplyr)

# Cleaned dataset
adult <- read.csv("C:/Users/Asus/Downloads/original_cleaned.csv", stringsAsFactors = FALSE)


# BASIC DATA EXPLORATION

cat("=== DATASET OVERVIEW ===\n")
cat("Dimensions:", nrow(adult), "rows x", ncol(adult), "columns\n\n")

cat("Structure:\n")
str(adult)

cat("\nFirst few rows:\n")
print(head(adult))

cat("\nSummary Statistics:\n")
print(summary(adult))
#EDA
# TARGET VARIABLE ANALYSIS income_level

cat(" TARGET VARIABLE income_level\n")

# Class distribution
income_count <- table(adult$income_level)
income_prop <- prop.table(income_count) * 100

# Barplot
barplot(income_count,
        main = "Income Level Distribution\n(Class Balance Check)",
        xlab = "Income Level",
        ylab = "Count",
        col = c("red", "orange"),
        ylim = c(0, max(income_count) * 1.2))
text(x = barplot(income_count, plot = FALSE), 
     y = income_count + max(income_count) * 0.05, 
     labels = paste0(income_count, "\n(", round(income_prop, 1), "%)"), 
     pos = 3, cex = 1, font = 2)

# Pie chart
pie(income_count,
    main = "Income Level Distribution",
    col = c("red", "orange"),
    labels = paste0(names(income_count), "\n", 
                    income_count, "\n(", round(income_prop, 1), "%)"),
    cex = 1)

# CATEGORICAL FEATURES vs TARGET

# GENDER vs INCOME
cat("\n=== GENDER vs INCOME ===\n")
gender_income <- table(adult$sex, adult$income_level)
print(gender_income)
cat("\nProportions within each gender:\n")
print(round(prop.table(gender_income, margin = 1) * 100, 2))

# Count plot - side by side bars
barplot(gender_income,
        beside = TRUE,
        main = "Income Distribution by Gender",
        xlab = "Income Level",
        ylab = "Count",
        col = c("lightpink", "lightblue"),
        legend.text = rownames(gender_income),
        args.legend = list(x = "topright", cex = 0.9))

# Proportion plot - properly formatted
prop_gender <- prop.table(gender_income, margin = 1) * 100
barplot(prop_gender,
        beside = TRUE,
        main = "Income Proportions by Gender\n(% within each gender)",
        xlab = "Income Level",
        ylab = "Percentage (%)",
        col = c("lightpink", "lightblue"),
        legend.text = rownames(gender_income),
        args.legend = list(x = "topright", cex = 0.9),
        ylim = c(0, 100))

# EDUCATION vs INCOME
cat("\n=== EDUCATION vs INCOME ===\n")
edu_income <- table(adult$education_simple, adult$income_level)
print(edu_income)
cat("\nProportions within each education level:\n")
print(round(prop.table(edu_income, margin = 1) * 100, 2))

# Count plot
barplot(edu_income,
        beside = TRUE,
        main = "Income Distribution by Education Level",
        xlab = "Income Level",
        ylab = "Count",
        col = rainbow(nrow(edu_income)),
        legend.text = rownames(edu_income),
        args.legend = list(x = "topleft", cex = 0.7),
        las = 1)

# Proportion plot
prop_edu <- prop.table(edu_income, margin = 1) * 100
barplot(prop_edu,
        beside = TRUE,
        main = "Income Proportions by Education\n(% within each education level)",
        xlab = "Income Level",
        ylab = "Percentage (%)",
        col = rainbow(nrow(edu_income)),
        legend.text = rownames(edu_income),
        args.legend = list(x = "topleft", cex = 0.7),
        las = 1,
        ylim = c(0, 100))

# WORKCLASS vs INCOME
cat("\n=== WORKCLASS vs INCOME ===\n")
work_income <- table(adult$workclass, adult$income_level)
print(work_income)
cat("\nProportions within each workclass:\n")
print(round(prop.table(work_income, margin = 1) * 100, 2))

# Count plot
barplot(work_income,
        beside = TRUE,
        main = "Income Distribution by Workclass",
        xlab = "Income Level",
        ylab = "Count",
        col = rainbow(nrow(work_income)),
        legend.text = rownames(work_income),
        args.legend = list(x = "topright", cex = 0.6),
        las = 1)

# Proportion plot
prop_work <- prop.table(work_income, margin = 1) * 100
barplot(prop_work,
        beside = TRUE,
        main = "Income Proportions by Workclass\n(% within each workclass)",
        xlab = "Income Level",
        ylab = "Percentage (%)",
        col = rainbow(nrow(work_income)),
        legend.text = rownames(work_income),
        args.legend = list(x = "topright", cex = 0.6),
        las = 1,
        ylim = c(0, 100))

# MARITAL STATUS vs INCOME
cat("\n=== MARITAL STATUS vs INCOME ===\n")
marital_income <- table(adult$marital_status, adult$income_level)
print(marital_income)
cat("\nProportions within each marital status:\n")
print(round(prop.table(marital_income, margin = 1) * 100, 2))

# Count plot
barplot(marital_income,
        beside = TRUE,
        main = "Income Distribution by Marital Status",
        xlab = "Income Level",
        ylab = "Count",
        col = rainbow(nrow(marital_income)),
        legend.text = rownames(marital_income),
        args.legend = list(x = "topright", cex = 0.6),
        las = 1)

# Proportion plot
prop_marital <- prop.table(marital_income, margin = 1) * 100
barplot(prop_marital,
        beside = TRUE,
        main = "Income Proportions by Marital Status\n(% within each status)",
        xlab = "Income Level",
        ylab = "Percentage (%)",
        col = rainbow(nrow(marital_income)),
        legend.text = rownames(marital_income),
        args.legend = list(x = "topright", cex = 0.6),
        las = 1,
        ylim = c(0, 100))

# OCCUPATION vs INCOME
cat("\n=== OCCUPATION vs INCOME ===\n")
occ_income <- table(adult$occupation, adult$income_level)
print(occ_income)

# Count plot
barplot(occ_income,
        beside = TRUE,
        main = "Income Distribution by Occupation",
        xlab = "Income Level",
        ylab = "Count",
        col = rainbow(nrow(occ_income)),
        legend.text = rownames(occ_income),
        args.legend = list(x = "topright", cex = 0.5),
        las = 1)

# Proportion plot
prop_occ <- prop.table(occ_income, margin = 1) * 100
barplot(prop_occ,
        beside = TRUE,
        main = "Income Proportions by Occupation\n(% within each occupation)",
        xlab = "Income Level",
        ylab = "Percentage (%)",
        col = rainbow(nrow(occ_income)),
        legend.text = rownames(occ_income),
        args.legend = list(x = "topright", cex = 0.5),
        las = 1,
        ylim = c(0, 100))

# RELATIONSHIP vs INCOME
cat("\n=== RELATIONSHIP vs INCOME ===\n")
rel_income <- table(adult$relationship, adult$income_level)
print(rel_income)

# Count plot
barplot(rel_income,
        beside = TRUE,
        main = "Income Distribution by Relationship Status",
        xlab = "Income Level",
        ylab = "Count",
        col = rainbow(nrow(rel_income)),
        legend.text = rownames(rel_income),
        args.legend = list(x = "topright", cex = 0.7),
        las = 1)

# Proportion plot
prop_rel <- prop.table(rel_income, margin = 1) * 100
barplot(prop_rel,
        beside = TRUE,
        main = "Income Proportions by Relationship\n(% within each relationship)",
        xlab = "Income Level",
        ylab = "Percentage (%)",
        col = rainbow(nrow(rel_income)),
        legend.text = rownames(rel_income),
        args.legend = list(x = "topright", cex = 0.7),
        las = 1,
        ylim = c(0, 100))

# RACE vs INCOME
cat("\n=== RACE vs INCOME ===\n")
race_income <- table(adult$race, adult$income_level)
print(race_income)

# Count plot
barplot(race_income,
        beside = TRUE,
        main = "Income Distribution by Race",
        xlab = "Income Level",
        ylab = "Count",
        col = rainbow(nrow(race_income)),
        legend.text = rownames(race_income),
        args.legend = list(x = "topright", cex = 0.7),
        las = 1)

# Proportion plot
prop_race <- prop.table(race_income, margin = 1) * 100
barplot(prop_race,
        beside = TRUE,
        main = "Income Proportions by Race\n(% within each race)",
        xlab = "Income Level",
        ylab = "Percentage (%)",
        col = rainbow(nrow(race_income)),
        legend.text = rownames(race_income),
        args.legend = list(x = "topright", cex = 0.7),
        las = 1,
        ylim = c(0, 100))

# AGE GROUP vs INCOME
cat("\n=== AGE GROUP vs INCOME ===\n")
age_group_income <- table(adult$age_group, adult$income_level)
print(age_group_income)

# Count plot
barplot(age_group_income,
        beside = TRUE,
        main = "Income Distribution by Age Group",
        xlab = "Income Level",
        ylab = "Count",
        col = rainbow(nrow(age_group_income)),
        legend.text = rownames(age_group_income),
        args.legend = list(x = "topleft", cex = 0.7),
        las = 1)

# Proportion plot
prop_age <- prop.table(age_group_income, margin = 1) * 100
barplot(prop_age,
        beside = TRUE,
        main = "Income Proportions by Age Group\n(% within each age group)",
        xlab = "Income Level",
        ylab = "Percentage (%)",
        col = rainbow(nrow(age_group_income)),
        legend.text = rownames(age_group_income),
        args.legend = list(x = "topleft", cex = 0.7),
        las = 1,
        ylim = c(0, 100))

# HOURS CATEGORY vs INCOME
cat("\n=== HOURS CATEGORY vs INCOME ===\n")
hours_income <- table(adult$hours_category, adult$income_level)
print(hours_income)

# Count plot
barplot(hours_income,
        beside = TRUE,
        main = "Income Distribution by Hours Category",
        xlab = "Income Level",
        ylab = "Count",
        col = rainbow(nrow(hours_income)),
        legend.text = rownames(hours_income),
        args.legend = list(x = "topright", cex = 0.7),
        las = 1)

# Proportion plot
prop_hours <- prop.table(hours_income, margin = 1) * 100
barplot(prop_hours,
        beside = TRUE,
        main = "Income Proportions by Hours Category\n(% within each category)",
        xlab = "Income Level",
        ylab = "Percentage (%)",
        col = rainbow(nrow(hours_income)),
        legend.text = rownames(hours_income),
        args.legend = list(x = "topright", cex = 0.7),
        las = 1,
        ylim = c(0, 100))

# HAS CAPITAL vs INCOME
cat("\n=== HAS CAPITAL vs INCOME ===\n")
capital_income <- table(adult$has_capital, adult$income_level)
print(capital_income)

# Count plot
barplot(capital_income,
        beside = TRUE,
        main = "Income Distribution by Capital Ownership",
        xlab = "Income Level",
        ylab = "Count",
        col = c("coral", "skyblue"),
        legend.text = rownames(capital_income),
        args.legend = list(x = "topright", cex = 0.9))

# Proportion plot
prop_capital <- prop.table(capital_income, margin = 1) * 100
barplot(prop_capital,
        beside = TRUE,
        main = "Income Proportions by Capital Ownership\n(% within each group)",
        xlab = "Income Level",
        ylab = "Percentage (%)",
        col = c("coral", "skyblue"),
        legend.text = rownames(capital_income),
        args.legend = list(x = "topright", cex = 0.9),
        ylim = c(0, 100))

# COUNTRY REGION vs INCOME
cat("\n=== COUNTRY REGION vs INCOME ===\n")
region_income <- table(adult$country_region, adult$income_level)
print(region_income)

# Count plot
barplot(region_income,
        beside = TRUE,
        main = "Income Distribution by Country Region",
        xlab = "Income Level",
        ylab = "Count",
        col = rainbow(nrow(region_income)),
        legend.text = rownames(region_income),
        args.legend = list(x = "topright", cex = 0.7),
        las = 1)

# Proportion plot
prop_region <- prop.table(region_income, margin = 1) * 100
barplot(prop_region,
        beside = TRUE,
        main = "Income Proportions by Country Region\n(% within each region)",
        xlab = "Income Level",
        ylab = "Percentage (%)",
        col = rainbow(nrow(region_income)),
        legend.text = rownames(region_income),
        args.legend = list(x = "topright", cex = 0.7),
        las = 1,
        ylim = c(0, 100))

# NUMERICAL FEATURES vs TARGET

# AGE vs INCOME
cat("\n=== AGE vs INCOME ===\n")

# Summary statistics by income level
cat("Age summary by income:\n")
print(tapply(adult$age, adult$income_level, summary))

# Boxplot
boxplot(age ~ income_level,
        data = adult,
        main = "Age Distribution by Income Level\n",
        xlab = "Income Level",
        ylab = "Age",
        col = c("red", "orange"),
        notch = TRUE,
        outline = TRUE)

# Density plot
dens_low <- density(adult$age[adult$income_level == "low_income"])
dens_high <- density(adult$age[adult$income_level == "high_income"])

plot(dens_low,
     main = "Age Density by Income Level",
     xlab = "Age",
     ylab = "Density",
     col = "red",
     lwd = 3,
     ylim = range(c(dens_low$y, dens_high$y)))
lines(dens_high, col = "green4", lwd = 3)
legend("topright", 
       legend = c("Low Income", "High Income"),
       col = c("red", "green4"),
       lwd = 3)

# Side-by-side histograms
par(mfrow = c(1, 2))
hist(adult$age[adult$income_level == "low_income"],
     breaks = 30,
     main = "Age: Low Income",
     xlab = "Age",
     col = rgb(1, 0, 0, 0.5),
     xlim = range(adult$age))
hist(adult$age[adult$income_level == "high_income"],
     breaks = 30,
     main = "Age: High Income",
     xlab = "Age",
     col = rgb(0, 1, 0, 0.5),
     xlim = range(adult$age))
par(mfrow = c(1, 1))

# EDUCATION NUMBER vs INCOME
cat("\n=== EDUCATION YEARS vs INCOME ===\n")

cat("Education summary by income:\n")
print(tapply(adult$education_num, adult$income_level, summary))

boxplot(education_num ~ income_level,
        data = adult,
        main = "Education Years by Income Level\n",
        xlab = "Income Level",
        ylab = "Years of Education",
        col = c("red", "orange"),
        notch = TRUE)

# Density plot
dens_edu_low <- density(adult$education_num[adult$income_level == "low_income"])
dens_edu_high <- density(adult$education_num[adult$income_level == "high_income"])

plot(dens_edu_low,
     main = "Education Years Density by Income",
     xlab = "Years of Education",
     ylab = "Density",
     col = "red",
     lwd = 3,
     ylim = range(c(dens_edu_low$y, dens_edu_high$y)))
lines(dens_edu_high, col = "green4", lwd = 3)
legend("topright", 
       legend = c("Low Income", "High Income"),
       col = c("red", "green4"),
       lwd = 3)



# Density plot
dens_hrs_low <- density(adult$hours_per_week[adult$income_level == "low_income"])
dens_hrs_high <- density(adult$hours_per_week[adult$income_level == "high_income"])

plot(dens_hrs_low,
     main = "Hours Worked Density by Income",
     xlab = "Hours per Week",
     ylab = "Density",
     col = "red",
     lwd = 3,
     ylim = range(c(dens_hrs_low$y, dens_hrs_high$y)))
lines(dens_hrs_high, col = "green4", lwd = 3)
legend("topright", 
       legend = c("Low Income", "High Income"),
       col = c("red", "green4"),
       lwd = 3)

# DOTCHARTS 
# Average age by occupation
avg_age_occ <- tapply(adult$age, adult$occupation, mean)

# % high income by occupation
high_income_pct <- prop.table(
  table(adult$occupation, adult$income_level),
  margin = 1
)[, "high_income"] * 100

# Sort occupations by average age
ord <- order(avg_age_occ)

dotchart(
  x = as.numeric(avg_age_occ[ord]),
  labels = names(avg_age_occ)[ord],
  main = "Average Age by Occupation\n % High Income in that occupation",
  xlab = "Average Age",
  pch = 19,
  col = colorRampPalette(c("red", "yellow", "green"))(length(ord))[
    rank(high_income_pct[names(avg_age_occ)[ord]])
  ],
  cex = 1.1
)

# FEATURE DISTRIBUTIONS

# Categorical features
par(mfrow = c(2, 3))
barplot(sort(table(adult$workclass), decreasing = TRUE), 
        main = "Workclass", las = 2, cex.names = 0.6, col = "skyblue")
barplot(sort(table(adult$education), decreasing = TRUE), 
        main = "Education", las = 2, cex.names = 0.5, col = "orange")
barplot(sort(table(adult$marital_status), decreasing = TRUE), 
        main = "Marital Status", las = 2, cex.names = 0.5, col = "orange")
barplot(sort(table(adult$occupation), decreasing = TRUE), 
        main = "Occupation", las = 2, cex.names = 0.4, col = "pink")
barplot(table(adult$sex), main = "Gender", col = c("lightpink", "lightblue"))
barplot(table(adult$race), main = "Race", las = 2, cex.names = 0.6, col = "lightcoral")
par(mfrow = c(1, 1))

# Numerical features
par(mfrow = c(2, 3))
hist(adult$age, breaks = 30, main = "Age", xlab = "Age", 
     col = "lightblue", border = "white")
hist(adult$education_num, breaks = 16, main = "Education Years", 
     xlab = "Years", col = "orange", border = "white")
hist(adult$hours_per_week, breaks = 30, main = "Hours per Week", 
     xlab = "Hours", col = "orange", border = "white")
hist(adult$capital_loss[adult$capital_loss > 0], breaks = 30, 
     main = "Capital Loss (>0)", xlab = "Amount", col = "pink", border = "white")
plot(density(adult$age), main = "Age Density", xlab = "Age", 
     col = "darkblue", lwd = 2)
par(mfrow = c(1, 1))


# Stratified scatter plot matrix
set.seed(123)
stratified_sample <- adult %>%
  group_by(income_level) %>%
  sample_n(size = min(500, n()), replace = FALSE) %>%
  ungroup()

pairs(~ age + education_num + hours_per_week,
      data = stratified_sample,
      main = "Scatter Plot Matrix (Stratified n=1000)",
      pch = 19,
      col = ifelse(stratified_sample$income_level == "low_income", 
                   rgb(1, 0, 0, 0.3), rgb(0, 1, 0, 0.3)),
      cex = 0.6)


