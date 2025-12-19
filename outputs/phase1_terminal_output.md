output of preprocessing 

```tex
Total rows: 32561 
Unique rows: 32537 
Exact duplicate rows: 24 
Duplicate groups: 23 
Max repeats of a single row: 3 
Unique values in target:
[1] "<=50K" ">50K" 

Counts per target value:

<=50K  >50K 
24698  7839

Class percentages (%):

<=50K  >50K 
75.91 24.09

Train rows: 26029  | Test rows: 6508 
CV folds: 5 

=== Missingness (TRAIN) ===
# A tibble: 3 x 4
  column         na_count     n na_pct
  <chr>             <int> <int>  <dbl>
1 occupation         1471 26029   5.65
2 workclass          1467 26029   5.64
3 native_country      485 26029   1.86

====================================================
Column: workclass 
====================================================
# A tibble: 4 x 4
  income_level is_missing     n pct_within_class
  <fct>        <lgl>      <int>            <dbl>
1 low_income   FALSE      18440            93.3
2 low_income   TRUE        1318             6.67
3 high_income  FALSE       6122            97.6
4 high_income  TRUE         149             2.38

Contingency table (rows=target, cols=missing?):

              FALSE  TRUE
  low_income  18440  1318
  high_income  6122   149

Chi-square test:

        Pearson's Chi-squared test with Yates' continuity correction

data:  tab
X-squared = 164.28, df = 1, p-value < 2.2e-16


Cramer's V (effect size):
[1] 0.0796

====================================================
Column: occupation
====================================================
# A tibble: 4 x 4
  income_level is_missing     n pct_within_class
  <fct>        <lgl>      <int>            <dbl>
1 low_income   FALSE      18436            93.3
2 low_income   TRUE        1322             6.69
3 high_income  FALSE       6122            97.6
4 high_income  TRUE         149             2.38

Contingency table (rows=target, cols=missing?):

              FALSE  TRUE
  low_income  18436  1322
  high_income  6122   149

Chi-square test:

        Pearson's Chi-squared test with Yates' continuity correction

data:  tab
X-squared = 165.41, df = 1, p-value < 2.2e-16


Cramer's V (effect size):
[1] 0.0799

====================================================
Column: native_country 
====================================================
# A tibble: 4 x 4
  income_level is_missing     n pct_within_class
  <fct>        <lgl>      <int>            <dbl>
1 low_income   FALSE      19389            98.1
2 low_income   TRUE         369             1.87
3 high_income  FALSE       6155            98.2
4 high_income  TRUE         116             1.85

Contingency table (rows=target, cols=missing?):

              FALSE  TRUE
  low_income  19389   369
  high_income  6155   116

Chi-square test:

        Pearson's Chi-squared test with Yates' continuity correction

data:  tab
X-squared = 0.0013908, df = 1, p-value = 0.9703


Cramer's V (effect size):
[1] 6e-04
Numeric columns:
[1] "age"            "education_num"  "capital_gain"   "capital_loss"   "hours_per_week"
# A tibble: 5 x 16
  column             n missing    q1    q3   iqr lower upper outlier_n outlier_pct   p01   p05   p95   p99   min   max
  <chr>          <int>   <int> <dbl> <dbl> <dbl> <dbl> <dbl>     <int>       <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl>
1 hours_per_week 26029       0    40    45     5  32.5  52.5      7149      27.5       8    18    60    80     1    99
2 capital_gain   26029       0     0     0     0   0     0        2190       8.41      0     0  5013 15024     0 99999
3 capital_loss   26029       0     0     0     0   0     0        1199       4.61      0     0     0  1980     0  4356
4 education_num  26029       0     9    12     3   4.5  16.5       955       3.67      3     5    14    16     1    16
5 age            26029       0    28    48    20  -2    78         106       0.407    17    20    63    74    17    90
Categorical columns (excluding target):
[1] "workclass"      "education"      "marital_status" "occupation"     "relationship"   "race"           "sex"            "native_country"

=== Categorical summary (TRAIN) ===
# A tibble: 8 x 7
  column         n_levels na_count na_pct top_level          top_n top_pct
  <chr>             <int>    <int>  <dbl> <chr>              <int>   <dbl>
1 native_country       40      485   1.86 United-States      23306    91.2
2 education            16        0   0    HS-grad             8425    32.4
3 occupation           14     1471   5.65 Prof-specialty      3307    13.5
4 workclass             8     1467   5.64 Private            18144    73.9
5 marital_status        7        0   0    Married-civ-spouse 11983    46.0
6 relationship          6        0   0    Husband            10543    40.5
7 race                  5        0   0    White              22245    85.5
8 sex                   2        0   0    Male               17406    66.9

----------------------------------
Column: workclass 
----------------------------------
x
         Private Self-emp-not-inc        Local-gov             <NA>        State-gov     Self-emp-inc      Federal-gov      Without-pay
           18144             2014             1686             1467             1040              876              787               11
    Never-worked
               4

----------------------------------
Column: education
----------------------------------
x
     HS-grad Some-college    Bachelors      Masters    Assoc-voc         11th   Assoc-acdm         10th      7th-8th  Prof-school          9th
        8425         5813         4324         1381         1112          902          850          736          511          451          415
        12th    Doctorate      5th-6th      1st-4th    Preschool
         346          319          267          134           43 
Rare levels (<1% of non-NA): 2
Very rare levels (<0.5% of non-NA): 1

----------------------------------
Column: marital_status
----------------------------------
x
   Married-civ-spouse         Never-married              Divorced             Separated               Widowed Married-spouse-absent
                11983                  8507                  3564                   813                   809                   332
    Married-AF-spouse
                   21
Rare levels (<1% of non-NA): 1
Very rare levels (<0.5% of non-NA): 1

----------------------------------
Column: occupation
----------------------------------
x
   Prof-specialty   Exec-managerial      Craft-repair      Adm-clerical             Sales     Other-service Machine-op-inspct              <NA>
             3307              3274              3267              3043              2907              2595              1630              1471
 Transport-moving Handlers-cleaners   Farming-fishing      Tech-support   Protective-serv   Priv-house-serv      Armed-Forces
             1254              1099               800               734               527               116                 5

----------------------------------
Column: relationship
----------------------------------
x
       Husband  Not-in-family      Own-child      Unmarried           Wife Other-relative
         10543           6638           4023           2765           1279            781
Rare levels (<1% of non-NA): 0
Very rare levels (<0.5% of non-NA): 0

----------------------------------
Column: race
----------------------------------
x
             White              Black Asian-Pac-Islander Amer-Indian-Eskimo              Other
             22245               2466                846                243                229
Rare levels (<1% of non-NA): 2
Very rare levels (<0.5% of non-NA): 0

----------------------------------
Column: sex
----------------------------------
x
  Male Female
 17406   8623 
Rare levels (<1% of non-NA): 0
Very rare levels (<0.5% of non-NA): 0

----------------------------------
Column: native_country
----------------------------------
x
     United-States             Mexico               <NA>        Philippines            Germany             Canada        Puerto-Rico        El-Salvador  
             23306                517                485                160                104                 98                 94                 84  
             India               Cuba            England              South              China              Italy            Vietnam            Jamaica 
                80                 77                 70                 64                 62                 58                 57                 56  
         Guatemala              Japan Dominican-Republic           Columbia
                55                 54                 48                 47
... (21more levels not shown)

Saved bundle: outputs/preprocess_bundle.rds
```


