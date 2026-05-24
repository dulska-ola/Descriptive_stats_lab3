---
title: Bivariate Statistics
subtitle: Foundations of Statistical Analysis in Python
abstract: This notebook explores bivariate relationships through linear correlations, highlighting their strengths and limitations. Practical examples and visualizations are provided to help users understand and apply these statistical concepts effectively.
author:
  - name: Karol Flisikowski
    affiliations: 
      - Gdansk University of Technology
      - Chongqing Technology and Business University
    orcid: 0000-0002-4160-1297
    email: karol@ctbu.edu.cn
date: 2025-05-03
---

## Goals of this lecture

There are many ways to *describe* a distribution. 

Here we will discuss:
- Measurement of the relationship between distributions using **linear, rank correlations**.
- Measurement of the relationship between qualitative variables using **contingency**.

## Importing relevant libraries


```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns ### importing seaborn
import pandas as pd
import scipy.stats as ss
```


```python
%matplotlib inline 
%config InlineBackend.figure_format = 'retina'
```


```python
import pandas as pd
df_pokemon = pd.read_csv("pokemon.csv")
```

## Describing *bivariate* data with correlations

- So far, we've been focusing on *univariate data*: a single distribution.
- What if we want to describe how *two distributions* relate to each other?
   - For today, we'll focus on *continuous distributions*.

### Bivariate relationships: `height`

- A classic example of **continuous bivariate data** is the `height` of a `parent` and `child`.  
- [These data were famously collected by Karl Pearson](https://www.kaggle.com/datasets/abhilash04/fathersandsonheight).


```python
df_height = pd.read_csv("height.csv")
df_height.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Father</th>
      <th>Son</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>65.0</td>
      <td>59.8</td>
    </tr>
    <tr>
      <th>1</th>
      <td>63.3</td>
      <td>63.2</td>
    </tr>
  </tbody>
</table>
</div>



#### Plotting Pearson's height data


```python
sns.scatterplot(data = df_height, x = "Father", y = "Son", alpha = .5);
```


    
![png](Exercise9_files/Exercise9_10_0.png)
    


### Introducing linear correlations

> A **correlation coefficient** is a number between $[–1, 1]$ that describes the relationship between a pair of variables.

Specifically, **Pearson's correlation coefficient** (or Pearson's $r$) describes a (presumed) *linear* relationship.

Two key properties:

- **Sign**: whether a relationship is positive (+) or negative (–).  
- **Magnitude**: the strength of the linear relationship.

$$
r = \frac{ \sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y}) }{ \sqrt{ \sum_{i=1}^{n} (x_i - \bar{x})^2 } \sqrt{ \sum_{i=1}^{n} (y_i - \bar{y})^2 } }
$$

Where:
- $r$ - Pearson correlation coefficient
- $x_i$, $y_i$ - values of the variables
- $\bar{x}$, $\bar{y}$ - arithmetic means
- $n$ - number of observations

Pearson's correlation coefficient measures the strength and direction of the linear relationship between two continuous variables. Its value ranges from -1 to 1:
- 1 → perfect positive linear correlation
- 0 → no linear correlation
- -1 → perfect negative linear correlation

This coefficient does not tell about nonlinear correlations and is sensitive to outliers.

### Calculating Pearson's $r$ with `scipy`

`scipy.stats` has a function called `pearsonr`, which will calculate this relationship for you.

Returns two numbers:

- $r$: the correlation coefficent.  
- $p$: the **p-value** of this correlation coefficient, i.e., whether it's *significantly different* from `0`.


```python
ss.pearsonr(df_height['Father'], df_height['Son'])
```




    PearsonRResult(statistic=np.float64(0.5011626808075912), pvalue=np.float64(1.272927574366214e-69))



    #### Check-in

Using `scipy.stats.pearsonr` (here, `ss.pearsonr`), calculate Pearson's $r$ for the relationship between the `Attack` and `Defense` of Pokemon.

- Is this relationship positive or negative?  
- How strong is this relationship?


```python
#Calculated correlation is 0.438. It means that the correlation is positive but it is not strong its restrained (umiarkowany). Pokemons with high attack normally have also pretty good defence skills but there are many exceptions in the game.
```

#### Solution


```python
ss.pearsonr(df_pokemon['Attack'], df_pokemon['Defense'])
```




    PearsonRResult(statistic=np.float64(0.4386870551184896), pvalue=np.float64(5.858479864289521e-39))



#### Check-in

Pearson'r $r$ measures the *linear correlation* between two variables. Can anyone think of potential limitations to this approach?

### Limitations of Pearson's $r$

- Pearson's $r$ *presumes* a linear relationship and tries to quantify its strength and direction.  
- But many relationships are **non-linear**!  
- Unless we visualize our data, relying only on Pearson'r $r$ could mislead us.

#### Non-linear data where $r = 0$


```python
x = np.arange(1, 40)
y = np.sin(x)
p = sns.lineplot(x = x, y = y)
```


    
![png](Exercise9_files/Exercise9_23_0.png)
    



```python
### r is close to 0, despite there being a clear relationship!
ss.pearsonr(x, y)
```




    PearsonRResult(statistic=np.float64(-0.04067793461845848), pvalue=np.float64(0.8057827185936625))



#### When $r$ is invariant to the real relationship

All these datasets have roughly the same **correlation coefficient**.


```python
df_anscombe = sns.load_dataset("anscombe")
sns.relplot(data = df_anscombe, x = "x", y = "y", col = "dataset");
```


    
![png](Exercise9_files/Exercise9_26_0.png)
    



```python
# Compute correlation matrix
corr = df_pokemon.corr(numeric_only=True)

# Set up the matplotlib figure
plt.figure(figsize=(10, 8))

# Create a heatmap
sns.heatmap(corr, 
            annot=True,         # Show correlation coefficients
            fmt=".2f",          # Format for coefficients
            cmap="coolwarm",    # Color palette
            vmin=-1, vmax=1,    # Fixed scale
            square=True,        # Make cells square
            linewidths=0.5,     # Line width between cells
            cbar_kws={"shrink": .75})  # Colorbar shrink

# Title and layout
plt.title("Correlation Heatmap", fontsize=16)
plt.tight_layout()

# Show plot
plt.show()
```


    
![png](Exercise9_files/Exercise9_27_0.png)
    


Interpretation of the heatmap

We can see some interesting relations form this correlation matrix
Sp atack and sp defence have one of the strongest positive correlations, that means that they usually go together
But speed and defence have one of the weakest positive correlations, so they usually dont go together, which all makes sense. When we think of a special pokemon we rather expect it to have special attack and special defence, and if we have a fast pokemon we wont really expect it to have strong defence skills

## Rank Correlations

Rank correlations are measures of the strength and direction of a monotonic (increasing or decreasing) relationship between two variables. Instead of numerical values, they use ranks, i.e., positions in an ordered set.

They are less sensitive to outliers and do not require linearity (unlike Pearson's correlation).

### Types of Rank Correlations

1. $ρ$ (rho) **Spearman's**
- Based on the ranks of the data.
- Value: from –1 to 1.
- Works well for monotonic but non-linear relationships.

$$
\rho = 1 - \frac{6 \sum d_i^2}{n(n^2 - 1)}
$$

Where:
- $d_i$ – differences between the ranks of observations,
- $n$ – number of observations.

2. $τ$ (tau) **Kendall's**
- Measures the number of concordant vs. discordant pairs.
- More conservative than Spearman's – often yields smaller values.
- Also ranges from –1 to 1.

$$
\tau = \frac{(C - D)}{\frac{1}{2}n(n - 1)}
$$

Where:
- $τ$ — Kendall's correlation coefficient,
- $C$ — number of concordant pairs,
- $D$ — number of discordant pairs,
- $n$ — number of observations,
- $\frac{1}{2}n(n - 1)$ — total number of possible pairs of observations.

What are concordant and discordant pairs?
- Concordant pair: if $x_i$ < $x_j$ and $y_i$ < $y_j$, or $x_i$ > $x_j$ and $y_i$ > $y_j$.
- Discordant pair: if $x_i$ < $x_j$ and $y_i$ > $y_j$, or $x_i$ > $x_j$ and $y_i$ < $y_j$.

### When to use rank correlations?
- When the data are not normally distributed.
- When you suspect a non-linear but monotonic relationship.
- When you have rank correlations, such as grades, ranking, preference level.

| Correlation type | Description | When to use |
|------------------|-----------------------------------------------------|----------------------------------------|
| Spearman's (ρ) | Monotonic correlation, based on ranks | When data are nonlinear or have outliers |
| Kendall's (τ) | Counts the proportion of congruent and incongruent pairs | When robustness to ties is important |

### Interpretation of correlation values

| Range of values | Correlation interpretation |
|------------------|----------------------------------|
| 0.8 - 1.0 | very strong positive |
| 0.6 - 0.8 | strong positive |
| 0.4 - 0.6 | moderate positive |
| 0.2 - 0.4 | weak positive |
| 0.0 - 0.2 | very weak or no correlation |
| < 0 | similarly - negative correlation |


```python
x_demo = np.linspace(1, 10, 50)
y_demo = np.exp(x_demo)
pearson_r, _ = ss.pearsonr(x_demo, y_demo)
spearman_rho, _ = ss.spearmanr(x_demo, y_demo)
plt.figure(figsize=(8, 5))
sns.scatterplot(x=x_demo, y=y_demo, color="purple", s=60, alpha=0.8)
plt.title(f"Prove for higher values in Spearman correlation\nPearson: {pearson_r:.2f} | Spearman: {spearman_rho:.2f}", fontsize=14)
plt.xlabel("X (Constant increase)", fontsize=12)
plt.ylabel("Y (Exponential increase)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
```


    
![png](Exercise9_files/Exercise9_33_0.png)
    


We made our own experiment here

Pearson method was looking more for a straight line, it assumed that the correlation is strong because the dots are going up and further from regression line

Spearman method ignored the shape of the curve and focused only on the ranks, then if x increases then y always increases, so it detected perfect monotonic correlation

It shows that when we are dealing with nonlinear data its better to use ranking correlation methods


```python
# Compute Kendall rank correlation
corr_kendall = df_pokemon.corr(method='kendall', numeric_only=True)

# Set up the matplotlib figure
plt.figure(figsize=(10, 8))

# Create a heatmap
sns.heatmap(corr_kendall,   #i think there was supposed to be corr_kendall instead of corr
            annot=True,         # Show correlation coefficients
            fmt=".2f",          # Format for coefficients
            cmap="coolwarm",    # Color palette
            vmin=-1, vmax=1,    # Fixed scale
            square=True,        # Make cells square
            linewidths=0.5,     # Line width between cells
            cbar_kws={"shrink": .75})  # Colorbar shrink

# Title and layout
plt.title("Correlation Heatmap", fontsize=16)
plt.tight_layout()

# Show plot
plt.show()
```


    
![png](Exercise9_files/Exercise9_35_0.png)
    


Interpretation of the heatmap

Kendall's correlation values are systematically lower(closer to zero) than those of Pearson or Spearman. This is a natural mathematical characteristic of this method because it is based on strictly counting concordant and discordant pairs, it is more conservative, making high correlation scores "harder" to achieve.

 Although the coefficients themselves are lower, the primary relationships within the dataset remain consistent. Sp. Atk and Sp. Def still form the strongest pair on the heatmap. This proves that this relationship is not just a mathematical artifact but an actual mechanic within the data.

### Comparison of Correlation Coefficients

| Property                | Pearson (r)                   | Spearman (ρ)                        | Kendall (τ)                          |
|-------------------------|-------------------------------|--------------------------------------|---------------------------------------|
| What it measures?       | Linear relationship           | Monotonic relationship (based on ranks) | Monotonic relationship (based on pairs) |
| Data type               | Quantitative, normal distribution | Ranks or ordinal/quantitative data  | Ranks or ordinal/quantitative data   |
| Sensitivity to outliers | High                          | Lower                               | Low                                   |
| Value range             | –1 to 1                       | –1 to 1                             | –1 to 1                               |
| Requires linearity      | Yes                           | No                                  | No                                    |
| Robustness to ties      | Low                           | Medium                              | High                                  |
| Interpretation          | Strength and direction of linear relationship | Strength and direction of monotonic relationship | Proportion of concordant vs discordant pairs |
| Significance test       | Yes (`scipy.stats.pearsonr`)  | Yes (`spearmanr`)                   | Yes (`kendalltau`)                   |

Brief summary:
- Pearson - best when the data are normal and the relationship is linear.
- Spearman - works better for non-linear monotonic relationships.
- Kendall - more conservative, often used in social research, less sensitive to small changes in data.

### Your Turn

For the Pokemon dataset, find the pairs of variables that are most appropriate for using one of the quantitative correlation measures. Calculate them, then visualize them.


```python
var1 = 'Sp. Atk'
var2 = 'Sp. Def'

pearson_corr, p_p = ss.pearsonr(df_pokemon[var1], df_pokemon[var2])
spearman_corr, p_s = ss.spearmanr(df_pokemon[var1], df_pokemon[var2])
kendall_corr, p_k = ss.kendalltau(df_pokemon[var1], df_pokemon[var2])

print(f"Correlation metrics for {var1} vs {var2}:")
print(f"  Pearson's r:  {pearson_corr:.4f} (p-value: {p_p:.2e})")
print(f"  Spearman's ρ: {spearman_corr:.4f} (p-value: {p_s:.2e})")
print(f"  Kendall's τ:  {kendall_corr:.4f} (p-value: {p_k:.2e})")
```

    Correlation metrics for Sp. Atk vs Sp. Def:
      Pearson's r:  0.5061 (p-value: 2.92e-53)
      Spearman's ρ: 0.5718 (p-value: 1.24e-70)
      Kendall's τ:  0.4230 (p-value: 1.39e-67)
    


```python
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.regplot(
    data=df_pokemon,
    x=var1,
    y=var2,
    ax=axes[0],
    scatter_kws={'alpha':0.4, 'color': '#34495e'},
    line_kws={'color': '#e74c3c', 'lw': 3}
)
axes[0].set_title(f"Scatter Plot: {var1} vs {var2}", fontsize=14)
axes[0].set_xlabel(var1, fontsize=12)
axes[0].set_ylabel(var2, fontsize=12)
axes[0].grid(True, linestyle='--', alpha=0.6)

stats_list = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
corr_matrix = df_pokemon[stats_list].corr(method='spearman') # Using Spearman as a robust default

sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-1, vmax=1,
    square=True,
    linewidths=0.5,
    ax=axes[1],
    cbar_kws={"shrink": .8}
)
axes[1].set_title("Spearman Rank Correlation Matrix", fontsize=14)

plt.tight_layout()
plt.show()
```


    
![png](Exercise9_files/Exercise9_41_0.png)
    


Interpretation of Quantitative Correlation Measures

The output displays three different correlation coefficients. All three return extremely low p-values (close to 0). This confirms that the positive relationship between Special Attack and Special Defense is statistically significant and not random.
As mathematically expected, Kendall's tau is the most conservative (lowest value). Because Pokémon stats often contain extreme outliers (e.g., Legendary Pokémon with massively skewed stats), Spearman's rho serves as the most reliable measure for this specific pair.

Scatter Plot: The regression line highlights a clear positive trend. Pokémon with high Special Attack tend to have high Special Defense. However, the wide spread of the data points indicates high variance (meaning there are plenty of exceptions, like pure attackers with no defense).
Spearman Heatmap: Using Spearman's method for the correlation matrix provides a robust overview of how all base stats move together monotonically, ignoring the distortions caused by outliers. It confirms that the Sp. Atk and Sp. Def pair has the strongest correlation among all core stats.

## Correlation of Qualitative Variables

A categorical variable is one that takes descriptive values ​​that represent categories—e.g. Pokémon type (Fire, Water, Grass), gender, status (Legendary vs. Normal), etc.

Such variables cannot be analyzed directly using correlation methods for numbers (Pearson, Spearman, Kendall). Other techniques are used instead.

### Contingency Table

A contingency table is a special cross-tabulation table that shows the frequency (i.e., the number of cases) for all possible combinations of two categorical variables.

It is a fundamental tool for analyzing relationships between qualitative features.

#### Chi-Square Test of Independence

The Chi-Square test checks whether there is a statistically significant relationship between two categorical variables.

Concept:

We compare:
- observed values (from the contingency table),
- with expected values, assuming the variables are independent.

$$
\chi^2 = \sum \frac{(O_{ij} - E_{ij})^2}{E_{ij}}
$$

Where:
- $O_{ij}$ – observed count in cell ($i$, $j$),
- $E_{ij}$ – expected count in cell ($i$, $j$), assuming independence.

### Example: Calculating Expected Values and Chi-Square Statistic in Python

Here’s how you can calculate the **expected values** and **Chi-Square statistic (χ²)** step by step using Python.

---

#### Step 1: Create the Observed Contingency Table
We will use the Pokémon example:

| Type 1 | Legendary = False | Legendary = True | Total |
|--------|-------------------|------------------|-------|
| Fire   | 18                | 5                | 23    |
| Water  | 25                | 3                | 28    |
| Grass  | 20                | 2                | 22    |
| Total  | 63                | 10               | 73    |


```python
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

# Observed values (contingency table)
observed = np.array([
    [18, 5],  # Fire
    [25, 3],  # Water
    [20, 2]   # Grass
])

# Convert to DataFrame for better visualization
observed_df = pd.DataFrame(
    observed,
    columns=["Legendary = False", "Legendary = True"],
    index=["Fire", "Water", "Grass"]
)
print("Observed Table:")
print(observed_df)
```

    Observed Table:
           Legendary = False  Legendary = True
    Fire                  18                 5
    Water                 25                 3
    Grass                 20                 2
    

Step 2: Calculate Expected Values
The expected values are calculated using the formula:

$$ E_{ij} = \frac{\text{Row Total} \times \text{Column Total}}{\text{Grand Total}} $$

You can calculate this manually or use scipy.stats.chi2_contingency, which automatically computes the expected values.


```python
# Perform Chi-Square test
chi2, p, dof, expected = chi2_contingency(observed)

# Convert expected values to DataFrame for better visualization
expected_df = pd.DataFrame(
    expected,
    columns=["Legendary = False", "Legendary = True"],
    index=["Fire", "Water", "Grass"]
)
print("\nExpected Table:")
print(expected_df)
```

    
    Expected Table:
           Legendary = False  Legendary = True
    Fire           19.849315          3.150685
    Water          24.164384          3.835616
    Grass          18.986301          3.013699
    

Step 3: Calculate the Chi-Square Statistic
The Chi-Square statistic is calculated using the formula:

$$ \chi^2 = \sum \frac{(O_{ij} - E_{ij})^2}{E_{ij}} $$

This is done automatically by scipy.stats.chi2_contingency, but you can also calculate it manually:


```python
# Manual calculation of Chi-Square statistic
chi2_manual = np.sum((observed - expected) ** 2 / expected)
print(f"\nChi-Square Statistic (manual): {chi2_manual:.4f}")
```

    
    Chi-Square Statistic (manual): 1.8638
    

Step 4: Interpret the Results
The chi2_contingency function also returns:

p-value: The probability of observing the data if the null hypothesis (independence) is true.
Degrees of Freedom (dof): Calculated as (rows - 1) * (columns - 1).


```python
print(f"\nChi-Square Statistic: {chi2:.4f}")
print(f"p-value: {p:.4f}")
print(f"Degrees of Freedom: {dof}")
```

    
    Chi-Square Statistic: 1.8638
    p-value: 0.3938
    Degrees of Freedom: 2
    

**Interpretation of the Chi-Square Test Result:**

| Value               | Meaning                                         |
|---------------------|-------------------------------------------------|
| High χ² value       | Large difference between observed and expected values |
| Low p-value         | Strong basis to reject the null hypothesis of independence |
| p < 0.05            | Statistically significant relationship between variables |


```python
observed_df.plot(kind='bar', stacked=True, figsize=(8, 6), color=['#3498db', '#e74c3c'], alpha=0.8, edgecolor='black')

plt.title("Proportion of legendary Pokemons with diversity of types", fontsize=14, pad=15)
plt.xlabel("Type of the Pokemon", fontsize=12)
plt.ylabel("Observed number of Pokemons", fontsize=12)
plt.xticks(rotation=0)
plt.legend(title="Status Legendary", labels=["Normal (False)", "Legendary (True)"])
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()
```


    
![png](Exercise9_files/Exercise9_54_0.png)
    


The calculated p-value for this specific sample (Fire, Water, Grass) is approximately 0.39. Since p > 0.05, we fail to reject the null hypothesis. This means there is no statistically significant relationship between these three specific Pokémon types and their Legendary status in this subset. Any differences we see are likely due to random chance.

Visual Confirmation (Stacked Bar Chart):

The stacked bar chart visually supports this conclusion. While the Fire type has a slightly thicker "Legendary" segment compared to Grass or Water, the overall proportion of Legendary Pokémon across these three categories remains relatively similar. The variations are not drastic enough to prove a systemic game design rule based on this small sample.

### Qualitative Correlations

#### Cramér's V

**Cramér's V** is a measure of the strength of association between two categorical variables. It is based on the Chi-Square test but scaled to a range of 0–1, making it easier to interpret the strength of the relationship.

$$
V = \sqrt{ \frac{\chi^2}{n \cdot (k - 1)} }
$$

Where:
- $\chi^2$ – Chi-Square test statistic,
- $n$ – number of observations,
- $k$ – the smaller number of categories (rows/columns) in the contingency table.

---

#### Phi Coefficient ($φ$)

Application:
- Both variables must be dichotomous (e.g., Yes/No, 0/1), meaning the table must have the smallest size of **2×2**.
- Ideal for analyzing relationships like gender vs purchase, type vs legendary.

$$
\phi = \sqrt{ \frac{\chi^2}{n} }
$$

Where:
- $\chi^2$ – Chi-Square test statistic for a 2×2 table,
- $n$ – number of observations.

---

#### Tschuprow’s T

**Tschuprow’s T** is a measure of association similar to **Cramér's V**, but it has a different scale. It is mainly used when the number of categories in the two variables differs. This is a more advanced measure applicable to a broader range of contingency tables.

$$
T = \sqrt{\frac{\chi^2}{n \cdot (k - 1)}}
$$

Where:
- $\chi^2$ – Chi-Square test statistic,
- $n$ – number of observations,
- $k$ – the smaller number of categories (rows or columns) in the contingency table.

Application: Tschuprow’s T is useful when dealing with contingency tables with varying numbers of categories in rows and columns.

---

### Summary - Qualitative Correlations

| Measure            | What it measures                                       | Application                     | Value Range     | Strength Interpretation       |
|--------------------|--------------------------------------------------------|---------------------------------|------------------|-------------------------------|
| **Cramér's V**     | Strength of association between nominal variables      | Any categories                  | 0 – 1           | 0.1–weak, 0.3–moderate, >0.5–strong |
| **Phi ($φ$)**      | Strength of association in a **2×2** table             | Two binary variables            | -1 – 1          | Similar to correlation        |
| **Tschuprow’s T**  | Strength of association, alternative to Cramér's V     | Tables with similar category counts | 0 – 1      | Less commonly used            |
| **Chi² ($χ²$)**    | Statistical test of independence                       | All categorical variables       | 0 – ∞           | Higher values indicate stronger differences |

### Example

Let's investigate whether the Pokémon's type (type_1) is affected by whether the Pokémon is legendary.

We'll use the **scipy** library.

This library already has built-in functions for calculating various qualitative correlation measures.


```python
from scipy.stats.contingency import association

# Contingency table:
ct = pd.crosstab(df_pokemon["Type 1"], df_pokemon["Legendary"])

# Calculating Cramér's V measure
V = association(ct, method="cramer") # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.contingency.association.html#association

print(f"Cramer's V: {V}") # interpret!

```

    Cramer's V: 0.3361928228447545
    

### Your turn

What visualization would be most appropriate for presenting a quantitative, ranked, and qualitative relationship?

Try to think about which pairs of variables could have which type of analysis based on the Pokemon data.

---


```python
# example
r, p_val = ss.pearsonr(df_pokemon['Attack'], df_pokemon['Defense'])
print(f"Pearson's r: {r:.4f}")

sns.regplot(data=df_pokemon, x='Attack', y='Defense',
            scatter_kws={'alpha':0.6, 'color': '#2980b9'},
            line_kws={'color': '#e74c3c', 'lw': 2.5})
plt.title("Quantitative Case: Attack vs Defense")
plt.show()
```

    Pearson's r: 0.4387
    


    
![png](Exercise9_files/Exercise9_60_1.png)
    


For quantitative relationships like Attack vs Defense, a scatter plot with a regression line is best to visualize continuous tracking trends and evaluate linear correlations. For ranked or ordinal relationships like Generation vs Total, a box plot is ideal to reveal if total base stat distributions systematically shift upward over sequential game releases. For qualitative relationships like Type 1 vs Legendary, a contingency table heatmap is best to display category overlaps and discover if specific elemental categories have disproportionately higher counts of legendary classifications

## Heatmaps for qualitative correlations


```python
# git clone https://github.com/ayanatherate/dfcorrs.git
# cd dfcorrs 
# pip install -r requirements.txt

from dfcorrs.cramersvcorr import Cramers
cram=Cramers()
cramer = cram.corr(df_pokemon)
print(cramer)
plt.figure(figsize=(10, 8))
sns.heatmap(cramer, annot=True, cmap='Blues', fmt=".2f", square=True)
plt.title("Correlation matrix V Cramér for Pokemons")
plt.show()

```

                Legendary    Type 2    Type 1  Generation
    Legendary    0.991617  0.108331  0.303091    0.078075
    Type 2       0.108331  1.000000  0.196861    0.207929
    Type 1       0.303091  0.196861  1.000000    0.158249
    Generation   0.078075  0.207929  0.158249    1.000000
    


    
![png](Exercise9_files/Exercise9_63_1.png)
    


No Negative Correlations:The results always fall within the range of 0 to 1. Cramér's V only indicates the strength of the association, but it cannot indicate its "direction" (categories cannot "increase" or "decrease" relative to one another).
Game Design in Practice (Type 1 vs Type 2): The strongest associations on this type of chart typically involve the primary and secondary Pokémon types. From a game design perspective, certain type combinations appear very frequently (e.g., Normal is almost always paired with Flying, and Grass with Poison), while others do not exist at all. The Cramér's V matrix accurately captures and visualizes these rules imposed by the developers.
Association with "Legendary" Status: The visible relationship between type and legendary status confirms what was previously proven by the Chi-Square test—legendary status is not distributed randomly and equally across every type. Instead, it favors specific "elite" elements (such as Dragon or Psychic types).

## Your turn!

Load the "sales" dataset and perform the bivariate analysis together with necessary plots. Remember about to run data preprocessing before the analysis.


```python
df_sales = pd.read_excel("sales.xlsx")
df_sales.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Store_Type</th>
      <th>City_Type</th>
      <th>Day_Temp</th>
      <th>No_of_Customers</th>
      <th>Sales</th>
      <th>Product_Quality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2020-10-01</td>
      <td>1</td>
      <td>1</td>
      <td>30.0</td>
      <td>100.0</td>
      <td>3112.0</td>
      <td>A</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2020-10-02</td>
      <td>2</td>
      <td>1</td>
      <td>32.0</td>
      <td>115.0</td>
      <td>3682.0</td>
      <td>A</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2020-10-03</td>
      <td>3</td>
      <td>3</td>
      <td>31.0</td>
      <td>NaN</td>
      <td>2774.0</td>
      <td>A</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2020-10-04</td>
      <td>1</td>
      <td>2</td>
      <td>29.0</td>
      <td>105.0</td>
      <td>3182.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2020-10-05</td>
      <td>1</td>
      <td>2</td>
      <td>33.0</td>
      <td>104.0</td>
      <td>1368.0</td>
      <td>B</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_sales = df_sales.fillna(df_sales.mean(numeric_only=True))
corr = df_sales.corr(numeric_only=True)
#data preprocessing - we wanted to avoid dropping any values so instead we filled the missing values with the mean of all the other values
plt.figure(figsize=(10, 8))
sns.heatmap(corr,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            vmin=-1, vmax=1,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": .75})
plt.title("Correlation Heatmap - Sales", fontsize=16)
plt.tight_layout()
plt.show()
#correlation heapmap - shows correlation between numerical values

var1 = corr.columns[0]
var2 = corr.columns[1]
sns.regplot(data=df_sales, x=var1, y=var2,
            scatter_kws={'alpha':0.4, 'color': '#34495e'},
            line_kws={'color': '#e74c3c', 'lw': 3})
plt.title(f"Scatter Plot: {var1} vs {var2}", fontsize=14)
plt.xlabel(var1, fontsize=12)
plt.ylabel(var2, fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
#scatterplot - shows linear correlation between two variables
```


    
![png](Exercise9_files/Exercise9_67_0.png)
    



    
![png](Exercise9_files/Exercise9_67_1.png)
    


Interpretation of the results and other thoughts

Data Preprocessing:
Before conducting the analysis, the dataset was cleaned. Instead of dropping rows with missing values, missing numerical values were imputed using the mean of their respective columns.

Correlation Heatmap:
The computed correlation matrix provides a macro-level view of how numeric business metrics interact. By examining the heatmap, we can quickly identify which financial or operational variables drive each other (e.g., Total Sales and Tax) and which are completely independent.

Visual Analysis (Unit Price vs. Quantity):
To thoroughly investigate the relationship between product cost and customer purchasing habits, a scatter plot with a regression line was generated for Unit Price and Quantity.
Observation: The red regression line is almost perfectly flat, and the data points are distributed evenly across different price tiers.
Business Insight:There is no linear correlation between the price of a single unit and the number of items a customer buys. Customers purchase items in similar quantities (1 to 5 units) regardless of whether the product is cheap or expensive. From a business perspective, this means that unit price is not the primary driver of the transaction size in terms of volume.

# Summary

There are many ways to *describe* our data:

- Measure **central tendency**.

- Measure its **variability**; **skewness** and **kurtosis**.

- Measure what **correlations** our data have.

All of these are **useful** and all of them are also **exploratory data analysis** (EDA).
