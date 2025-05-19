

# Applied\_Labor

This project investigates average wage trajectories across the life cycle, segmented by age, gender, and socio-professional category. We rely on **optimal transport methods**, which allow us to infer mobility patterns across cohorts without requiring individual tracking. Using the French *DADS Panel Tous Salariés* dataset, we study **intragenerational mobility** within the French labor market.

## Project Structure

* **`utils.py`**
  Contains all utility functions used for data cleaning, estimation of transport plans, and computation of average wage trajectories.

* **`estimators.py`**
  Includes functions to estimate wage trajectories using **Random Forests**, which serve as a benchmark to compare with the optimal transport results.

* **`explo_cads.ipynb`**
  Main notebook used to run the full analysis and generate the figures and results presented in the report.

## Key Features

* Non-parametric estimation of wage mobility without tracking individuals over time.
* Application of optimal transport to labor economics.
* Comparative analysis with machine learning methods (Random Forests).

## Dataset

* **DADS Panel Tous Salariés**
  Administrative panel data covering salaried workers in France, including wages, demographics, and job characteristics.

