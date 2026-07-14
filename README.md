# Cognifyz Data Analysis Internship Submission

## Student Information
- **Name:** Sahil Kumar
- **Domain:** Data Analysis
- **Levels Completed:** Level 2 & Level 3

---

## Completed Tasks

### Level 2: Core Data Insights
* **Restaurant Ratings Analysis**: Explored rating distributions (overall vs. active) and identified rating frequency peaks and average voting behaviors.
* **Cuisine Combinations Analysis**: Analyzed common cuisine combinations (e.g., Chinese & North Indian) and calculated if specific pairings yield higher average ratings.
* **Geographic Analysis**: Plotted the geographic location of restaurants using coordinates and identified restaurant count hotspots by city.
* **Chains vs. Independent Outlets**: Compared count, average ratings, and average votes for chain restaurants vs. independent establishments.

### Level 3: Advanced Analytics & Correlations
* **Reviews Analysis**: Grouped and analyzed the distribution of categorical rating texts (sentiment proxies) like Excellent, Good, Average, etc.
* **Votes Correlation Analysis**: Computed Pearson correlation between rating and votes, identifying highest and lowest voted outlets.
* **Price Range vs. Services Analysis**: Investigated how service options (online delivery, table booking) vary across restaurant price ranges.

---

## Submission Structure
- `Dataset.csv`: The shared dataset containing raw restaurant records.
- `Level_2/`:
  - `level2_analysis.py`: Python script performing Level 2 tasks.
  - `level2_output.txt`: Console output log from executing `level2_analysis.py`.
  - `rating_distribution.png`: Histogram plotting the rating distribution.
  - `geographic_distribution.png`: Scatter plot mapping restaurant density by coordinates.
- `Level_3/`:
  - `level3_analysis.py`: Python script performing Level 3 tasks.
  - `level3_output.txt`: Console output log from executing `level3_analysis.py`.
  - `rating_vs_votes_regression.png`: Scatter plot showing rating vs votes regression.
  - `services_by_price_range.png`: Grouped bar chart comparing online delivery/table bookings across price tiers.

*Note: The dataset `Dataset.csv` is located at the root of the folder and is shared by the scripts in both subdirectories.*
