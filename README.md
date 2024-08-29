
# SpaceX Falcon 9 Landing Prediction

## Project Overview

Example of a successful and launch : 

![1724936691783](image/README/1724936691783.png)

Several examples of an unsuccessful landing : 

![1724938046898](image/README/1724938046898.png)

In this project, we aim to predict the successful landing of the Falcon 9 first stage, a critical component of SpaceX's cost-saving strategy. SpaceX advertises Falcon 9 rocket launches at a significantly lower cost of $62 million, compared to other providers charging upwards of $165 million. This dramatic reduction is largely due to SpaceX's ability to reuse the first stage of their rockets. By accurately predicting whether the first stage will land successfully, we can estimate the overall cost of a launch more precisely.

## Why This Matters

Accurately predicting the landing outcome not only provides a clearer understanding of launch economics but also offers valuable insights for other aerospace companies. Competing companies could use this prediction model to assess and refine their own bidding strategies when vying for contracts against SpaceX. Furthermore, the model could help in optimizing recovery operations and potentially increase the efficiency of reusable launch systems, ultimately driving down costs and fostering innovation in the industry.

Beyond cost estimation, the insights from this prediction model could be used to develop a decision support system for launch providers. This system could simulate various launch scenarios, including weather conditions and rocket configurations, to optimize launch windows and enhance the probability of a successful first stage landing. Such a tool would be invaluable for new market entrants looking to compete with established players like SpaceX, providing them with a competitive edge in the increasingly crowded commercial spaceflight market.

## Project Structure

- `data/`: Contains raw and processed data.
- `notebooks/`: Jupyter notebooks for exploration and model training.
- `src/`: Source code for data processing and model development.
- `scripts/`: Standalone scripts for running tasks.
- `results/`: Results including models, logs, and figures.
- `tests/`: Unit tests for validating the code.

# Collecting the data

In this task, we will make a GET request to the SpaceX API using the following URL: `https://api.spacexdata.com/v4/launches/past` to retrieve the necessary raw data. We will then save this data into a CSV file for further analysis.

# Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```
