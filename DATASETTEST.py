from sparql_langgraph_pipeline import *
from ASKpipeline import *
import pandas as pd

# Read the data
data = pd.read_csv('Reverted_LC.csv')

# Extract questions from the dataframe
questions = data['SAE Question'].tolist()

max_attempts = 3  # Set your desired value
max_workers = 4   # Set your desired value

results = run_sparql_pipeline_batch(questions, max_attempts, max_workers)

# Add results back to the dataframe
data['sparql_query'] = [r.get('sparql_query', '') for r in results]
data['results'] = [r.get('results', {}) for r in results]
data['error'] = [r.get('error', '') for r in results]
data['attempt'] = [r.get('attempt', 0) for r in results]

# Save the results
data.to_json('Reverted_LC_processed.json', orient = 'records')

print("Processing complete!")