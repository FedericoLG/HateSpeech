# PREPROCESSING HATEXPLAIN DATASET 

from collections import Counter
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
import numpy as np # type: ignore

# Entry Sample:
"""json_data = {
  "24198545_gab": {
    "post_id": "24198545_gab",
    "annotators": [
      {
        "label": "hatespeech",
        "annotator_id": 4,
        "target": ["African"]
      },
      {
        "label": "hatespeech",
        "annotator_id": 3,
        "target": ["African"]
      },
      {
        "label": "offensive",
        "annotator_id": 5,
        "target": ["African"]
      }
    ],
    "rationales":[
    [0,0,0,0,0,0,0,0,1,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    ],
    "post_tokens": ["and","this","is","why","i","end","up","with","nigger","trainee","doctors","who","can","not","speak","properly","lack","basic","knowledge","of","biology","it","truly","scary","if","the","public","only","knew"]
  }
}
"""



def json_to_dataframe(json_data):
    # Prepare lists to hold extracted data
    post_ids = []
    annotator_ids = []
    labels = []
    targets = []
    post_tokens = []

    # Loop through each entry in the JSON
    for post_id, post_info in json_data.items():
        for annotator in post_info['annotators']:
            post_ids.append(post_id)
            annotator_ids.append(annotator['annotator_id'])
            labels.append(annotator['label'])
            targets.append(", ".join(annotator['target']))
            post_tokens.append(" ".join(post_info['post_tokens']))
    
    # Create a DataFrame from the lists
    df = pd.DataFrame({
        'post_id': post_ids,
        'annotator_id': annotator_ids,
        'label': labels,
        'target': targets,
        'post_tokens': post_tokens
    })
    
    return df


# Here's the code to drop the annotator_id column and group rows by post_id based on the majority rule for the label column:

def process_dataframe(df):
    # Step 1: Drop the 'annotator_id' column
    df = df.drop(columns=['annotator_id'])

    # Step 2: Group by 'post_id' and apply the majority rule on 'label'
    def majority_label(labels):
        count = Counter(labels)
        majority = count.most_common(1)[0][0]  # Get the most common label
        return majority

    # Group by 'post_id', apply majority rule, and aggregate other columns
    grouped_df = df.groupby('post_id').agg({
        'label': majority_label,
        'target': lambda x: x.iloc[0],  # Take the first value of target (assuming all targets are the same)
        'post_tokens': lambda x: x.iloc[0]  # Take the first list of post_tokens
    }).reset_index()

    return grouped_df

###################################
# DIAGNOSTICS FOR DATASET BALANCE #
################################# #

"""
plot_label_distribution: This function will visualize the distribution of each label in your dataset, 
helping you understand how many samples are associated with each label.

plot_label_cooccurrence_matrix: This function will create a heatmap of the label co-occurrence matrix, 
allowing you to see which labels frequently appear together.

calculate_label_imbalance: This function will compute and print imbalance metrics, such as the Gini index and entropy, 
which give you a quantitative measure of how balanced your labels are.

detect_underrepresented_labels: This function identifies labels that are underrepresented in your dataset based on a threshold you specify, 
helping you identify potential issues with class imbalance.
"""

# 1. LABEL DISTRIBUTION FUNCTION

def plot_label_distribution(df, label_column):
    """
    Plots the distribution of each label in the dataset.

    Args:
    - df (pd.DataFrame): The DataFrame containing the dataset.
    - label_column (str): The name of the column containing the label lists.
    """
    # Initialize a counter for each label
    label_counts = pd.Series([0]*len(df[label_column].iloc[0]), index=range(len(df[label_column].iloc[0])))

    # Count occurrences of each label
    for labels in df[label_column]:
        label_counts += pd.Series(labels)

    # Plot the distribution
    plt.figure(figsize=(10, 6))
    label_counts.plot(kind='bar')
    plt.xlabel('Label Index')
    plt.ylabel('Number of Samples')
    plt.title('Label Distribution')
    plt.show()

# Example usage:
#plot_label_distribution(df, 'label')


# 2. Label Co-Occurrence Matrix


def plot_label_cooccurrence_matrix(df, label_column):
    """
    Plots a heatmap of the label co-occurrence matrix.

    Args:
    - df (pd.DataFrame): The DataFrame containing the dataset.
    - label_column (str): The name of the column containing the label lists.
    """
    num_labels = len(df[label_column].iloc[0])
    cooccurrence_matrix = np.zeros((num_labels, num_labels))

    for labels in df[label_column]:
        labels_array = np.array(labels)
        cooccurrence_matrix += np.outer(labels_array, labels_array)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cooccurrence_matrix, annot=True, fmt="d", cmap="Blues")
    plt.xlabel('Label Index')
    plt.ylabel('Label Index')
    plt.title('Label Co-Occurrence Matrix')
    plt.show()

# Example usage:
#plot_label_cooccurrence_matrix(df, 'label')


#  3. Label Imbalance Metrics



def calculate_label_imbalance(df, label_column):
    """
    Calculates imbalance metrics for the label distribution.

    Args:
    - df (pd.DataFrame): The DataFrame containing the dataset.
    - label_column (str): The name of the column containing the label lists.

    Returns:
    - imbalance_metrics (dict): Dictionary containing imbalance metrics like Gini index and entropy.
    """
    # Initialize a counter for each label
    label_counts = pd.Series([0]*len(df[label_column].iloc[0]), index=range(len(df[label_column].iloc[0])))

    # Count occurrences of each label
    for labels in df[label_column]:
        label_counts += pd.Series(labels)

    # Normalize the counts to get label frequencies
    label_frequencies = label_counts / label_counts.sum()

    # Calculate Gini index
    gini_index = 1 - np.sum(label_frequencies ** 2)

    # Calculate entropy
    entropy = -np.sum(label_frequencies * np.log2(label_frequencies + 1e-9))  # Adding a small epsilon to avoid log(0)

    imbalance_metrics = {
        "Gini Index": gini_index,
        "Entropy": entropy
    }

    return imbalance_metrics

# Example usage:
#imbalance_metrics = calculate_label_imbalance(df, 'label')
#print(imbalance_metrics)

# 4. Imbalance Detection Function

def detect_underrepresented_labels(df, label_column, threshold=0.05):
    """
    Detects labels that are underrepresented in the dataset.

    Args:
    - df (pd.DataFrame): The DataFrame containing the dataset.
    - label_column (str): The name of the column containing the label lists.
    - threshold (float): The threshold below which a label is considered underrepresented (as a fraction of the total samples).

    Returns:
    - underrepresented_labels (list): List of label indices that are underrepresented.
    """
    total_samples = len(df)
    label_counts = pd.Series([0]*len(df[label_column].iloc[0]), index=range(len(df[label_column].iloc[0])))

    for labels in df[label_column]:
        label_counts += pd.Series(labels)

    underrepresented_labels = label_counts[label_counts / total_samples < threshold].index.tolist()
    
    return underrepresented_labels

# Example usage:
#underrepresented_labels = detect_underrepresented_labels(df, 'label')
#print("Underrepresented Labels:", underrepresented_labels)


