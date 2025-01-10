
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    results_dir = Path("results/")
    out_dir = results_dir / "features"
    out_dir.mkdir(exist_ok=True)
    results_dirs = results_dir.glob("AGE*")

    cols = ["feature_names", "feature_names_short", "num_features", "comment"]
    dfs = []
    data = []
    for indiv_dir in results_dirs:
        df = pd.read_csv(indiv_dir / "study_results.csv")
        df = df.loc[:, cols]
        dfs.append(df)

        feature_list = [eval(t) if t is not np.nan else [] for t in df.loc[:, "feature_names"].values]
        data.append(feature_list)

    features = {
        'Amniotic fluid': 0,
        'Apparent size': 1,
        'Appearance of brain': 2,
        'Apparent difficulty in obtaining plane': 3,
        'Ossification of skull': 4,
        'Position within pelvis/uterus': 5,
        'Relationship between the size of the fetus and sector width': 6,
        'Shadow': 7,
        'Shape of skull': 8,
    }

    oh_encodings = np.array([np.array(one_hot_encode(data_subset, features)) for data_subset in data])

    df = pd.DataFrame(oh_encodings.reshape(oh_encodings.shape[0] * oh_encodings.shape[1], oh_encodings.shape[2]), columns=list(features.keys()))
    num_each_feature = df.sum()
    participant_ids = [[i]*65 for i in range(10)]
    participant_ids = np.array(participant_ids).reshape(-1, 1)[:, 0]

    feature_names = [
        'Amniotic fluid',
        'Apparent size',
        'Appearance of brain',
        'Apparent difficulty in obtaining plane',
        'Ossification of skull',
        'Position within pelvis/uterus',
        'Relationship between the size of \n the fetus and sector width',
        'Shadow',
        'Shape of skull'
    ]

    plt.figure(figsize=(7, 4))
    plt.barh(feature_names, num_each_feature)
    #plt.xticks(rotation=45, ha='right')
    plt.xlabel("No. images")
    plt.tight_layout()
    plt.savefig(out_dir / "features_count.png", bbox_inches="tight")
    plt.show()

    sns.displot(df.sum(axis=1), bins=np.arange(-0.5, 8.5))
    plt.xlabel("No. features per image")
    plt.ylabel("No. images")
    plt.tight_layout()
    plt.savefig(out_dir / "features_number_per_image.pdf", bbox_inches="tight")
    plt.show()

    comments = pd.concat([sub_df.loc[:, "comment"] for sub_df in dfs])
    comments = comments.dropna()

    counts = df.sum(axis=1)
    counts = pd.DataFrame(counts)
    counts["participant_id"] = participant_ids
    sns.displot(counts, x=0, bins=np.arange(-0.5, 8.5), col="participant_id")
    plt.xlabel("No. features per image")
    plt.ylabel("No. images")
    plt.tight_layout()
    plt.show()

    df["participant_id"] = participant_ids
    participant_summary = df.groupby('participant_id').sum()

    participant_summary.T.plot(kind='bar', stacked=True, figsize=(10, 6))
    plt.title('Feature Labeling Frequency by Participant')
    plt.ylabel('Frequency')
    plt.xlabel('Features')
    plt.legend(title='Participant', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    diversity = (participant_summary > 0).sum(axis=1)

    plt.bar(diversity.index, diversity.values)
    plt.title('Labeling Diversity by Participant')
    plt.ylabel('Number of Features Labeled')
    plt.xlabel('Participant')
    plt.show()

    plt.figure(figsize=(12, 8))
    sns.heatmap(participant_summary, annot=False, cmap='viridis', cbar=True)
    plt.xlabel('Features')
    plt.ylabel('Participant')
    plt.show()

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot the heatmap, transpose the data to swap axes
    sns.heatmap(participant_summary.T, annot=False, cmap='viridis', cbar=True, ax=ax)

    # Set axis labels and title
    ax.set_title('Heatmap of Label Frequencies by Participant', fontsize=14)
    ax.set_xlabel('Participants', fontsize=12)
    ax.set_ylabel('Features', fontsize=12)

    plt.tight_layout()
    plt.show()

    print("Done!")


def one_hot_encode(data, features):
    """
    This function takes a list of lists (data) and returns a one-hot encoded representation.
    """
    encoded_data = []
    for sample in data:
        encoding = [0] * len(features)
        for feature in sample:
            encoding[features[feature]] = 1
        encoded_data.append(encoding)
    return encoded_data


if __name__ == "__main__":
    main()
