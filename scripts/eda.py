import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os


class EDA:

    ALLOWED_PRODUCTS = [
        "Credit card",
        "Personal loan",
        "Buy Now, Pay Later (BNPL)",
        "Savings account",
        "Money transfers",
    ]

    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None

    # ------------------------------------------------------------------ #
    #  1. Load                                                             #
    # ------------------------------------------------------------------ #
    def load_data(self):
        self.df = pd.read_csv(self.filepath, low_memory=False)
        print(f"Loaded dataset: {self.df.shape[0]:,} rows × {self.df.shape[1]} columns")
        print("\nColumns:", self.df.columns.tolist())
        print("\nSample rows:")
        print(self.df.head(3))

    # ------------------------------------------------------------------ #
    #  2. EDA                                                              #
    # ------------------------------------------------------------------ #
    def plot_product_distribution(self):
        counts = self.df["Product"].value_counts()
        plt.figure(figsize=(12, 6))
        sns.barplot(x=counts.values, y=counts.index, palette="viridis")
        plt.title("Distribution of Complaints Across Products")
        plt.xlabel("Number of Complaints")
        plt.ylabel("Product")
        plt.tight_layout()
        plt.show()
        print("\nProduct counts:")
        print(counts.to_string())

    def analyze_narrative_length(self):
        self.df["narrative_length"] = (
            self.df["Consumer complaint narrative"]
            .fillna("")
            .apply(lambda x: len(x.split()))
        )

        # Filter to rows that actually have a narrative for the length histogram
        has_narrative = self.df[self.df["Consumer complaint narrative"].notna()]

        print("\nNarrative length statistics (word count):")
        print(has_narrative["narrative_length"].describe().round(1))
        print(f"\nVery short narratives (< 20 words): {(has_narrative['narrative_length'] < 20).sum():,}")
        print(f"Very long narratives  (> 500 words): {(has_narrative['narrative_length'] > 500).sum():,}")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Full distribution
        sns.histplot(has_narrative["narrative_length"], bins=60, ax=axes[0], color="steelblue")
        axes[0].set_title("Narrative Length Distribution (full range)")
        axes[0].set_xlabel("Word Count")
        axes[0].set_ylabel("Number of Complaints")

        # Zoomed-in view (≤ 500 words)
        clipped = has_narrative[has_narrative["narrative_length"] <= 500]
        sns.histplot(clipped["narrative_length"], bins=60, ax=axes[1], color="coral")
        axes[1].set_title("Narrative Length Distribution (≤ 500 words)")
        axes[1].set_xlabel("Word Count")
        axes[1].set_ylabel("Number of Complaints")

        plt.tight_layout()
        plt.show()

    def narrative_presence(self):
        with_narrative = self.df["Consumer complaint narrative"].notna().sum()
        without_narrative = self.df["Consumer complaint narrative"].isna().sum()
        total = len(self.df)

        print(f"\nWith narrative   : {with_narrative:,}  ({with_narrative/total*100:.1f}%)")
        print(f"Without narrative: {without_narrative:,}  ({without_narrative/total*100:.1f}%)")

        if "narrative_length" not in self.df.columns:
            self.df["narrative_length"] = (
                self.df["Consumer complaint narrative"]
                .fillna("")
                .apply(lambda x: len(x.split()))
            )

        has_narrative = self.df[self.df["Consumer complaint narrative"].notna()]
        print(f"Min length : {has_narrative['narrative_length'].min()} words")
        print(f"Max length : {has_narrative['narrative_length'].max()} words")

        # Pie chart
        plt.figure(figsize=(6, 6))
        plt.pie(
            [with_narrative, without_narrative],
            labels=["With Narrative", "Without Narrative"],
            autopct="%1.1f%%",
            colors=["#4CAF50", "#F44336"],
            startangle=90,
        )
        plt.title("Complaints With vs Without Narrative")
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------ #
    #  3. Filter                                                           #
    # ------------------------------------------------------------------ #
    def filter_data(self):
        before = len(self.df)
        self.df = self.df[self.df["Product"].isin(self.ALLOWED_PRODUCTS)]
        after_product = len(self.df)

        self.df = self.df.dropna(subset=["Consumer complaint narrative"])
        after_narrative = len(self.df)

        print(f"\nFiltering:")
        print(f"  Rows before             : {before:,}")
        print(f"  After product filter    : {after_product:,}  (removed {before - after_product:,})")
        print(f"  After narrative filter  : {after_narrative:,}  (removed {after_product - after_narrative:,})")
        print(f"\nProduct distribution after filtering:")
        print(self.df["Product"].value_counts().to_string())

    # ------------------------------------------------------------------ #
    #  4. Clean                                                            #
    # ------------------------------------------------------------------ #
    @staticmethod
    def clean_text(text):
        text = str(text).lower()
        # remove CFPB redaction placeholder
        text = re.sub(r"\bxx+\b", "", text)
        # remove boilerplate openers
        text = re.sub(r"i am writing to (file|submit|report) a complaint[.,]?", "", text)
        text = re.sub(r"to whom it may concern[.,]?", "", text)
        # remove special characters (keep letters, digits, spaces)
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        # collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def clean_narratives(self):
        self.df["cleaned_narrative"] = self.df["Consumer complaint narrative"].apply(self.clean_text)
        self.df["cleaned_length"] = self.df["cleaned_narrative"].apply(lambda x: len(x.split()))

        print("\nSample cleaned narratives:")
        for i, row in self.df[["Consumer complaint narrative", "cleaned_narrative"]].head(3).iterrows():
            print(f"\n  [Original] {row['Consumer complaint narrative'][:120]}...")
            print(f"  [Cleaned]  {row['cleaned_narrative'][:120]}...")

        print(f"\nCleaned length stats:")
        print(self.df["cleaned_length"].describe().round(1))

    # ------------------------------------------------------------------ #
    #  5. Save                                                             #
    # ------------------------------------------------------------------ #
    def save_cleaned_data(self, output_path="data/cleaned_complaints.csv"):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.df.to_csv(output_path, index=False)
        print(f"\nCleaned data saved to: {output_path}  ({len(self.df):,} rows)")

    # ------------------------------------------------------------------ #
    #  Run all steps                                                       #
    # ------------------------------------------------------------------ #
    def run(self, save_output=True):
        self.load_data()
        self.plot_product_distribution()
        self.analyze_narrative_length()
        self.narrative_presence()
        self.filter_data()
        self.clean_narratives()
        if save_output:
            self.save_cleaned_data()


if __name__ == "__main__":
    eda = EDA("data/complaints.csv")
    eda.run()
