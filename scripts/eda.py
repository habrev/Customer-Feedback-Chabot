import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re 


class eda:
    
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None

    def load_data(self):
        self.df = pd.read_csv(self.filepath)
        print("dataset shape before:", self.df.shape)

    def plot_product_distribution(self):
        plt.figure(figsize=(10, 5))
        sns.countplot(data=self.df, y="Product", order=self.df["Product"].value_counts().index)
        plt.title("Distribution of Complaints Across Products")
        plt.xlabel("Count")
        plt.ylabel("Product")
        plt.show()

    def analyze_narrative_length(self):
        self.df["narrative_length"] = self.df["Consumer complaint narrative"].astype(str).apply(lambda x: len(x.split()))
        plt.figure(figsize=(10, 5))
        sns.histplot(self.df["narrative_length"], bins=50)
        plt.title("Distribution of Narrative Length (Word Count)")
        plt.xlabel("Word Count")
        plt.ylabel("Number of Complaints")
        plt.show()

    def narative_presence(self):
        with_narrative = self.df["Consumer complaint narrative"].notna().sum()
        without_narrative = self.df["Consumer complaint narrative"].isna().sum()
        print("With narrative:", with_narrative)
        print("Without narrative:", without_narrative)
        print("Min length:", self.df["narrative_length"].min())
        print("Max length:", self.df["narrative_length"].max())

    def filter_data(self):
        allowed_products = [
            "Credit card",
            "Personal loan",
            "Buy Now, Pay Later (BNPL)",
            "Savings account",
            "Money transfers"
        ]

        self.df = self.df[self.df["Product"].isin(allowed_products)]
        self.df = self.df.dropna(subset=["Consumer complaint narrative"])  # remove empty narratives
        print("Dataset shape after filtering:", df.shape)  


    @staticmethod
    def clean_text(text):
        text = text.lower()  # lowercase
        text = re.sub(r"[^a-z0-9\s]", "", text)  # remove special characters
        text = re.sub(r"\s+", " ", text).strip()  # normalize whitespace        
        # remove common boilerplate phrases (example)
        text = re.sub(r"i am writing to file a complaint", "", text)
        return text

    def clean_narratives(self):
        self.df["cleaned_narrative"] = self.df["Consumer complaint narrative"].apply(self.clean_text)
        print("\nSample cleaned narratives:")
        print(self.df["cleaned_narrative"].head(5))

