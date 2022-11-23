import re
import pandas as pd

df_HR = pd.read_csv('data/HR.csv')
df_Legal = pd.read_csv('data/Legal.csv')
df_Finance = pd.read_csv('data/Finance.csv')


def clean_data(df):
    for idx, row in df.iterrows():
        if not isinstance(row["Topic"], str) or not isinstance(row["Body"], str):
            df.drop([idx], axis=0, inplace=True)  # Drop any rows that contain non-string values
        else:
            string_topic = row["Topic"]
            string_topic = re.sub(r'\[([^]]+)\]', ' ', string_topic)  # Remove anything in square brackets
            string_topic = re.sub(r'\s{2,}', ' ', string_topic)  # Remove excess whitespace between words
            df.at[idx, "Topic"] = string_topic

            string_body = row["Body"]
            string_body = string_body.strip()
            string_body = re.sub(r'\[([^]]+)\]', ' ', string_body)  # Remove anything in square brackets
            df.at[idx, "Body"] = string_body

        df["Text"] = df["Topic"] + " " + df["Body"]  # Concatenate text columns
        df_out = df.drop(["Topic", "Body"], axis=1)  # Drop surplus columns
        df_out = df_out.reset_index(drop=True)

    return df_out


df_HR = clean_data(df_HR)
df_HR["Tag"] = 0
df_Legal = clean_data(df_Legal)
df_Legal["Tag"] = 1
df_Finance = clean_data(df_Finance)
df_Finance["Tag"] = 2

# HR[0] Legal[1] Finance[2]

df_data = pd.concat([df_HR, df_Legal, df_Finance])
df_data = df_data.sample(frac=1, random_state=42).reset_index(drop=True)
df_data.to_csv('data/set/data.csv', index=False)
