import glob
import streamlit as st
import plotly.express as px
import pandas as pd

from nltk.sentiment import SentimentIntensityAnalyzer

filepaths = sorted(glob.glob("diary/*.txt"))

analyzer = SentimentIntensityAnalyzer()

negativity = []
positivity = []
for filepath in filepaths:
    with open(filepath, "r") as file:
        content = file.read()

    scores = analyzer.polarity_scores(content)
    positivity.append(scores["pos"])
    negativity.append(scores["neg"])

dates = [name.strip(".txt").strip("diary/") for name in filepaths]

# make dataframes
df_pos = pd.DataFrame({"Date": dates, "Positivity": positivity})
df_neg = pd.DataFrame({"Date": dates, "Negativity": negativity})

st.title("Diary Tone")
st.subheader("Positivity")
pos_figure = px.line(x=dates, y=positivity,
                     labels={"x": "Date", "y": "Positivity"})

st.plotly_chart(pos_figure)

st.subheader("Negativity")
neg_figure = px.line(x=dates, y=negativity,
                     labels={"x": "Date", "y": "Negativity"})
st.plotly_chart(neg_figure)
# st.subheader("Positivity")
# pos_figure = px.line(df_pos, x="Date", y="Positivity")
# st.plotly_chart(pos_figure)
#
# st.subheader("Negativity")
# neg_figure = px.line(df_neg, x="Date", y="Negativity")
# st.plotly_chart(neg_figure)

# Create a wide-format DataFrame
df = pd.DataFrame({
    "Date": dates,
    "Positivity": positivity,
    "Negativity": negativity
})

# Reshape to long format
df_long = df.melt(id_vars="Date", value_vars=["Positivity", "Negativity"],
                  var_name="Tone", value_name="Score")
# Define custom colors
custom_colors = {
    "Positivity": "green",
    "Negativity": "red"
}

# Plot both lines on the same chart
fig = px.line(df_long, x="Date", y="Score", color="Tone", color_discrete_map=custom_colors,
              labels={"Date": "Date", "Score": "Tone Score", "Tone": "Tone"},
              title="Diary Tone Over Time")

st.plotly_chart(fig)