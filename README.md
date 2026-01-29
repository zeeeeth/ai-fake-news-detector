# ai-fake-news-detector

## Project Overview
<ins>Description:</ins> A fullstack application that uses AI to detect fake news articles or claims by analysing text content and predicting the likelihood that an article is misleading or false.

<ins>Problem Solved:</ins> Helps combat misinformation by providing users with a tool to verify the credibility of news articles. 

## Datasets
<ins>ISOT Fake News Dataset:</ins> Provides a database of fake and real news articles. I used columns "label", "title", "text", "subject" and "date" to train the article analyser.

<ins>LIAR Dataset:</ins> Provides a database of statements, with their truth rated on a scale of pants-fire (least true), false, barely-true, half-true, mostly-true, true (most true). I used columns "statement", "label", "speaker", "party" and "context" to train the article analyser

## Tech Stack
<ins>Frontend:</ins> React.js for the framework, Material-MUI for styling
<ins>Backend:</ins> Python with Flask for handling API requests
<ins>AI Models:</ins> PyTorch, Hugging Face Transformers

## Challenges and Considerations
<ins>Bias:</ins> Potential biases in training data




Train article_roberta model on ISOT: python -m src.training.train_isot --epochs 1 --max_length 128 --limit 20000 --batch_size 8