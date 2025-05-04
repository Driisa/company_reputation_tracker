# Company Reputation Tracker

A AI powered application that tracks company mentions from news sources, analyzes sentiment using OpenAI, and presents insights through an interactive dashboard.

## Visit the website:
    https://vitwip.github.io/Company_reputation_tracker-main/
   
## Features

- **Company & Aliases** – Input multiple names or aliases for each company.  
- **News Fetching** – Pull mentions from [NewsAPI](https://newsapi.org/) for the past days.  
- **Content Extraction** – Uses BeautifulSoup to retrieve full article text for more accurate analysis.
- **Advanced Sentiment Analysis** – Leverages OpenAI's GPT-4o-mini model to label each mention as POSITIVE, NEUTRAL, or NEGATIVE, with a score between -1 and +1.  
- **Static Dashboard** – Visualize sentiment distribution, timeline trends, and detailed mention tables in real-time.  
- **Static GitHub Pages Dashboard** – A static version of the dashboard available online without running the Python application.
- **SQLite Database** – Persist all companies, aliases, and mentions in a local database.  
- **GitHub Actions Automation** – Automatically fetch and analyze new mentions every day (6:00 AM UTC) and push updates back to the repository.  
- **Logging** – A robust logging system tracks errors, warnings, and pipeline updates.

---


## GitHub Actions for Automated Updates

1. **Configure Secrets**:  
   - In your GitHub repo → **Settings** → **Secrets and variables** → **Actions**  
   - Add `NEWSAPI_KEY` and `OPENAI_API_KEY` as repository secrets.

2. **Daily Schedule**:  
   - The included workflow (`.github/workflows/daily_tracker.yml`) runs every day at 6:00 AM UTC:
     - Fetches & analyzes new mentions for all companies.
     - Commits the updated database or logs back to the repository.
     - Stores any run artifacts for reference.
     - Generates static data for the GitHub Pages dashboard.

3. **Manual Trigger**:  
   - In **Actions** → **Daily Company Reputation Tracker** → **Run workflow**.

4. **Current Status**:  
   - GitHub Actions is now fully configured and operational.
   - The workflow automatically updates the GitHub Pages dashboard with fresh data.

---

## Pipeline Process

The Company Reputation Tracker uses an automated pipeline to collect, analyze, and visualize company mentions. Here's how the entire process works:

1. **Data Collection**:
   - The pipeline starts by querying the NewsAPI for articles mentioning each company and its aliases.
   - Articles from the past days are retrieved (configurable timeframe).
   - For each article, metadata like title, source, publication date, and URL are collected.

2. **Content Extraction**:
   - The system uses BeautifulSoup to scrape the full text content from each article URL.
   - This provides more context for accurate sentiment analysis compared to just using headlines or snippets.
   - If an article can't be scraped, the system falls back to using the available description from NewsAPI.

3. **Sentiment Analysis**:
   - Each article's content is processed by OpenAI's GPT-4o-mini model.
   - The AI evaluates the sentiment specifically regarding the company mentioned.
   - Articles receive both a categorical label (POSITIVE, NEUTRAL, NEGATIVE) and a numerical score (-1.0 to +1.0).
   - If OpenAI is unavailable, a fallback rule-based analyzer is used instead.

4. **Database Storage**:
   - All mentions and their analysis results are stored in the SQLite database.
   - The system avoids duplicates by checking against existing URLs.
   - Historical data is preserved for trend analysis.

5. **Dashboard Updates**:
   - The dashboard reads from the database to display real-time insights.
   - Visualizations are automatically refreshed when new data is available.
   - For GitHub Pages, static JSON files are generated to power the online dashboard.

6. **Automated Execution**:
   - GitHub Actions runs this entire pipeline daily at 6:00 AM UTC.
   - The workflow commits any database changes back to the repository.
   - Logs are generated at each step for troubleshooting and auditing.

---

## Advanced Sentiment Analysis

- **OpenAI**: By providing `OPENAI_API_KEY`, the application uses the "gpt-4o-mini" model to analyze text sentiment.
- **Scoring**: Ranges from `-1.0` (very negative) to `1.0` (very positive).
- **Thresholds**: Mentions are labeled `POSITIVE`, `NEGATIVE`, or `NEUTRAL`.
- **Article Content**: BeautifulSoup scrapes the full body of each article to improve analysis accuracy.
- **Fallback**: If OpenAI is unavailable, a simple rule-based sentiment analyzer will be used.

---

## Project Structure

```
company_tracker/
├── api_client.py           # Fetches news and applies sentiment analysis
├── db.py                   # SQLite models & utility functions
├── index.html              # Dash web application
├── runner.py               # Main script for fetching & analyzing mentions
├── setup.py                # Setup script for easy installation
├── logger.py               # Logging utilities
├── generate_static_data.py # Transform data from sql file to json format
├── requirements.txt
├── .github/
│   └── workflows/
│       └── daily_tracker.yml  # GitHub Actions pipeline
└── README.md
```

---

## Dashboard Highlights

- **Sentiment Overview**: Bar chart showing distribution of POSITIVE, NEUTRAL, NEGATIVE mentions.  
- **Average Score**: Real-time sentiment performance, updated with new mentions.  
- **Sentiment Timeline**: See how sentiment changes over time, with optional trend lines.  
- **Recent Mentions**: Table of the latest articles, color-coded by sentiment.  

---

## Troubleshooting

1. **API Errors**: 
   - Ensure `NEWSAPI_KEY` is correct and that you have an active internet connection.
   - Check if you've exceeded your NewsAPI daily limit (60 requests for free tier).

2. **OpenAI Errors**: 
   - If `OPENAI_API_KEY` is missing or invalid, the app falls back to a simple rule-based sentiment analyzer.
   - If you're seeing quota errors, check your OpenAI usage limits.

3. **Logs**: 
   - Check the `logs/` folder for detailed error messages and pipeline steps.

4. **Database Issues**:
   - If the database becomes corrupted, delete `company_tracker.db` and run `db.py` to recreate it.
   - Consider making regular backups of your database if you have important data.

5. **GitHub Pages**:
   - You can access the live dashboard at `https://vitwip.github.io/Company_reputation_tracker-main/`.
   - The static dashboard is automatically updated by GitHub Actions with fresh data.
   - If the dashboard isn't displaying correctly, ensure your repository is configured for GitHub Pages in Settings → Pages → Build and deployment → Source: "GitHub Actions".

---

## Contributers

- Oliver Højbjerre-Frandnsen
- Vittorio Wollner Infante Papa
- Daniel Sørensen Riisager

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

Enjoy using the **Company Reputation Tracker**!
