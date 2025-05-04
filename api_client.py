import aiohttp
import asyncio
import json
import os
import time
import random
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from openai import AsyncOpenAI
from dotenv import load_dotenv
from typing import Dict, Union, List, Any

from logger import get_logger, log_function_call, log_info, log_warning, log_error
from config import (
    NEWSAPI_KEY, 
    OPENAI_API_KEY, 
    MAX_CONTENT_LENGTH,
    REQUEST_DELAY_MIN,
    REQUEST_DELAY_MAX
)

# Get logger
logger = get_logger()

# Initialize sentiment analysis with caching
_sentiment_analyzer = None

class SentimentResult:
    """Class to represent sentiment analysis results."""
    
    def __init__(self, label: str = "NEUTRAL", score: float = 0.0):
        self.label = label
        self.score = score
        
    def to_dict(self) -> Dict[str, Union[str, float]]:
        """Convert to dictionary."""
        return {
            "label": self.label,
            "score": self.score
        }

async def get_sentiment_analyzer():
    """Get or initialize the OpenAI sentiment analysis."""
    global _sentiment_analyzer
    
    if _sentiment_analyzer is None:
        try:
            log_info("Initializing OpenAI for sentiment analysis...")
            
            # Configure the OpenAI client
            if not OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY is not set in .env file")
                
            client = AsyncOpenAI(api_key=OPENAI_API_KEY)
            
            # Create a function to analyze sentiment using OpenAI
            async def analyze_with_openai(text: str) -> SentimentResult:
                if not text or len(text.strip()) == 0:
                    return SentimentResult()
                
                # Truncate text to avoid token limits
                truncated_text = text[:MAX_CONTENT_LENGTH] if len(text) > MAX_CONTENT_LENGTH else text
                
                # Prompt for sentiment analysis
                try:
                    response = await client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": "You are a sentiment analysis expert. Analyze the text and respond with ONLY a JSON object containing 'label' (either 'POSITIVE', 'NEGATIVE', or 'NEUTRAL') and 'score' (a number between -1.0 and 1.0)."},
                            {"role": "user", "content": f"Analyze the sentiment of this text: {truncated_text}"}
                        ],
                        temperature=0,
                        max_tokens=100
                    )
                    
                    response_text = response.choices[0].message.content
                    
                    # Extract JSON from response
                    try:
                        # Find JSON in the response
                        json_start = response_text.find('{')
                        json_end = response_text.rfind('}')
                        
                        if json_start >= 0 and json_end >= 0:
                            json_str = response_text[json_start:json_end+1]
                            result = json.loads(json_str)
                            
                            # Validate the result format
                            if 'label' in result and 'score' in result:
                                # Ensure label is one of the expected values
                                if result['label'] not in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']:
                                    result['label'] = 'NEUTRAL'
                                    
                                # Ensure score is within expected range
                                score = float(result['score'])
                                result['score'] = max(min(score, 1.0), -1.0)
                                
                                return SentimentResult(result['label'], result['score'])
                    except Exception as json_error:
                        log_error(f"Error parsing OpenAI response: {json_error}", exc_info=True)
                    
                    # Fallback based on raw response
                    if 'positive' in response_text.lower():
                        return SentimentResult("POSITIVE", 0.7)
                    elif 'negative' in response_text.lower():
                        return SentimentResult("NEGATIVE", -0.7)
                    else:
                        return SentimentResult()
                        
                except Exception as e:
                    log_error(f"Error calling OpenAI API: {e}", exc_info=True)
                    return SentimentResult()
            
            _sentiment_analyzer = analyze_with_openai
            log_info("OpenAI initialized successfully!")
            
        except Exception as e:
            log_error(f"Error initializing OpenAI: {e}", exc_info=True)
            log_warning("Falling back to basic sentiment analysis...")
            
            # Define a simple rule-based sentiment analyzer
            async def simple_sentiment_analyzer(text: str) -> SentimentResult:
                if not text or len(text.strip()) == 0:
                    return SentimentResult()
                
                # Convert to lowercase for case-insensitive matching
                text = text.lower()
                
                # Define positive and negative word lists
                positive_words = [
                    "good", "great", "excellent", "positive", "profit", "profits", "growth", 
                    "increase", "up", "higher", "best", "success", "successful", "gain",
                    "improve", "improved", "improving", "innovation", "innovative", "exceed",
                    "exceeded", "beating", "record", "strong", "strength", "robust", "progress"
                ]
                
                negative_words = [
                    "bad", "poor", "negative", "loss", "losses", "decline", "decrease", "down",
                    "lower", "worst", "fail", "failed", "failure", "drop", "dropped", "weak",
                    "weakness", "concern", "concerned", "worry", "risk", "risky", "problem",
                    "issue", "trouble", "difficult", "challenging", "disappointing", "disappointed",
                    "miss", "missed", "below", "recall", "lawsuit", "investigation", "crash"
                ]
                
                # Count occurrences
                positive_count = sum(1 for word in positive_words if word in text)
                negative_count = sum(1 for word in negative_words if word in text)
                
                # Calculate score between -1 and 1
                total = positive_count + negative_count
                if total == 0:
                    return SentimentResult()
                
                score = (positive_count - negative_count) / (positive_count + negative_count)
                
                # Determine label
                if score > 0.2:
                    label = "POSITIVE"
                elif score < -0.2:
                    label = "NEGATIVE"
                else:
                    label = "NEUTRAL"
                
                return SentimentResult(label, score)
            
            _sentiment_analyzer = simple_sentiment_analyzer
            log_info("Simple sentiment analyzer initialized as fallback")
    
    return _sentiment_analyzer

class NewsClient:
    """Client for fetching news from NewsAPI with async support."""
    
    def __init__(self):
        self.api_key = NEWSAPI_KEY
        self.base_url = "https://newsapi.org/v2/everything"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.session = None
    
    async def __aenter__(self):
        """Create aiohttp session for context manager usage."""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close aiohttp session."""
        if self.session:
            await self.session.close()
            self.session = None
    
    @log_function_call
    async def fetch_article_content(self, url: str) -> str:
        """Fetch and extract the main content from an article URL using BeautifulSoup."""
        if not url:
            return ""
            
        try:
            # Add a small delay to avoid overwhelming the server
            await asyncio.sleep(random.uniform(REQUEST_DELAY_MIN, REQUEST_DELAY_MAX))
            
            # Create a session if not already created
            if not self.session:
                self.session = aiohttp.ClientSession()
                
            # Make the request to the article URL
            async with self.session.get(url, headers=self.headers, timeout=10) as response:
                if response.status != 200:
                    log_warning(f"Error fetching {url}: HTTP {response.status}")
                    return ""
                
                html_content = await response.text()
                
            # Parse the HTML content
            soup = BeautifulSoup(html_content, 'lxml')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
                
            # Get the text content
            text = soup.get_text(separator=' ', strip=True)
            
            # Clean up the text (remove extra whitespace)
            text = ' '.join(text.split())
                
            log_info(f"Successfully extracted content from {url} ({len(text)} chars)")
            return text
            
        except Exception as e:
            log_error(f"Error extracting content from {url}: {e}", exc_info=True)
            return ""
    
    @log_function_call
    async def fetch_mentions(self, company_name: str, aliases: List[str], days: int = 7, limit: int = 15) -> List[Dict[str, Any]]:
        """Fetch mentions of a company from NewsAPI asynchronously.
        
        Args:
            company_name: Name of the company
            aliases: List of company aliases
            days: Number of days to look back
            limit: Maximum number of articles to return (default: 15)
            
        Returns:
            List of mention dictionaries
        """
        if not self.api_key:
            log_error("NewsAPI key is not set")
            raise ValueError("NewsAPI key is not set. Please set NEWSAPI_KEY in .env file.")
        
        # Combine company name and aliases for search
        search_terms = [company_name] + [alias.strip() for alias in aliases if alias.strip()]
        query = " OR ".join([f'"{term}"' for term in search_terms])
        
        # Calculate date range
        to_date = datetime.now().strftime('%Y-%m-%d')
        from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        # Request parameters
        params = {
            'q': query,
            'from': from_date,
            'to': to_date,
            'language': 'en',
            'sortBy': 'publishedAt',
            'pageSize': min(limit, 100),  # NewsAPI limits to 100 max per request
            'apiKey': self.api_key
        }
        
        try:
            # Create a session if not already created
            if not self.session:
                self.session = aiohttp.ClientSession()
                
            # Make API request
            async with self.session.get(self.base_url, params=params) as response:
                if response.status != 200:
                    log_error(f"Error from NewsAPI: HTTP {response.status}")
                    return []
                
                data = await response.json()
            
            if data.get('status') != 'ok':
                log_error(f"Error from NewsAPI: {data.get('message', 'Unknown error')}")
                return []
            
            # Process and normalize results
            mentions = []
            article_content_tasks = []
            articles = data.get('articles', [])[:limit]  # Apply limit here too
            
            # Create tasks for fetching article content
            for article in articles:
                url = article.get('url', '')
                article_content_tasks.append(self.fetch_article_content(url))
            
            # Run all content fetching tasks concurrently
            article_contents = await asyncio.gather(*article_content_tasks)
            
            # Process the results
            for i, article in enumerate(articles):
                published_at = None
                if article.get('publishedAt'):
                    try:
                        published_at = datetime.strptime(article['publishedAt'], '%Y-%m-%dT%H:%M:%SZ')
                    except ValueError:
                        try:
                            published_at = datetime.strptime(article['publishedAt'], '%Y-%m-%dT%H:%M:%S.%fZ')
                        except ValueError:
                            published_at = None
                
                # Get the article URL
                url = article.get('url', '')
                
                # Get the content from the completed task
                scraped_content = article_contents[i]
                
                # If BeautifulSoup returns empty content, use NewsAPI content as fallback
                if not scraped_content or scraped_content.strip() == "":
                    # Use description or content from NewsAPI as fallback
                    api_content = article.get('content', "")
                    api_description = article.get('description', "")
                    
                    # Prefer content over description if available
                    if api_content and len(api_content) > 10:  # Ensure it's not just a short snippet
                        scraped_content = api_content
                        log_info(f"Using NewsAPI content as fallback for {url}")
                    elif api_description and len(api_description) > 10:
                        scraped_content = api_description
                        log_info(f"Using NewsAPI description as fallback for {url}")
                    else:
                        log_warning(f"No content available for {url} from either scraping or NewsAPI")
            
                mention = {
                    'title': article.get('title', 'No title'),
                    'content': scraped_content,
                    'url': url,
                    'source': article.get('source', {}).get('name', 'Unknown'),
                    'published_at': published_at
                }
                mentions.append(mention)
            
            log_info(f"Found {len(mentions)} mentions for {company_name} (limited to {limit})")
            return mentions
            
        except Exception as e:
            log_error(f"Error fetching mentions: {e}", exc_info=True)
            return []

async def analyze_sentiment(text: str) -> Dict[str, Union[str, float]]:
    """Analyze sentiment using OpenAI."""
    if not text:
        log_info("Empty text provided for sentiment analysis, returning NEUTRAL")
        return {"label": "NEUTRAL", "score": 0.0}
    
    try:
        # Get or initialize the sentiment analyzer
        sentiment_analyzer = await get_sentiment_analyzer()
        
        # Use a reasonable text length to avoid issues with very long texts
        truncated_text = text[:1000] if text else ""
        
        # Analyze sentiment
        result = await sentiment_analyzer(truncated_text)
        
        return result.to_dict()
    except Exception as e:
        log_error(f"Error analyzing sentiment: {e}", exc_info=True)
        return {"label": "NEUTRAL", "score": 0.0}

async def analyze_mentions(mentions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Analyze sentiment for a list of mentions asynchronously."""
    log_info(f"Analyzing sentiment for {len(mentions)} mentions")
    enriched_mentions = []
    
    # Load analyzer once for all mentions
    await get_sentiment_analyzer()
    
    # Create tasks for sentiment analysis
    analysis_tasks = []
    for mention in mentions:
        # Combine title and content for better analysis
        text = f"{mention.get('title', '')} {mention.get('content', '')}"
        analysis_tasks.append(analyze_sentiment(text))
    
    # Run all analysis tasks concurrently
    sentiment_results = await asyncio.gather(*analysis_tasks)
    
    # Process the results
    for i, mention in enumerate(mentions):
        sentiment = sentiment_results[i]
        
        # Add sentiment to mention
        enriched_mention = mention.copy()
        enriched_mention['sentiment'] = sentiment["label"]
        enriched_mention['sentiment_score'] = sentiment["score"]
        
        enriched_mentions.append(enriched_mention)
    
    return enriched_mentions

# Example usage
async def main():
    # Test NewsAPI client
    async with NewsClient() as news_client:
        test_mentions = await news_client.fetch_mentions("Tesla", ["TSLA", "Tesla Inc."], days=3)
        log_info(f"Found {len(test_mentions)} mentions")
        
        if test_mentions:
            # Test sentiment analysis
            sentiment_result = await analyze_sentiment(test_mentions[0]["title"])
            log_info(f"Sentiment: {sentiment_result['label']}, Score: {sentiment_result['score']}")
            
            # Test batch sentiment analysis
            enriched_mentions = await analyze_mentions(test_mentions[:2])
            log_info(f"Analyzed {len(enriched_mentions)} mentions")
            for mention in enriched_mentions:
                log_info(f"Title: {mention['title'][:50]}...")
                log_info(f"Sentiment: {mention['sentiment']}, Score: {mention['sentiment_score']}")
                log_info("---")

if __name__ == "__main__":
    # Initialize logging
    from src.utils.logger import log_startup, log_shutdown
    log_startup()
    
    # Run the async main function
    asyncio.run(main())
    
    # Log shutdown
    log_shutdown()