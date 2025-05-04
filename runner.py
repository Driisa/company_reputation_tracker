#!/usr/bin/env python
"""
Main runner script for the Company Reputation Tracker.
This script runs the reputation tracking pipeline for companies.
"""

import argparse
import json
import os
import asyncio
from datetime import datetime

import db
from api_client import NewsClient, analyze_mentions
from logger import get_logger, log_function_call, log_startup, log_shutdown, log_info, log_warning, log_error
from config import NEWSAPI_KEY, OPENAI_API_KEY, ARTICLE_LIMIT


# Get logger
logger = get_logger()

# Check for API keys and import API client only if keys are available
try:
    from api_client import NewsClient, analyze_mentions
    
    # Check for required API keys
    news_api_key = os.environ.get('NEWSAPI_KEY')
    openai_api_key = os.environ.get('OPENAI_API_KEY')
    
    API_AVAILABLE = True
    if not news_api_key:
        log_warning("NEWSAPI_KEY environment variable not set. News fetching will be skipped.")
        API_AVAILABLE = False
    if not openai_api_key:
        log_warning("OPENAI_API_KEY environment variable not set. Sentiment analysis will be skipped.")
        API_AVAILABLE = False
    
    if not API_AVAILABLE:
        log_warning("Running in data generation mode only - no new data will be fetched.")
except ImportError as e:
    log_error(f"API client import failed: {str(e)}")
    API_AVAILABLE = False

@log_function_call
def process_company(company_id, article_limit=15):
    """Process a single company.
    
    Args:
        company_id: ID of the company to process
        article_limit: Maximum number of articles to process (default: 15)
    """
    # Get company data
    company = db.get_company(company_id)
    if not company:
        log_error(f"Company with ID {company_id} not found.")
        return None
    
    log_info(f"Processing company: {company.name} (ID: {company.id})")
    
    # Skip API calls if not available
    if not API_AVAILABLE:
        log_warning(f"Skipping API calls for {company.name} - API not available")
        return {
            "company_name": company.name,
            "company_id": company.id,
            "mentions_added": 0,
            "status": "skipped",
            "message": "API not available. Skipping data fetching."
        }
    
    # Get company aliases
    aliases = db.get_company_aliases(company_id)
    
    # 1. Fetch mentions
    news_client = NewsClient()
    try:
        mentions = news_client.fetch_mentions(company.name, aliases, limit=article_limit)
        log_info(f"Found {len(mentions)} mentions for {company.name}")
        
        if not mentions:
            return {
                "company_name": company.name,
                "company_id": company.id,
                "mentions_added": 0,
                "status": "success",
                "message": "No new mentions found."
            }
    except Exception as e:
        log_error(f"Error fetching mentions: {str(e)}", exc_info=True)
        return {
            "company_name": company.name,
            "company_id": company.id,
            "status": "error",
            "message": f"Error fetching mentions: {str(e)}"
        }
    
    # 2. Analyze sentiment
    try:
        enriched_mentions = analyze_mentions(mentions)
        log_info(f"Completed sentiment analysis for {len(enriched_mentions)} mentions")
    except Exception as e:
        log_error(f"Error analyzing sentiment: {str(e)}", exc_info=True)
        return {
            "company_name": company.name,
            "company_id": company.id,
            "status": "error",
            "message": f"Error analyzing sentiment: {str(e)}"
        }
    
    # 3. Save to database
    try:
        mentions_added = db.add_mentions(company_id, enriched_mentions)
        log_info(f"Added {mentions_added} mentions to the database")
        
        # Ensure database changes are committed
        session = db.get_db()
        try:
            session.commit()
            log_info(f"Database changes for company {company.name} have been committed")
        except Exception as e:
            session.rollback()
            log_error(f"Error committing database changes: {str(e)}", exc_info=True)
        finally:
            session.close()
    except Exception as e:
        log_error(f"Error saving mentions: {str(e)}", exc_info=True)
        return {
            "company_name": company.name,
            "company_id": company.id,
            "status": "error",
            "message": f"Error saving mentions: {str(e)}"
        }
    
    # 4. Get updated stats
    stats = db.get_sentiment_stats(company_id)
    
    # Return summary
    return {
        "company_name": company.name,
        "company_id": company.id,
        "mentions_added": mentions_added,
        "status": "success",
        "stats": stats
    }

def run_all_companies(article_limit=15):
    """Process all companies in the database."""
    # Initialize database
    db.init_db()
    
    # Get all companies
    companies = db.get_companies()
    if not companies:
        log_warning("No companies found in the database.")
        return {
            "timestamp": datetime.now().isoformat(),
            "status": "warning",
            "message": "No companies found in the database."
        }
    
    results = []
    for company in companies:
        result = process_company(company.id, article_limit)
        results.append(result)
    
    # Ensure database changes are committed
    session = db.get_db()
    try:
        session.commit()
        log_info("All database changes have been committed")
    except Exception as e:
        session.rollback()
        log_error(f"Error committing database changes: {str(e)}", exc_info=True)
    finally:
        session.close()
    
    # Generate summary
    successful = sum(1 for r in results if r and r.get("status") == "success")
    skipped = sum(1 for r in results if r and r.get("status") == "skipped")
    total_mentions = sum(r.get("mentions_added", 0) for r in results if r and r.get("status") == "success")
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "companies_processed": len(results),
        "successful": successful,
        "skipped": skipped,
        "failed": len(results) - successful - skipped,
        "total_new_mentions": total_mentions,
        "details": results
    }
    
    # Save summary to file
    with open("pipeline_run_report.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    log_info(f"Pipeline completed. Processed {len(results)} companies, added {total_mentions} new mentions.")
    return summary

def add_new_company(name, aliases):
    """Add a new company to track."""
    # Initialize database
    db.init_db()
    
    company = db.add_company(name, aliases)
    if company:
        log_info(f"Added company: {name} (ID: {company.id})")
        return company.id
    else:
        log_error(f"Failed to add company: {name}")
        return None

if __name__ == "__main__":
    log_startup("Company Reputation Tracker")
    
    parser = argparse.ArgumentParser(description="Company Reputation Tracker")
    parser.add_argument("--company", type=int, help="Company ID to process")
    parser.add_argument("--all", action="store_true", help="Process all companies")
    parser.add_argument("--add", action="store_true", help="Add a new company")
    parser.add_argument("--name", type=str, help="Company name (for --add)")
    parser.add_argument("--aliases", type=str, help="Company aliases, comma-separated (for --add)")
    parser.add_argument("--limit", type=int, default=15, help="Limit the number of articles to process (default: 10)")
    parser.add_argument("--generate-only", action="store_true", help="Skip API calls and only generate static data")
    
    args = parser.parse_args()
    
    # If no arguments are provided, default to --all
    if not any([args.company, args.all, args.add, args.generate_only]):
        args.all = True
    
    if args.generate_only:
        log_info("Running in generate-only mode. Skipping API calls.")
        # We'll proceed to generate static data without API calls
    
    if args.add:
        if not args.name:
            log_error("Error: --name is required when adding a company")
        else:
            aliases = args.aliases.split(",") if args.aliases else []
            add_new_company(args.name, aliases)
    
    elif args.company:
        result = process_company(args.company, args.limit)
        log_info(json.dumps(result, indent=2))
    
    elif args.all:
        run_all_companies(args.limit)
    
    else:
        # List all companies
        db.init_db()
        companies = db.get_companies()
        
        if companies:
            log_info("Available companies:")
            for company in companies:
                aliases = company.aliases.split(",") if company.aliases else []
                log_info(f"  ID {company.id}: {company.name} (Aliases: {', '.join(aliases)})")
        else:
            log_warning("No companies found. Add a company with --add --name COMPANY_NAME")
    
    # Always generate static data at the end
    try:
        log_info("Generating static data files...")
        from generate_static_data import generate_all_data
        generate_all_data()
        log_info("Static data generation completed")
    except Exception as e:
        log_error(f"Error generating static data: {str(e)}", exc_info=True)
    
    log_shutdown("Company Reputation Tracker")
