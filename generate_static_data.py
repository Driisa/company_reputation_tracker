import json
import os
from datetime import datetime
from db import get_companies, get_company, get_mentions, get_sentiment_stats, get_sentiment_timeline_data, init_db
import traceback
from logger import get_logger, log_info, log_error

# Get logger
logger = get_logger()

# Ensure the data directory exists
data_dir = os.path.join(os.path.dirname(__file__), 'assets', 'data')
os.makedirs(data_dir, exist_ok=True)

def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")

def generate_company_list():
    """Generate a JSON file with the list of all companies"""
    try:
        # Initialize DB before accessing
        init_db()
        
        companies = get_companies()
        company_list = [{
            'id': company.id,
            'name': company.name,
            'aliases': company.aliases.split(',') if company.aliases else []
        } for company in companies]
        
        with open(os.path.join(data_dir, 'companies.json'), 'w') as f:
            json.dump(company_list, f, default=json_serial)
        
        log_info(f"Generated companies.json with {len(company_list)} companies")
        return company_list
    except Exception as e:
        log_error(f"Error generating company list: {str(e)}")
        traceback.print_exc()
        return []

def generate_company_data(company_id):
    """Generate JSON data for a specific company"""
    try:
        company = get_company(company_id)
        if not company:
            log_error(f"Company with ID {company_id} not found")
            return None
        
        # Get company stats
        stats = get_sentiment_stats(company_id)
        
        # Get company mentions
        mentions = get_mentions(company_id)
        mentions_data = [{
            'id': mention.id,
            'title': mention.title,
            'content': mention.content,
            'sentiment': mention.sentiment,
            'sentiment_score': mention.sentiment_score,
            'url': mention.url,
            'source': mention.source,
            'published_at': mention.published_at
        } for mention in mentions]
        
        # Get timeline data
        timeline_mentions = get_sentiment_timeline_data(company_id)
        timeline_data = [{
            'date': mention.published_at,
            'score': mention.sentiment_score,
            'sentiment': mention.sentiment
        } for mention in timeline_mentions if mention.published_at and mention.sentiment_score]
        
        # Create company data object
        company_data = {
            'company': {
                'id': company.id,
                'name': company.name,
                'aliases': company.aliases.split(',') if company.aliases else []
            },
            'stats': stats,
            'mentions': mentions_data,
            'timeline': timeline_data
        }
        
        # Save to file
        with open(os.path.join(data_dir, f'company_{company_id}.json'), 'w') as f:
            json.dump(company_data, f, default=json_serial)
        
        log_info(f"Generated data for company: {company.name} (ID: {company_id})")
        return company_data
    except Exception as e:
        log_error(f"Error generating data for company {company_id}: {str(e)}")
        traceback.print_exc()
        return None

def generate_all_data():
    """Generate all JSON data files"""
    # Initialize DB before accessing
    init_db()
    
    # Generate company list
    companies = generate_company_list()
    
    # Generate data for each company
    all_company_data = []
    for company in companies:
        company_data = generate_company_data(company['id'])
        if company_data:
            all_company_data.append(company_data)
    
    # Generate a combined data file with the first company's data as default
    if all_company_data:
        default_company_data = all_company_data[0]
        with open(os.path.join(data_dir, 'dashboard_data.json'), 'w') as f:
            json.dump(default_company_data, f, default=json_serial)
        log_info(f"Generated dashboard_data.json with default company: {default_company_data['company']['name']}")
    else:
        log_error("No company data was generated. Dashboard data will not be updated.")
        
    # Generate a timestamp file to track when the data was last updated
    timestamp_data = {
        "last_updated": datetime.now().isoformat(),
        "companies_processed": len(companies),
        "success": len(all_company_data)
    }
    with open(os.path.join(data_dir, 'last_update.json'), 'w') as f:
        json.dump(timestamp_data, f)

if __name__ == "__main__":
    generate_all_data()
    print("Static data generation complete!")