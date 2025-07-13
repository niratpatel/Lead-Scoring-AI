
    # Lead Scoring Dataset - Feature Documentation
    
    ## ML Model Training Features (Structured Data):
    These features are used to train the machine learning model:
    
    ### Core Predictive Features:
    1. **credit_score**: CIBIL score (550-850) - Financial capability indicator
    2. **income_inr**: Annual income in INR - Purchasing power
    3. **time_spent_website_secs**: Website engagement time - Behavioral signal
    4. **lead_source**: Marketing channel - Source quality varies
    5. **age_group**: Life stage - Affects buying propensity
    6. **form_submissions**: Number of forms filled - Engagement level
    7. **previous_interactions**: Returning visitor indicator
    8. **page_views**: Website exploration depth
    
    ### Additional Features:
    - **city**: Location tier affects purchasing power
    - **property_type_preference**: Specific interest area
    - **education**: Educational background
    - **profession**: Job category
    - **family_background**: Family status
    - **day_of_week**: Temporal pattern
    - **hour_of_day**: Time-based behavior
    - **last_interaction_channel**: Communication preference
    
    ## Reranking Features (Unstructured Data):
    These features are used ONLY for reranking after ML prediction:
    
    ### Text-based Features:
    1. **comments**: User comments/feedback - Used for sentiment analysis and intent detection
    
    ## Key Design Principles:
    - **Separation of Concerns**: ML model handles structured data, reranker handles text
    - **No Data Leakage**: Target variable generated WITHOUT considering comments
    - **Realistic Distribution**: ~20% high intent leads (not 99%!)
    - **Realistic Correlation**: Comments correlate with intent but don't determine it
    - **Scalable Architecture**: Can easily swap ML model or reranker independently
    
    ## Data Flow:
    1. Generate structured features
    2. Create target variable using ONLY structured features with realistic distribution
    3. Generate comments that correlate with (but don't determine) target
    4. Train ML model on structured features only
    5. Use reranker to adjust scores based on comments
    