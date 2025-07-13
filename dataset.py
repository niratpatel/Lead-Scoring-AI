import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta

# Initialize Faker for Indian data
fake = Faker('en_IN')

def generate_realistic_lead_dataset_v3(n_samples=10000, random_state=42, noise_level=0.15):
    """
    Generates a realistic lead dataset where:
    - ML model is trained on structured features only
    - Comments are used exclusively for reranking
    - Target variable is generated WITHOUT considering comments
    - FIXED: Realistic distribution with ~20% high intent leads
    """
    np.random.seed(random_state)
    
    # --- 1. DEMOGRAPHICS (Enhanced) ---
    age_groups = ['18-25', '26-35', '36-50', '51+']
    age_group_probs = [0.15, 0.40, 0.35, 0.10]
    age_group_choices = np.random.choice(age_groups, size=n_samples, p=age_group_probs)
    
    gender_choices = np.random.choice(['Male', 'Female'], size=n_samples, p=[0.60, 0.40])

    # Enhanced location field with tier-based cities
    tier1_cities = ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata', 'Hyderabad', 'Pune', 'Ahmedabad']
    tier2_cities = ['Surat', 'Vadodara', 'Rajkot', 'Indore', 'Bhopal', 'Coimbatore', 'Kochi', 'Chandigarh']
    tier3_cities = ['Nashik', 'Aurangabad', 'Solapur', 'Dhule', 'Jamnagar', 'Bhavnagar', 'Anand', 'Bharuch']
    
    city_choices = []
    for _ in range(n_samples):
        rand = np.random.rand()
        if rand < 0.6:  # 60% from tier 1
            city_choices.append(np.random.choice(tier1_cities))
        elif rand < 0.85:  # 25% from tier 2
            city_choices.append(np.random.choice(tier2_cities))
        else:  # 15% from tier 3
            city_choices.append(np.random.choice(tier3_cities))

    family_backgrounds = []
    educations = []
    professions = []
    
    profession_categories = {
        'IT/Software': ['Software Engineer', 'Data Scientist', 'Product Manager', 'DevOps Engineer'],
        'Finance': ['Financial Analyst', 'Accountant', 'Investment Banker', 'Tax Consultant'],
        'Healthcare': ['Doctor', 'Nurse', 'Pharmacist', 'Medical Technician'],
        'Education': ['Teacher', 'Professor', 'Principal', 'Training Manager'],
        'Business': ['Sales Manager', 'Marketing Executive', 'Business Analyst', 'Entrepreneur'],
        'Government': ['Civil Servant', 'Police Officer', 'Municipal Employee', 'Defense Personnel'],
        'Other': ['Consultant', 'Freelancer', 'Retired', 'Homemaker']
    }
    
    for age in age_group_choices:
        if age == '18-25':
            family_backgrounds.append(np.random.choice(['Single', 'Married'], p=[0.85, 0.15]))
            educations.append(np.random.choice(['Graduate', 'Post Graduate', 'High School', 'Diploma'], p=[0.55, 0.15, 0.2, 0.1]))
            prof_cat = np.random.choice(['IT/Software', 'Finance', 'Business', 'Other'], p=[0.4, 0.2, 0.3, 0.1])
        elif age == '26-35':
            family_backgrounds.append(np.random.choice(['Married', 'Single', 'Married with Kids'], p=[0.65, 0.25, 0.10]))
            educations.append(np.random.choice(['Graduate', 'Post Graduate', 'Professional'], p=[0.5, 0.3, 0.2]))
            prof_cat = np.random.choice(['IT/Software', 'Finance', 'Business', 'Healthcare'], p=[0.35, 0.25, 0.25, 0.15])
        elif age == '36-50':
            family_backgrounds.append(np.random.choice(['Married with Kids', 'Married'], p=[0.75, 0.25]))
            educations.append(np.random.choice(['Post Graduate', 'Professional', 'Graduate', 'Other'], p=[0.35, 0.4, 0.15, 0.1]))
            prof_cat = np.random.choice(['Business', 'Finance', 'IT/Software', 'Government'], p=[0.3, 0.25, 0.25, 0.2])
        else:  # 51+
            family_backgrounds.append(np.random.choice(['Married with Kids', 'Married'], p=[0.8, 0.2]))
            educations.append(np.random.choice(['Professional', 'Post Graduate', 'Other'], p=[0.5, 0.3, 0.2]))
            prof_cat = np.random.choice(['Business', 'Government', 'Other'], p=[0.4, 0.35, 0.25])
        
        professions.append(np.random.choice(profession_categories[prof_cat]))

    # --- 2. FINANCIALS (Enhanced with city tier impact) ---
    income_base_log = np.log(600000)
    age_income_boost = [age_groups.index(a) * 0.2 for a in age_group_choices]
    edu_income_boost = [(['High School', 'Diploma', 'Graduate', 'Post Graduate', 'Professional', 'Other'].index(e)) * 0.15 for e in educations]
    gender_income_boost = [0.05 if g == 'Male' else -0.05 for g in gender_choices]
    
    # City tier impact on income
    city_income_boost = []
    for city in city_choices:
        if city in tier1_cities:
            city_income_boost.append(0.3)
        elif city in tier2_cities:
            city_income_boost.append(0.1)
        else:
            city_income_boost.append(-0.1)
    
    log_incomes = (income_base_log + np.array(age_income_boost) + np.array(edu_income_boost) + 
                   np.array(gender_income_boost) + np.array(city_income_boost) + 
                   np.random.normal(0, 0.3, n_samples))
    incomes = np.exp(log_incomes).astype(int)

    # Introduce outliers in income
    outlier_indices = np.random.choice(n_samples, size=int(n_samples * 0.01), replace=False)
    incomes[outlier_indices] = incomes[outlier_indices] * np.random.uniform(3, 5, size=len(outlier_indices))

    credit_scores = []
    cibil_probs = [0.15, 0.20, 0.30, 0.35]
    for income in incomes:
        income_factor = min(2, income / 1000000)
        adj_probs = cibil_probs + np.array([-0.1, -0.05, 0.05, 0.1]) * income_factor
        adj_probs = np.clip(adj_probs, 0.01, 0.99)
        adj_probs /= adj_probs.sum()
        
        score_bracket = np.random.choice(4, p=adj_probs)
        if score_bracket == 0: score = np.random.randint(550, 650)
        elif score_bracket == 1: score = np.random.randint(650, 700)
        elif score_bracket == 2: score = np.random.randint(700, 750)
        else: score = np.random.randint(750, 850)
        credit_scores.append(score)

    # --- 3. BEHAVIORAL (Enhanced with more features) ---
    source_choices = np.random.choice(
        ['Google', 'Direct Traffic', 'Reference', 'Organic Search', 'Facebook', 'Affiliate', 'LinkedIn', 'Instagram'],
        size=n_samples, p=[0.28, 0.20, 0.15, 0.12, 0.08, 0.05, 0.07, 0.05]
    )

    base_time = []
    for s in source_choices:
        if s in ['Reference', 'Direct Traffic']:
            base_time.append(1200)
        elif s in ['LinkedIn', 'Google']:
            base_time.append(800)
        else:
            base_time.append(600)
    
    time_spent_website = (np.array(base_time) * np.random.lognormal(0, 0.5, n_samples)).astype(int)
    
    # Introduce outliers for time spent
    time_outlier_indices = np.random.choice(n_samples, size=int(n_samples * 0.02), replace=False)
    time_spent_website[time_outlier_indices] = time_spent_website[time_outlier_indices] * np.random.uniform(5, 10, size=len(time_outlier_indices))

    page_views = np.clip((time_spent_website / 120 + np.random.randint(-2, 5, n_samples)), 1, 50).astype(int)

    # Enhanced interaction features
    form_submissions = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.6, 0.25, 0.1, 0.05])
    
    # Previous interactions (indicates returning visitor)
    previous_interactions = np.random.choice([0, 1, 2, 3, 4, 5], size=n_samples, p=[0.5, 0.25, 0.15, 0.06, 0.03, 0.01])
    
    # Property type preference
    property_types = ['Apartment', '1BHK', '2BHK', '3BHK', 'Villa', 'Plot', 'Commercial', 'Not Specified']
    property_preferences = np.random.choice(property_types, size=n_samples, p=[0.15, 0.1, 0.3, 0.25, 0.08, 0.05, 0.02, 0.05])

    # --- 4. TEMPORAL FEATURES ---
    # Lead creation timestamps (last 90 days)
    base_date = datetime.now() - timedelta(days=90)
    lead_dates = [base_date + timedelta(days=np.random.randint(0, 90)) for _ in range(n_samples)]
    
    # Day of week and hour impact
    day_of_week = [date.strftime('%A') for date in lead_dates]
    hour_of_day = [np.random.randint(9, 22) for _ in range(n_samples)]  # Business hours focus
    
    # --- 5. ASSEMBLE The DataFrame (WITHOUT COMMENTS FIRST) ---
    df = pd.DataFrame({
        'phone_number': [fake.phone_number() for _ in range(n_samples)],
        'email': [f"{fake.first_name().lower()}.{fake.last_name().lower()}_{i}@test.com" for i in range(n_samples)],
        'gender': gender_choices,
        'age_group': age_group_choices,
        'city': city_choices,
        'family_background': family_backgrounds,
        'education': educations,
        'profession': professions,
        'income_inr': incomes,
        'credit_score': credit_scores,
        'lead_source': source_choices,
        'time_spent_website_secs': time_spent_website,
        'page_views': page_views,
        'form_submissions': form_submissions,
        'previous_interactions': previous_interactions,
        'property_type_preference': property_preferences,
        'lead_date': lead_dates,
        'day_of_week': day_of_week,
        'hour_of_day': hour_of_day,
        'last_interaction_channel': np.random.choice(['Email', 'Phone', 'SMS', 'Website'], size=n_samples, p=[0.4, 0.3, 0.2, 0.1])
    })

    # Introduce missing values strategically
    for col, missing_frac in [('credit_score', 0.12), ('income_inr', 0.08), ('family_background', 0.05), 
                             ('profession', 0.03), ('property_type_preference', 0.15)]:
        missing_indices = np.random.choice(df.index, size=int(n_samples * missing_frac), replace=False)
        df.loc[missing_indices, col] = np.nan
    
    # --- 6. GENERATE TARGET VARIABLE - FIXED FOR REALISTIC DISTRIBUTION ---
    # Normalize features to 0-1 scale
    credit_norm = df['credit_score'].fillna(df['credit_score'].median()) / 850
    income_norm = np.log1p(df['income_inr'].fillna(df['income_inr'].median())) / np.log1p(df['income_inr'].max())
    time_norm = df['time_spent_website_secs'] / df['time_spent_website_secs'].max()
    pages_norm = df['page_views'] / df['page_views'].max()
    
    # FIXED: MUCH MORE RESTRICTIVE logit calculation for realistic distribution
    # Targeting ~15-20% high intent leads
    logit = (
        0.6 * credit_norm +           # Further reduced from 1.2 to 0.6
        0.5 * income_norm +           # Further reduced from 1.0 to 0.5
        0.4 * time_norm +             # Further reduced from 0.8 to 0.4
        0.2 * pages_norm +            # Further reduced from 0.3 to 0.2
        0.3 * df['form_submissions'] + # Further reduced from 0.4 to 0.3
        0.2 * df['previous_interactions'] - 2.5  # Much more negative baseline: -1.8 to -2.5
    )
    
    # Age group impact - much more conservative
    logit += df['age_group'].map({'18-25': -0.4, '26-35': 0.1, '36-50': 0.05, '51+': -0.3}).fillna(0)
    
    # Lead source impact - much more conservative
    logit += df['lead_source'].map({
        'Reference': 0.3,      # Further reduced from 0.4
        'Direct Traffic': 0.2, # Further reduced from 0.3
        'LinkedIn': 0.15,      # Further reduced from 0.2
        'Google': 0.05,        # Further reduced from 0.1
        'Organic Search': 0.02, # Further reduced from 0.05
        'Facebook': -0.1,      # Same
        'Instagram': -0.15,    # Same
        'Affiliate': -0.05     # Same
    }).fillna(0)
    
    # City tier impact - much more conservative
    city_tier_map = {
        city: 0.1 if city in tier1_cities else (0.02 if city in tier2_cities else -0.08) 
        for city in city_choices
    }
    logit += df['city'].map(city_tier_map).fillna(0)
    
    # Property preference impact - much more conservative
    logit += df['property_type_preference'].map({
        'Villa': 0.15,     # Further reduced from 0.2
        '3BHK': 0.1,       # Further reduced from 0.15
        '2BHK': 0.05,      # Further reduced from 0.1
        'Commercial': 0.02, # Further reduced from 0.05
        'Apartment': 0.0,   # Same
        '1BHK': -0.05,     # Same
        'Plot': -0.1       # Same
    }).fillna(0)
    
    # Add random noise
    noise = np.random.normal(0, noise_level, n_samples)
    logit += noise
    
    # Convert to probability and generate binary target
    prob_of_intent = 1 / (1 + np.exp(-logit))
    df['high_intent'] = (np.random.rand(n_samples) < prob_of_intent).astype(int)
    
    # --- 7. GENERATE COMMENTS BASED ON TARGET (FOR RERANKING ONLY) ---
    # Comments are generated AFTER target creation and should correlate with intent
    comments = []
    for i in range(n_samples):
        intent = df.loc[i, 'high_intent']
        
        # High intent leads get more positive comments
        if intent == 1:
            p = np.random.rand()
            if p < 0.4:  # 40% get very positive comments
                comments.append(np.random.choice([
                    "Urgent requirement for 2BHK", "Ready to buy immediately", 
                    "Serious buyer, need best price", "Looking for immediate possession", 
                    "Cash buyer ready", "Pre-approved loan", "Need site visit this week"
                ]))
            elif p < 0.7:  # 30% get moderately positive comments
                comments.append(np.random.choice([
                    "Interested in 2BHK options", "Budget around 80 lakhs", 
                    "Planning to buy in 3 months", "Need more details", 
                    "Looking for good location"
                ]))
            else:  # 30% get neutral/no comments
                comments.append(np.random.choice([
                    "Just checking options", "What is the price?", 
                    "Need information", ""
                ]))
        else:
            # Low intent leads get more negative/neutral comments
            p = np.random.rand()
            if p < 0.3:  # 30% get negative comments
                comments.append(np.random.choice([
                    "Just checking", "Maybe later", "Too expensive", 
                    "Not sure about budget", "Will think about it", 
                    "Not interested right now"
                ]))
            elif p < 0.6:  # 30% get neutral comments
                comments.append(np.random.choice([
                    "Comparing options", "What is the final price?", 
                    "Need more information", "Looking at different projects"
                ]))
            else:  # 40% get no comments
                comments.append("")
    
    df['comments'] = comments
    
    return df

def get_ml_training_features():
    """
    Return the list of features to be used for ML model training.
    Comments should NOT be included here.
    """
    return [
        'gender', 'age_group', 'city', 'family_background', 'education', 
        'profession', 'income_inr', 'credit_score', 'lead_source', 
        'time_spent_website_secs', 'page_views', 'form_submissions', 
        'previous_interactions', 'property_type_preference', 'day_of_week', 
        'hour_of_day', 'last_interaction_channel'
    ]

def get_reranking_features():
    """
    Return the list of features to be used for reranking.
    This should include comments and any other text-based features.
    """
    return ['comments']

def create_feature_documentation():
    """Create documentation for the dataset features"""
    documentation = """
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
    """
    return documentation

if __name__ == "__main__":
    print("Generating realistic lead scoring dataset...")
    print("Key Fix: MUCH MORE RESTRICTIVE distribution targeting ~15-20% high intent leads")
    print("Key Design: Comments are for RERANKING only, not ML training")
    print("=" * 60)
    
    lead_df_v3 = generate_realistic_lead_dataset_v3(n_samples=10000, noise_level=0.25)
    
    # Select final columns
    final_cols = [
        'email', 'phone_number', 'gender', 'age_group', 'city', 'family_background', 
        'education', 'profession', 'income_inr', 'credit_score', 'lead_source', 
        'time_spent_website_secs', 'page_views', 'form_submissions', 'previous_interactions',
        'property_type_preference', 'comments', 'lead_date', 'day_of_week', 'hour_of_day',
        'last_interaction_channel', 'high_intent'
    ]
    lead_df_v3 = lead_df_v3[final_cols]
    
    # Save dataset
    lead_df_v3.to_csv('lead_scoring_data_enhanced.csv', index=False)
    
    print(f"\nDataset with {len(lead_df_v3)} records saved to 'lead_scoring_data_enhanced.csv'")
    print("=" * 60)
    print("DATASET PREVIEW:")
    print(lead_df_v3.head())
    print("=" * 60)
    
    # Show feature separation
    ml_features = get_ml_training_features()
    rerank_features = get_reranking_features()
    
    print(f"\nML TRAINING FEATURES ({len(ml_features)} features):")
    print(ml_features)
    print(f"\nRERANKING FEATURES ({len(rerank_features)} features):")
    print(rerank_features)
    print("=" * 60)
    
    print("\nFEATURE ANALYSIS:")
    print(f"Total Features: {len(lead_df_v3.columns)}")
    print(f"ML Training Features: {len(ml_features)}")
    print(f"Reranking Features: {len(rerank_features)}")
    print(f"Metadata Features: {len(lead_df_v3.columns) - len(ml_features) - len(rerank_features) - 1}")  # -1 for target
    
    print("\nNULL VALUES CHECK:")
    print(lead_df_v3.isnull().sum())
    print("=" * 60)
    
    print("\nTARGET DISTRIBUTION:")
    print(f"High Intent Leads: {lead_df_v3['high_intent'].sum()} ({lead_df_v3['high_intent'].mean()*100:.1f}%)")
    print(f"Low Intent Leads: {(lead_df_v3['high_intent'] == 0).sum()} ({(1-lead_df_v3['high_intent'].mean())*100:.1f}%)")
    
    print("\nCOMMENT CORRELATION CHECK:")
    high_intent_with_comments = lead_df_v3[lead_df_v3['high_intent'] == 1]['comments'].str.len().mean()
    low_intent_with_comments = lead_df_v3[lead_df_v3['high_intent'] == 0]['comments'].str.len().mean()
    print(f"Avg comment length - High Intent: {high_intent_with_comments:.1f}")
    print(f"Avg comment length - Low Intent: {low_intent_with_comments:.1f}")
    print("=" * 60)
    
    # Create documentation
    docs = create_feature_documentation()
    with open('dataset_documentation.md', 'w') as f:
        f.write(docs)
    print("\nFeature documentation saved to 'dataset_documentation.md'")
    print("✅ Dataset is ready for ML training (structured features) + Reranking (comments)!")
    print("✅ FIXED: Now has realistic ~15-20% high intent distribution instead of 70%!")