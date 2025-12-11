import streamlit as st
import pandas as pd
from rapidfuzz import fuzz, process
import jellyfish

# --- 1. EXPANDED DATASET ---
# This list includes original names, variations, misspellings, and compounds.
names = [
    # Female Indian Names
    "Geetha","Geeta","Gita","Gitu","Geethaa","Githa","Githaa","Geetha Devi","Sahana","Sahani",
    "Sanjana","Sneha","Snehal","Priya","Priyanka","Priti","Preeti","Preethi","Rekha","Reka", 
    "Kiran","Kiranmai","Shreya","Shriya","Nisha","Anjali","Anjalee","Anjali Devi","Shyla","Nisha Devi",
    "Rekhaa","Sahanaa","Sanjanaa","Sneha Devi","Kiranthi","Shiya","Nishaaya","Aishwarya","Deepika","Madhuri",
    "Kavya","Divya","Radhika","Lakshmi","Meera","Radha","Chitra","Bindu","Seema","Pallavi",
    "Priyanaka", "Preeti K", "Rithu", "Ritu", "Ritika", "Pooja", "Puja", "Jaya", "Vandana", "Swati",
    
    # Male Indian Names
    "Ramesh","Suresh","Rajesh","Mahesh","Ganesh","Dinesh","Naresh","Kiran Kumar","Ravi","Arjun",
    "Krishna","Vishal","Ajay","Vijay","Santosh","Anil","Sunil","Rahul","Amit","Ashok",
    "Shankar","Mohan","Prakash","Sanjay","Lokesh","Vinod","Chandresh","Harish","Naveen","Karthik",
    "Ramesh P", "Sures", "Vishaal", "Santhosh", "Amith", "Ashok Kumar", "Parveen", "Gopal Varma", 
    
    # English/Global Names
    "Sophia","Olivia","Isabella","Ava","Mia","Amelia","Harper","Evelyn","Abigail","Ella",
    "Liam","Noah","Ethan","James","Benjamin","Lucas","Henry","Alexander","William","Michael",
    "Sophie", "Olyvia", "Izabella", "Mya", "Amelia Rose", "Elah", "Emilie", "Scarlet", "Grayce", 
    "Lian", "Noha", "Ethan P", "Jamie", "Benjiman", "Lukas", "Henrey", "Alexandar",
    
    # Compound and Title Names
    "Dr. Geetha", "Prakash Singh", "Ms. Sneha", "Mr. Anil", "Lakshmi G.", "Rahul Sharma", 
    "Chitra Devi", "Priya K.", "Smith", "Smth", "Smyth", "John Smith"
]

# Pre-calculate Soundex codes for the entire dataset for faster phonetic checks
name_soundex_map = {name: jellyfish.soundex(name) for name in names}

# --- 2. ACCURATE HYBRID MATCHING FUNCTION ---
def get_hybrid_matches(query, name_list, top_n=10):
    """
    Combines rapidfuzz scoring with a Soundex phonetic bonus for better accuracy 
    on misspelled or phonetically similar names.
    """
    
    # 1. Base Score: Use WRatio for initial strong string matching
    # Get a large number of potential matches to include names that score lower initially
    matches = process.extract(
        query, 
        name_list, 
        scorer=fuzz.WRatio, 
        limit=len(name_list) 
    )
    
    # Convert to DataFrame
    df = pd.DataFrame(matches, columns=['Matched Name', 'WRatio Score', 'Index'])
    
    # Filter out the query name itself and duplicates
    df = df[df['Matched Name'].str.lower() != query.lower()]
    df = df.drop_duplicates(subset=['Matched Name']).reset_index(drop=True)

    # Calculate query's Soundex code
    query_soundex = jellyfish.soundex(query)
    
    # 2. Phonetic Score Bonus
    def calculate_bonus_score(row):
        # 1. Calculate phonetic match score (0 or 100)
        soundex_match = 100 if query_soundex == name_soundex_map.get(row['Matched Name']) else 0
        
        # 2. Add a weighted bonus (e.g., up to 10 points) if the names sound alike.
        # This rewards phonetically similar names that might have a lower string score.
        phonetic_bonus = soundex_match * 0.10  # 10% weight
        
        # 3. Create a final 'Hybrid Score'
        hybrid_score = row['WRatio Score'] + phonetic_bonus
        return min(100.0, hybrid_score) # Cap at 100

    df['Hybrid Score'] = df.apply(calculate_bonus_score, axis=1)
    
    # 3. Final Ranking
    df = df.sort_values(by='Hybrid Score', ascending=False).head(top_n)

    # Prepare final output
    if df.empty:
        return None, pd.DataFrame()

    best_match = (df.iloc[0]['Matched Name'], df.iloc[0]['Hybrid Score'])
    
    return best_match, df[['Matched Name', 'Hybrid Score', 'WRatio Score']]

# --- 3. STREAMLIT UI (User Interface) ---
st.set_page_config(page_title="Accurate Name Matcher", layout="wide")

st.title("🚀 Accurate Hybrid Name Matching System")
st.markdown("""
This tool uses a **Hybrid Scoring** system for high accuracy: 
**`WRatio (rapidfuzz)`** for general spelling + a **`Soundex` phonetic bonus** for names that sound alike.
""")

# Sidebar for configuration
st.sidebar.header("Configuration")
top_n_choice = st.sidebar.slider(
    "Number of Matches to Show:",
    min_value=5, max_value=20, value=10, step=5
)
st.sidebar.info("Try searching for 'Githa' (should match 'Geetha') or 'Lyam' (should match 'Liam')")

# Main input area
query_name = st.text_input("Enter a Name to Search For:", help="Input can be a full name or a misspelling.")

if st.button("Find Matches", type="primary"):
    if not query_name.strip():
        st.error("🛑 Please enter a name to search.")
    else:
        with st.spinner(f"Searching for matches for **{query_name}**..."):
            # NOTE: We assume get_hybrid_matches is defined correctly above this block
            best, ranked_df = get_hybrid_matches(query_name, names, top_n_choice)

        st.markdown("---")
        
        if best:
            st.success(f"### ✅ Best Hybrid Match Found: **{best[0]}**")
            st.subheader(f"Final Hybrid Score: {best[1]:.2f}/100")
            
            st.markdown("---")
            st.subheader(f"📌 Top {top_n_choice} Ranked Matches (Hybrid Score)")
            
            # Format the scores for display
            ranked_df['Hybrid Score'] = ranked_df['Hybrid Score'].apply(lambda x: f"{x:.2f}")
            ranked_df['WRatio Score'] = ranked_df['WRatio Score'].apply(lambda x: f"{x:.2f}")

            ranked_df.columns = ['Matched Name', 'Hybrid Score (Final)', 'WRatio Score (Base)']
            ranked_df.index = ranked_df.index + 1 # Start index from 1
            
            # FIX: Using a standard st.dataframe call without column_config
            st.dataframe(
                ranked_df, 
                use_container_width=True
            )
        else:
            st.warning("No similar names found in the dataset.")