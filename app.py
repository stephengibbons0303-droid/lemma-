import streamlit as st
import pandas as pd
import json
import io
from datetime import datetime
import requests

def call_openai_api(prompt, api_key, max_tokens=4000):
    """Call OpenAI GPT-4o API for lemmatization."""
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            },
            json={
                "model": "gpt-4o",
                "messages": [
                    {
                        "role": "system",
                        "content": """You are a linguistic expert helping with an educational assessment tool. 
Your job is to identify ROOT FORMS (lemmas) and generate MORPHOLOGICAL VARIATIONS for vocabulary words.

CRITICAL CONTEXT:
- This is for a marking correction tool that evaluates student responses
- Students use vocabulary from a provided list in their writing
- Students might use different forms of the same word (e.g., "analyze" vs "analyzing")
- We don't want to penalize students for using valid variations
- We NEVER want to reward students for using synonyms (those are different words)

YOUR TASK:
1. Identify the ROOT/LEMMA (dictionary form) of each word or phrase
2. Generate ALL morphological variations (inflections only)
3. Treat multi-word expressions like "watch out for" as SINGLE LEXICAL UNITS
4. Generate variations of the ENTIRE phrase (e.g., "watching out for", "watched out for")

STRICT RULES:
✅ DO: Provide morphological variations (same root, different form)
✅ DO: Treat phrasal verbs as single units
✅ DO: Include all verb tenses, noun plurals, adjective forms
❌ DON'T: Provide synonyms (e.g., "observe" for "watch")
❌ DON'T: Provide antonyms
❌ DON'T: Provide related words that aren't morphological variations
❌ DON'T: Break up multi-word expressions"""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": max_tokens,
                "temperature": 0.3
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            return data['choices'][0]['message']['content']
        else:
            error_data = response.json()
            st.error(f"API Error {response.status_code}: {error_data.get('error', {}).get('message', 'Unknown error')}")
            return None
    except Exception as e:
        st.error(f"Error calling OpenAI API: {str(e)}")
        return None

def generate_lemmatization_prompt(words_list):
    """Create a prompt for GPT-4o to lemmatize words and generate variations."""
    
    words_str = "\n".join([f"{i+1}. {word}" for i, word in enumerate(words_list)])
    
    prompt = f"""Analyze these vocabulary words/phrases and provide their ROOT FORM and ALL MORPHOLOGICAL VARIATIONS.

VOCABULARY LIST:
{words_str}

REMEMBER:
- For "watch out for" → variations are "watching out for", "watched out for", "watches out for" (NOT "observe", "look out for")
- For "analyze" → variations are "analyzes", "analyzing", "analyzed", "analysing" (NOT "examine", "study")
- These variations help recognize when students use different grammatical forms of the SAME word

OUTPUT FORMAT:
Return ONLY a valid JSON array with no additional text, markdown, or formatting:

[
  {{
    "original": "exact word/phrase from the list",
    "lemma": "root/dictionary form",
    "variations": ["all", "morphological", "variations", "including", "lemma"]
  }}
]

CRITICAL: Response must be ONLY valid JSON. No explanations, no markdown, no code blocks."""

    return prompt

def parse_gpt_response(response_text):
    """Parse GPT's JSON response."""
    try:
        # Remove any markdown code blocks if present
        cleaned = response_text.strip()
        if "```" in cleaned:
            # Extract content between code blocks
            start = cleaned.find("[")
            end = cleaned.rfind("]") + 1
            if start != -1 and end != 0:
                cleaned = cleaned[start:end]
        
        # Parse JSON
        data = json.loads(cleaned)
        return data
    except json.JSONDecodeError as e:
        st.error(f"JSON parsing error: {str(e)}")
        with st.expander("View raw response for debugging"):
            st.code(response_text)
        return None

def process_vocabulary_batch(words_list, api_key, batch_size=50):
    """Process vocabulary in batches of 50 words."""
    results = []
    total_batches = (len(words_list) + batch_size - 1) // batch_size
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(0, len(words_list), batch_size):
        batch = words_list[i:i+batch_size]
        batch_num = i // batch_size + 1
        
        status_text.text(f"Processing batch {batch_num}/{total_batches} ({len(batch)} words)...")
        
        # Generate prompt and call GPT-4o
        prompt = generate_lemmatization_prompt(batch)
        response = call_openai_api(prompt, api_key, max_tokens=4000)
        
        if response:
            parsed_data = parse_gpt_response(response)
            if parsed_data:
                results.extend(parsed_data)
            else:
                st.warning(f"⚠️ Batch {batch_num} failed to parse. Skipping...")
        else:
            st.warning(f"⚠️ Batch {batch_num} failed. Skipping...")
        
        # Update progress
        progress = min((i + batch_size) / len(words_list), 1.0)
        progress_bar.progress(progress)
    
    progress_bar.empty()
    status_text.empty()
    
    return results

def create_results_dataframe(results):
    """Convert results to a pandas DataFrame."""
    df_data = []
    for item in results:
        variations_list = item.get("variations", [])
        df_data.append({
            "Original": item.get("original", ""),
            "Lemma (Root)": item.get("lemma", ""),
            "Variations": ", ".join(variations_list),
            "Number of Variations": len(variations_list)
        })
    return pd.DataFrame(df_data)

# Streamlit App
def main():
    st.set_page_config(page_title="Vocabulary Lemmatization Tool", layout="wide", page_icon="📚")
    
    # Custom CSS
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1f77b4;
            margin-bottom: 0;
        }
        .sub-header {
            font-size: 1.1rem;
            color: #666;
            margin-top: 0;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<p class="main-header">📚 Vocabulary Lemmatization Tool</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Powered by GPT-4o for Educational Assessment</p>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Instructions in expandable section
    with st.expander("ℹ️ About This Tool", expanded=False):
        st.markdown("""
        ### Purpose
        This tool is designed for **educational assessment systems** that need to recognize when students use different 
        forms of vocabulary words without penalizing them.
        
        ### What It Does
        - ✅ **Identifies root forms (lemmas)** of words and phrases
        - ✅ **Generates morphological variations** (e.g., "analyze" → "analyzing", "analyzed", "analyzes")
        - ✅ **Handles multi-word expressions** (e.g., "watch out for" → "watching out for", "watched out for")
        - ✅ **Processes up to 50 words per batch** for efficiency
        
        ### What It Does NOT Do
        - ❌ **No synonyms** - Won't suggest "observe" for "watch" (students must use actual vocabulary words)
        - ❌ **No antonyms** - Won't suggest opposite words
        - ❌ **No semantic relations** - Only morphological variations of the same root
        
        ### Example
        **Vocabulary word:** "analyze"  
        **Student writes:** "I analyzed the data and I'm analyzing the trends"  
        **Result:** ✅ Both "analyzed" and "analyzing" are recognized as variations of "analyze"
        """)
    
    st.markdown("---")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # API Key input
    api_key = st.sidebar.text_input(
        "OpenAI API Key",
        type="password",
        help="Enter your OpenAI API key. Get one at https://platform.openai.com/api-keys"
    )
    
    if not api_key:
        st.sidebar.warning("⚠️ Please enter your OpenAI API key to use this tool")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### 📋 Instructions
    1. Enter your OpenAI API key above
    2. Upload a CSV file with vocabulary words
    3. Select the column containing vocabulary
    4. Click "Generate Lemmas and Variations"
    5. Download the results
    
    ### 📊 Processing
    - **Batch size:** 50 words per batch
    - **Model:** GPT-4o
    - **Context:** Educational assessment
    """)
    
    # File upload
    st.subheader("1️⃣ Upload Your Vocabulary List")
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file with at least one column containing vocabulary words"
    )
    
    if uploaded_file is not None:
        try:
            # Read the CSV
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File uploaded successfully! Found {len(df)} rows.")
            
            with st.expander("📄 Preview Original Data", expanded=True):
                st.dataframe(df.head(20), use_container_width=True)
            
            st.markdown("---")
            
            # Column selection
            st.subheader("2️⃣ Select Vocabulary Column")
            columns = df.columns.tolist()
            selected_column = st.selectbox(
                "Which column contains the vocabulary words?",
                options=columns,
                help="Select the column that contains the words/phrases you want to lemmatize"
            )
            
            # Get words list
            words_list = df[selected_column].dropna().astype(str).str.strip().tolist()
            words_list = [w for w in words_list if w and w.lower() != 'nan']  # Remove empty and NaN strings
            
            if len(words_list) == 0:
                st.error("❌ No valid words found in the selected column!")
            else:
                st.info(f"📊 Found **{len(words_list)} vocabulary items** to process")
                
                # Show first few words
                with st.expander("👀 Preview Vocabulary Items"):
                    preview_count = min(10, len(words_list))
                    for i, word in enumerate(words_list[:preview_count], 1):
                        st.text(f"{i}. {word}")
                    if len(words_list) > preview_count:
                        st.text(f"... and {len(words_list) - preview_count} more")
                
                st.markdown("---")
                
                # Process button
                st.subheader("3️⃣ Generate Lemmas and Variations")
                
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    process_button = st.button(
                        "🚀 Generate Lemmas and Variations",
                        type="primary",
                        disabled=not api_key,
                        use_container_width=True
                    )
                
                with col2:
                    estimated_batches = (len(words_list) + 49) // 50
                    st.metric("Batches", estimated_batches)
                
                with col3:
                    estimated_time = estimated_batches * 5
                    st.metric("Est. Time", f"~{estimated_time}s")
                
                if not api_key:
                    st.warning("⚠️ Please enter your OpenAI API key in the sidebar to continue")
                
                if process_button and api_key:
                    st.markdown("---")
                    st.subheader("🔄 Processing...")
                    
                    with st.spinner("Calling GPT-4o API..."):
                        results = process_vocabulary_batch(words_list, api_key, batch_size=50)
                    
                    if results and len(results) > 0:
                        st.success(f"✅ Successfully processed {len(results)} vocabulary items!")
                        
                        # Create DataFrame
                        results_df = create_results_dataframe(results)
                        
                        st.markdown("---")
                        
                        # Display results
                        st.subheader("📊 Results")
                        
                        # Statistics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Words/Phrases", len(results_df))
                        with col2:
                            avg_variations = results_df["Number of Variations"].mean()
                            st.metric("Avg. Variations", f"{avg_variations:.1f}")
                        with col3:
                            total_variations = results_df["Number of Variations"].sum()
                            st.metric("Total Variations", total_variations)
                        with col4:
                            max_variations = results_df["Number of Variations"].max()
                            st.metric("Max Variations", max_variations)
                        
                        # Results table
                        st.dataframe(
                            results_df,
                            use_container_width=True,
                            height=400
                        )
                        
                        # Download button
                        csv_buffer = io.StringIO()
                        results_df.to_csv(csv_buffer, index=False)
                        csv_data = csv_buffer.getvalue()
                        
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        
                        st.download_button(
                            label="⬇️ Download Results as CSV",
                            data=csv_data,
                            file_name=f"vocabulary_lemmas_{timestamp}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                        
                        # Detailed examples
                        with st.expander("🔍 View Detailed Examples"):
                            st.markdown("### First 5 Results:")
                            for idx, row in results_df.head(5).iterrows():
                                st.markdown(f"""
                                **{idx + 1}. {row['Original']}**
                                - **Lemma (Root):** {row['Lemma (Root)']}
                                - **Variations:** {row['Variations']}
                                - **Count:** {row['Number of Variations']} variations
                                """)
                                st.divider()
                    else:
                        st.error("❌ Failed to process vocabulary. Please check your API key and try again.")
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("Please ensure your CSV file is properly formatted.")
    
    else:
        st.info("👆 Please upload a CSV file to get started")
        
        # Show example
        st.markdown("---")
        st.subheader("💡 Example Usage")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📥 Sample Input CSV:**")
            st.code("""Word
analyze
watch out for
running
beautiful
children
take into account
demonstrate""")
        
        with col2:
            st.markdown("**📤 Expected Output:**")
            st.markdown("""
            - **analyze** → analyzing, analyzed, analyzes, analysing
            - **watch out for** → watching out for, watched out for, watches out for
            - **running** → run, runs, ran, running
            - **beautiful** → beautiful, more beautiful, most beautiful
            - **children** → child, children
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p><strong>Vocabulary Lemmatization Tool</strong> | Powered by GPT-4o</p>
        <p>Designed for educational assessment and vocabulary evaluation</p>
        <p style='font-size: 0.9rem;'>This tool helps recognize morphological variations without rewarding synonyms</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
