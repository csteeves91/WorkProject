# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import random
import time
import sys
import os
from dotenv import load_dotenv
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import urllib.request
#######################################
# Setup
#######################################

start_time = time.time()

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

fileName = (os.path.basename(__file__).split('.')[0])
csvName = f"{fileName}.csv"

origin_csv = (f"C:/Users/csteeves/Documents/myscripts/myscripts/Conversational Analyses/Individual Process/Seller NPS.csv")

destination_csv = (f"C:/Users/csteeves/Documents/myscripts/myscripts/Conversational Analyses/Individual Process/Resources/AI Processed.csv")

usecols = ["Index", "Answer"]
primary_column = "Answer"
index_column = "Index"

input_datapath = origin_csv
df = pd.read_csv(input_datapath, index_col=index_column, usecols=usecols, dtype=str)
df = df.fillna("None")
csv_length = int(len(df))

model = "gpt-3.5-turbo"
tokensAllowed = 2000
creativity = 0.7
seed = 12

if model == "gpt-3.5-turbo":
    in_cost = .5
    out_cost = 1.5
elif model == "gpt-4-turbo":
    in_cost = 10
    out_cost = 30
elif model == "gpt-4o":
    in_cost = 5
    out_cost = 15
else:
    in_cost = 0
    out_cost = 0

cost = []

#######################################
# Processing Functions
#######################################

def format_error(e):
    try:
        error_response = e.response.json()  # Assuming e has a response attribute with a json() method
        error_code = error_response.get('error', {}).get('code', 'N/A')
        error_message = error_response.get('error', {}).get('message', 'No error message provided')
        error_type = error_response.get('error', {}).get('type', 'N/A')
        error_param = error_response.get('error', {}).get('param', 'N/A')

        formatted_message = (
            f"\nOpenAI returned an API Error:\n\n"
            f"Error Code:{error_code}\n"
            f"Error Message:{error_message}\n"
            f"Error Type:{error_type}\n"
            f"Error Parameter:{error_param}\n"
        )
        return formatted_message
    except Exception as parse_error:
        return f"Failed to parse error response: {parse_error}"

def ai_process(string_to_process):
    # prompt = f"""
    #     <your task>
    #     </your task>

    #     <customer quote>
    #         {string_to_process}
    #     </customer quote>

    #     <rules & notes>
    #     </rules & notes>

    #     <good examples>
    #     </good examples>

    #     <bad examples> 
    #     </bad examples>
    # """

    prompt = f"""
    <your task>
        Note which of the "themes of interest" the customer quote relates to.
    </your task>

    <customer quote>
        {string_to_process}
    </customer quote>

    <themes of interest>
        - Search Experience
        - Discovering New Cards
        - Product Availability
        - Experience with Sellers
        - Price of Products
        - Customer Support
        - TCGplayer Direct
        - Site Navigation
        - Shipping Costs
        - Shipping Time
        - Other Shipping
        - Cart Optimization
        - Pricing Trends and Historical Data
        - Price Manipulation and Gouging
        - Order Tracking and Delivery Assurance
        - International Availability and Support
        - Trust and Reliability
        - Community Impact
        - Support for Local Game Stores (LGS)
        - Transparency and Communication
        - Product Condition
        - Convenience
        - Union and Employee Relations
        - Changes to Policies/Platform
        - Seller Fees
        - Cart Experience
        - Checkout Experience

    </themes of interest>
    
    <rules & notes>
        Rules:
        1. Do not make anything up.
        2. Only use the themes listed in <themes of interest>, do not create additional themes.
        3. Only include themes that are related to the <customer quote>.
        4. Reply only with related themes, delineated by a pipe character | . Nothing else.
        5. If the <customer quote> does not align with any of the <themes of interest>, or it is a one word answer, simply reply "Skipped".
        6. Don't include the <customer quote> in your reply.
        7. Review your reply before sending to ensure it follows all of the above rules.
    </rules & notes>

    <good examples>
        Quote: "We are not given the option to get a tracking number for our orders with sellers if the order is less than $50"
        Reply: Order Tracking and Delivery Assurance
        --
        Quote: "It is quite easy to search for products, the prices are better than my local game stores, and typically have more product than them as well."
        Reply: Search Experience | Price of Products | Product Availability | Convenience
    </good examples>

    <bad examples>
        Quote: "I love almost everything about tcgplayer, I am on the site at least 5x a day."
        Reply: Convenience
        
        This reply is bad because the quote does not directly relate to convenience.
        --
        Quote: "when items are out of stock, sellers dont make it right by getting me the card, i just want it replaced so i know its coming, i dont want to pay 5$ more for shipping on something i was already getting, just send me the out of stock card"
        Reply: Product Availability | Experience with Sellers | Shipping Costs
        
        This reply is bad because "Shipping Costs" does not belong, since the point of the feedback is not about shipping costs. Only "Product Availability" and "Experience with Sellers" belong in this reply. 
    </bad examples>
"""

    # random_time = random.uniform(61, 71)
    # time.sleep(random_time)

    response = client.chat.completions.create(
        model=model,
        temperature=creativity,
        max_tokens=tokensAllowed,
        seed=seed,
        messages=[
            {"role": "system", "content": "You are a highly skilled AI trained in qualitative data analysis. You are currently helping the User Experience team that works on TCGplayer.com. You follow instructions closely."},
            {"role": "assistant", "content": "Hi there! What are my instructions?"},
            {"role": "user", "content": prompt},
        ])
    
    inputCost = ((in_cost*response.usage.prompt_tokens)/1000000)
    outputCost = ((out_cost*response.usage.completion_tokens)/1000000)

    cost.append(inputCost)
    cost.append(outputCost)

    return response.choices[0].message.content.strip()

def process_dataframe_slices(slice_of_df):
    failed_indices = []
    analyses = []

    for idx, row in slice_of_df.iterrows():
        try:
            analysis = ai_process(row[primary_column])
            analyses.append(analysis)
        except Exception as e:
            print(format_error(e))
            failed_indices.append(idx)
            analyses.append(None)

    slice_of_df['Analysis'] = analyses
    return slice_of_df, failed_indices

def retry_failed_entries(failed_entries):
    retried_entries = []
    for idx, string_to_process in failed_entries:
        try:
            analysis = ai_process(string_to_process)
            retried_entries.append((idx, analysis))
        except Exception as e:
            print(f"Error reprocessing index {idx}: {e}")
            retried_entries.append((idx, None))
    return retried_entries

count = 0

#######################################
# Main Execution Block
#######################################

print_lock = threading.Lock()

if __name__ == '__main__':
    num_workers = 200

    chunks = np.array_split(df, num_workers)
    results = []
    all_failed_entries = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures_to_chunk = {executor.submit(process_dataframe_slices, chunk): chunk for chunk in chunks}
        
        # Process futures as they complete
        for future in as_completed(futures_to_chunk):
            result, failed_indices = future.result()
            results.append(result)
            all_failed_entries.extend(result.loc[failed_indices, primary_column].items())
            with print_lock:
                count += 1
                print(f"Working Job {count} of {csv_length}", end="\r")
                sys.stdout.flush()

    # Concatenate results back to a single DataFrame
    df_result = pd.concat(results).sort_values(by=index_column)

    # Retry failed entries if any
    if all_failed_entries:
        print(f"\n\n{len(all_failed_entries)} entries failed initially.")
        retry_now = input("Would you like to retry failed entries now? (y/n): ").strip().lower()
        if retry_now == 'y':
            retried_entries = retry_failed_entries(all_failed_entries)
            for idx, analysis in retried_entries:
                if analysis is not None:
                    df_result.at[idx, 'Analysis'] = analysis

    # Save the DataFrame to CSV
    df_result.to_csv(f"C:/Users/csteeves/Documents/myscripts/myscripts/Conversational Analyses/Individual Process/Seller NPS.csv")

    # Execution Calc
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Ending Report
    totalCost = sum(cost)
    print(f"\n\n\nComplete!")
    print(f"\nModel Used: {model}")
    print(f"\nRequests Processed: {csv_length}")
    print(f"\nTotal Cost: ${round(totalCost, 2)}")
    print(f"\nExecution Time: {round(execution_time, 2)}s\n\n")